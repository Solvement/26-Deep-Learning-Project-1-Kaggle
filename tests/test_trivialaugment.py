"""Small PIL/NumPy pixels and explicit RNG only; no Torch/model/data/GPU."""
import ast
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from research import trivialaugment as ta


class TrivialAugmentTests(unittest.TestCase):
    def pattern(self):
        row,col=np.indices((32,32))
        return np.stack(((row*31+col*7)%256,(row*3+col*47)%256,(row*13+col*11)%256),axis=-1).astype(np.uint8)

    def test_pinned_author_attribution_and_original_MIT_permission_are_preserved(self):
        self.assertEqual(ta.AUTHOR_COMMIT,"e6545d80affcc18b7024185a096e93e21010023e")
        self.assertEqual(ta.AUTHOR_SOURCE_SHA256,"8f5f9febb671fd12be067e311dbdcb837a2a29c672861f20c2cf03732fdd2868")
        license_path=Path(ta.__file__).parent/'licenses/TrivialAugment-MIT.txt'
        self.assertEqual(hashlib.sha256(license_path.read_bytes()).hexdigest(),"b97576d1345e463979111dd49bffea6cd5bbe010b7919fbfb514f3478e6c5835")

    def test_exact_fixed_space_and_all31_integer_quantizations(self):
        self.assertEqual(ta.OPERATIONS,("identity","autocontrast","equalize","rotate","solarize","color","posterize",
                                      "contrast","brightness","sharpness","shear_x","shear_y","translate_x","translate_y"))
        self.assertEqual((ta.SPACE,ta.STRENGTHS),("fixed_standard",31))
        for level in range(31):
            for name in ta.OPERATIONS:
                value=ta.magnitude(ta.Operation(name,level))
                if name in ("identity","autocontrast","equalize"):self.assertIsNone(value)
                elif name=="posterize":self.assertEqual(value,8-(4*level//30))
                elif name=="solarize":self.assertEqual(value,256-(256*level//30))
                elif name.startswith("translate_"):self.assertEqual(value,10*level//30)
                elif name=="rotate":self.assertEqual(value,level)
                elif name.startswith("shear_"):self.assertAlmostEqual(value,.01*level,places=14)
                else:self.assertAlmostEqual(value,.1+.06*level,places=14)
        self.assertEqual([ta.magnitude(ta.Operation("posterize",x)) for x in (0,7,8,14,15,22,23,29,30)],[8,8,7,7,6,6,5,5,4])

    def test_every_operation_level_and_geometric_sign_preserves_RGB_uint8_and_input(self):
        original=self.pattern(); original.flags.writeable=False
        for name in ta.OPERATIONS:
            for level in range(31):
                for sign in ((-1,1) if name in ta.SIGNED else (1,)):
                    with self.subTest(name=name,level=level,sign=sign):
                        before=original.tobytes(); output=ta.apply_operation(original,ta.Operation(name,level,sign))
                        pixels=np.asarray(output)
                        self.assertEqual((output.mode,output.size),("RGB",(32,32)))
                        self.assertEqual((pixels.dtype,pixels.shape),(np.uint8,(32,32,3)))
                        self.assertEqual(original.tobytes(),before)
                        self.assertTrue(np.isfinite(pixels).all())

    def test_posterization_corrected_endpoints_and_solarize_threshold_direction(self):
        source=self.pattern()
        self.assertTrue(np.array_equal(ta.apply_operation(source,ta.Operation("posterize",0)),source))
        maximum=np.asarray(ta.apply_operation(source,ta.Operation("posterize",30)))
        self.assertTrue(np.array_equal(maximum,source & np.uint8(0xf0)))
        self.assertGreater(maximum.max(),0)
        self.assertTrue(np.array_equal(ta.apply_operation(source,ta.Operation("solarize",0)),source))
        self.assertTrue(np.array_equal(ta.apply_operation(source,ta.Operation("solarize",30)),255-source))
        threshold=ta.magnitude(ta.Operation("solarize",15))
        expected=np.where(source<threshold,source,255-source).astype(np.uint8)
        self.assertTrue(np.array_equal(ta.apply_operation(source,ta.Operation("solarize",15)),expected))

    def test_translation_direction_and_exact_black_fill(self):
        source=np.empty((32,32,3),dtype=np.uint8);source[:]=(31,73,191)
        for axis,name in ((1,"translate_x"),(0,"translate_y")):
            for sign in (-1,1):
                output=np.asarray(ta.apply_operation(source,ta.Operation(name,30,sign)))
                expected=np.zeros_like(source)
                if axis==1:
                    if sign==1:expected[:,:22]=source[:,10:]
                    else:expected[:,10:]=source[:,:22]
                elif sign==1:expected[:22]=source[10:]
                else:expected[10:]=source[:22]
                self.assertTrue(np.array_equal(output,expected))

    def test_geometry_is_nearest_and_fill_has_no_interpolated_color(self):
        palette=np.array([(255,11,79),(23,241,17),(13,61,251),(183,151,97)],dtype=np.uint8)
        rows,cols=np.indices((32,32));source=palette[(rows//2+cols//3)%4]
        allowed={tuple(x) for x in palette};allowed.add((0,0,0))
        for name in ta.SIGNED:
            for level in range(31):
                for sign in (-1,1):
                    output=np.asarray(ta.apply_operation(source,ta.Operation(name,level,sign)))
                    self.assertTrue({tuple(x) for x in output.reshape(-1,3)}<=allowed)
                    if level==0:self.assertTrue(np.array_equal(output,source))
        white=np.full((32,32,3),255,dtype=np.uint8)
        for name in ("rotate","shear_x","shear_y"):
            output=np.asarray(ta.apply_operation(white,ta.Operation(name,30)))
            self.assertTrue((output==0).all(axis=-1).any())
            self.assertTrue((output==255).all(axis=-1).any())

    def test_operations_match_pinned_author_PIL_mechanics_for_all_levels(self):
        image=Image.fromarray(self.pattern())
        enhancers={"color":ImageEnhance.Color,"contrast":ImageEnhance.Contrast,
                   "brightness":ImageEnhance.Brightness,"sharpness":ImageEnhance.Sharpness}
        for level in range(31):
            # Explicit independent PIL references also catch channel swaps.
            expected={"identity":image,"autocontrast":ImageOps.autocontrast(image),"equalize":ImageOps.equalize(image),
                      "posterize":ImageOps.posterize(image,8-int(level*4/30)),
                      "solarize":ImageOps.solarize(image,256-int(level*256/30))}
            expected.update({name:enhancer(image).enhance(float(level)*(1.9-.1)/30+.1) for name,enhancer in enhancers.items()})
            for name,reference in expected.items():
                self.assertEqual(ta.apply_operation(image,ta.Operation(name,level)).tobytes(),reference.tobytes())
            for sign in (-1,1):
                geometries={"rotate":image.rotate(sign*int(level*30/30)),
                            "shear_x":image.transform(image.size,Image.Transform.AFFINE,(1,sign*float(level)*.3/30,0,0,1,0)),
                            "shear_y":image.transform(image.size,Image.Transform.AFFINE,(1,0,0,sign*float(level)*.3/30,1,0)),
                            "translate_x":image.transform(image.size,Image.Transform.AFFINE,(1,0,sign*int(level*10/30),0,1,0)),
                            "translate_y":image.transform(image.size,Image.Transform.AFFINE,(1,0,0,0,1,sign*int(level*10/30)))}
                for name,reference in geometries.items():
                    self.assertEqual(ta.apply_operation(image,ta.Operation(name,level,sign)).tobytes(),reference.tobytes())

    def test_input_is_copied_including_identity_and_noncontiguous_arrays(self):
        image=Image.fromarray(self.pattern());before=image.tobytes()
        result=ta.apply_operation(image,ta.Operation("identity",0))
        self.assertIsNot(result,image);result.putpixel((0,0),(0,0,0));self.assertEqual(image.tobytes(),before)
        source=self.pattern()[:,::-1];self.assertFalse(source.flags.c_contiguous)
        self.assertTrue(np.array_equal(ta.apply_operation(source,ta.Operation("identity",0)),source))

    def test_trace_and_local_numpy_state_replay(self):
        image=self.pattern();rng=np.random.Generator(np.random.PCG64(741));saved=copy.deepcopy(rng.bit_generator.state)
        first=[ta.augment(image,rng,return_trace=True) for _ in range(35)]
        rng.bit_generator.state=saved
        second=[ta.augment(image,rng,return_trace=True) for _ in range(35)]
        self.assertEqual([trace for _,trace in first],[trace for _,trace in second])
        self.assertEqual([im.tobytes() for im,_ in first],[im.tobytes() for im,_ in second])
        for output,trace in first:
            restored=ta.Operation(**json.loads(json.dumps(asdict(trace))))
            self.assertEqual(ta.apply_operation(image,restored).tobytes(),output.tobytes())

    def test_explicit_python_random_replay_and_parent_global_RNG_isolation(self):
        image=self.pattern();numpy_global=np.random.get_state();python_global=random.getstate()
        for rng in (np.random.default_rng(17),random.Random(17)):
            for _ in range(80):ta.augment(image,rng)
        after=np.random.get_state()
        self.assertEqual(numpy_global[0],after[0]);self.assertTrue(np.array_equal(numpy_global[1],after[1]))
        self.assertEqual(numpy_global[2:],after[2:]);self.assertEqual(python_global,random.getstate())
        rng=random.Random(2026);saved=rng.getstate();first=[ta.sample_operation(rng) for _ in range(40)]
        rng.setstate(saved);self.assertEqual(first,[ta.sample_operation(rng) for _ in range(40)])

    def test_sampling_has_one_op_and_closed_integer_bounds_and_sign(self):
        rng=np.random.default_rng(726);traces=[ta.sample_operation(rng) for _ in range(3000)]
        self.assertEqual({trace.name for trace in traces},set(ta.OPERATIONS))
        self.assertEqual({trace.level for trace in traces},set(range(31)))
        self.assertEqual({trace.sign for trace in traces if trace.name in ta.SIGNED},{-1,1})
        self.assertEqual({trace.sign for trace in traces if trace.name not in ta.SIGNED},{1})
        image=self.pattern()
        with patch.object(ta,"sample_operation",return_value=ta.Operation("brightness",30)) as sample,patch.object(ta,"apply_operation",wraps=ta.apply_operation) as apply:
            ta.augment(image,np.random.default_rng(1));sample.assert_called_once();apply.assert_called_once()

    def test_bad_images_traces_spaces_and_global_rng_modules_fail_closed(self):
        rng=np.random.default_rng(3);state=copy.deepcopy(rng.bit_generator.state)
        for image in (np.zeros((32,32,3),dtype=np.float32),np.zeros((3,32,32),dtype=np.uint8),
                      Image.new('L',(32,32)),Image.new('RGBA',(32,32)),Image.new('RGB',(31,32)),None):
            with self.assertRaises(ValueError):ta.augment(image,rng)
        self.assertEqual(state,rng.bit_generator.state)
        for trace in (ta.Operation("invert",3),ta.Operation("rotate",31),ta.Operation("rotate",True),
                      ta.Operation("rotate",3,0),ta.Operation("brightness",3,-1),ta.Operation("identity",3,1,"wide_standard"),None):
            with self.assertRaises(ValueError):ta.apply_operation(self.pattern(),trace)
        for invalid in (None,np.random,random,random.SystemRandom(),np.random.RandomState(0)):
            with self.assertRaises(TypeError):ta.augment(self.pattern(),invalid)

    def test_module_has_no_torch_runtime_trainer_loader_or_global_rng_calls(self):
        tree=ast.parse(Path(ta.__file__).read_text())
        imports={node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
        imports|={alias.name.split('.')[0] for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names}
        self.assertFalse(imports & {'torch','runtime','data','model','train','torchvision'})
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                text=ast.unparse(node.func)
                self.assertFalse(text.startswith('np.random.') or text.startswith('random.'),text)


if __name__=='__main__':unittest.main()
