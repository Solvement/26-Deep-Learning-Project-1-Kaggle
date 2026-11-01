"""Pure contracts run locally; Torch numerical comparisons are reserved for CI.

No official arrays, CUDA allocation, account API or real experiment is used.
"""
from __future__ import annotations

import ast
from contextlib import ExitStack, redirect_stdout
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research import preflight_core_control as preflight


class PureContracts(unittest.TestCase):
    def test_bare_help_is_no_action(self):
        with patch.object(preflight, "parent") as parent, patch.object(preflight, "worker") as worker, redirect_stdout(io.StringIO()) as output:
            preflight.main([])
        parent.assert_not_called()
        worker.assert_not_called()
        self.assertIn("--run", output.getvalue())

    def test_explicit_run_and_predeclared_seed_are_required(self):
        for args in (["--run", "new"], ["--seed", "2026"], ["--_manifest-sha", "a"*64]):
            with self.assertRaises(ValueError):
                preflight.main(args)
        with redirect_stdout(io.StringIO()), patch.object(preflight, "parent") as parent:
            preflight.main(["--run", "fresh", "--recipe", "wrn16_5", "--seed", "2027"])
            self.assertEqual(parent.call_args.args[0].seed, 2027)

    def test_exact_raw_configs_and_only_predeclared_architecture_change(self):
        primary, repeat = preflight.configuration(2026), preflight.configuration(2027)
        self.assertEqual(repeat, {**primary, "seed":2027})
        self.assertEqual((primary["depth"],primary["widen_factor"],primary["epochs"],primary["batch_size"]), (16,5,160,128))
        self.assertIs(primary["train_all"], False)
        self.assertEqual(primary["final_variant"], "ema")
        with self.assertRaises(ValueError):
            preflight.configuration(True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("train.py", "research/configs/wrn34_3.json", *(v[0] for v in preflight.CONFIGS.values())):
                (root/name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(preflight.ROOT/name, root/name)
            target = root/preflight.CONFIGS[2027][0]
            target.write_bytes(target.read_bytes()+b" ")
            with self.assertRaises(ValueError):
                preflight.configuration(2026, root)

    def test_static_parameter_count_includes_bn_affine_and_projection(self):
        width, n = 5, 2
        previous, total = 16, 432
        for channels in (16*width,32*width,64*width):
            total += 10*previous*channels+9*channels**2+2*previous+2*channels
            total += (n-1)*(18*channels**2+4*channels)
            previous = channels
        total += 2*previous+10*previous+10
        self.assertEqual(total, 4289754)
        self.assertEqual(6*n+1, 13)

    def test_shadow_300_exact_recipe_changes_only_epochs_and_repeat_seed(self):
        primary = preflight.configuration(2026, recipe="wrn34_3_300_bf16")
        repeat = preflight.configuration(2027, recipe="wrn34_3_300_bf16")
        self.assertEqual(repeat,{**primary,"seed":2027})
        self.assertEqual((primary["depth"],primary["widen_factor"],primary["epochs"]),(34,3,300))
        recipe = preflight.RECIPES["wrn34_3_300_bf16"]
        self.assertEqual((recipe["parameters"],recipe["bn_modules"],313*recipe["epochs"]),(4166298,31,93900))
        with self.assertRaises(ValueError):
            preflight.configuration(2026,recipe="any_config")

    def test_canonical_bf16_copies_preserve_legacy_recipe_and_original_bytes(self):
        legacy = preflight.ROOT/"research/configs/wrn34_3_300.json"
        legacy_bytes = legacy.read_bytes()
        normalized = legacy_bytes.replace(b"\r\n", b"\n")
        expected_lf = "cc55380bd03bd0ca2147c7e4a34c31de37345480d71a9721b65737717fc2f348"
        self.assertEqual(hashlib.sha256(normalized).hexdigest(),expected_lf)
        # The frozen worktree was CRLF on Windows; Git's original blob is LF.
        # Do not rewrite the old path to make the new protocol portable.
        if b"\r\n" in legacy_bytes:
            self.assertEqual(hashlib.sha256(legacy_bytes).hexdigest(),"73b6f816a5a1c19ed9fa458c151133c6003e2bff40899df6e9848e0a3b943b85")
        else:
            self.assertEqual(hashlib.sha256(legacy_bytes).hexdigest(),expected_lf)
        self.assertEqual((preflight.ROOT/preflight.WRN300_CONFIGS[2026][0]).read_bytes(),normalized)
        old_repeat = preflight.ROOT/"research/configs/wrn34_3_300_seed2027.json"
        self.assertEqual(old_repeat.read_bytes(),(preflight.ROOT/preflight.WRN300_CONFIGS[2027][0]).read_bytes())
        self.assertEqual(json.loads(legacy_bytes),preflight.read(preflight.ROOT/preflight.WRN300_CONFIGS[2026][0]))
        self.assertEqual(legacy.read_bytes(),legacy_bytes)
        untouched = subprocess.check_output(["git","diff","--name-only","HEAD","--",str(legacy),str(old_repeat)],cwd=preflight.ROOT,text=True)
        self.assertEqual(untouched.strip(),"")

    def test_all_four_preflight_configs_have_stable_git_index_and_checkout_bytes(self):
        # Exercise a disposable Git index under Windows-style autocrlf, without
        # touching this checkout's index, configuration, files or credentials.
        configs = [item for recipe in preflight.RECIPES.values() for item in recipe["configs"].values()]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git","init","-q",str(root)],check=True,capture_output=True)
            shutil.copyfile(preflight.ROOT/".gitattributes",root/".gitattributes")
            for name,digest in configs:
                raw = (preflight.ROOT/name).read_bytes()
                self.assertNotIn(b"\r",raw)
                self.assertEqual(hashlib.sha256(raw).hexdigest(),digest)
                (root/name).parent.mkdir(parents=True,exist_ok=True)
                (root/name).write_bytes(raw)
            names = [name for name,_ in configs]
            subprocess.run(["git","-C",str(root),"-c","core.autocrlf=false","add","--",".gitattributes"],check=True,capture_output=True)
            subprocess.run(["git","-C",str(root),"-c","core.autocrlf=true","-c","core.safecrlf=true","add","--",*names],check=True,capture_output=True)
            checkout = root/"isolated_checkout"
            checkout.mkdir()
            subprocess.run(["git","-C",str(root),"-c","core.autocrlf=true","checkout-index","--all","--prefix="+checkout.as_posix()+"/"],check=True,capture_output=True)
            for name,digest in configs:
                index_bytes = subprocess.check_output(["git","-C",str(root),"show",":"+name])
                self.assertEqual(hashlib.sha256(index_bytes).hexdigest(),digest)
                self.assertEqual((checkout/name).read_bytes(),index_bytes)

    def test_complete_reservation_hardstop_and_daily_limits(self):
        now = datetime(2026,11,1,tzinfo=timezone.utc)
        policy = {"concurrent_local_gpu_jobs":1,"local_gpu_seconds_per_utc_day":21600,"local_job_timeout_seconds":10800,
                  "stops_at":(now+timedelta(days=1)).isoformat()}
        self.assertEqual(preflight.bounded_seconds(policy, now, 0), 180)
        for used in (21500,float("nan"),-1):
            with self.assertRaises(ValueError):
                preflight.bounded_seconds(policy, now, used)
        with self.assertRaises(ValueError):
            preflight.bounded_seconds({**policy,"stops_at":(now+timedelta(seconds=179)).isoformat()},now,0)

    def test_atomic_lease_never_renames_or_overwrites_unknown_or_racing_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"local_lease.json"
            original = b'{"cleanup_requires_reconciliation":true,"pid":123}'
            path.write_bytes(original)
            with self.assertRaises(FileExistsError):
                preflight.claim_lease(path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(len(list(path.parent.iterdir())),1)
            path.unlink()
            preflight.claim_lease(path)
            first = path.read_bytes()
            with self.assertRaises(FileExistsError):
                preflight.claim_lease(path)
            self.assertEqual(path.read_bytes(),first)

    def test_unknown_cleanup_status_keeps_lease_even_when_other_flags_are_absent(self):
        clean = {"process_tree_cleanup_verified":True}
        self.assertTrue(preflight.lease_can_release({},clean))
        for owner in ({"status":"unknown"},{"status":"failed"},{"status":"cleanup_unverified"},
                      {"cleanup_requires_reconciliation":True},{"process_tree_cleanup_verified":False}):
            self.assertFalse(preflight.lease_can_release(owner,clean))
        self.assertFalse(preflight.lease_can_release({}, {"child_pid":1,"process_tree_cleanup_verified":False}))

    def test_frozen_accounting_reuse_does_not_import_runtime(self):
        before = "runtime" in sys.modules
        primitives = preflight.accounting()
        self.assertEqual("runtime" in sys.modules, before)
        self.assertTrue(callable(primitives["used_seconds"]))
        self.assertNotIn("acquire_lease",primitives)
        with tempfile.TemporaryDirectory() as temporary:
            now = datetime.now(timezone.utc)
            self.assertEqual(primitives["used_seconds"](Path(temporary),now,now+timedelta(days=1)),0)

    def official_fixture(self, folder):
        hashes = {}
        for name in preflight.ARRAYS:
            (folder/name).write_bytes(name.encode())
            hashes[name] = preflight.sha256(folder/name)
        identity = preflight.DATA_IDENTITY
        document = {"competition":"cifar-10-res-net-architecture-challenge-dl", "source_type":"official_kaggle_competition_training",
            "files_sha256":hashes, "source_provenance":{"official_train_npz_sha256":identity["archive_sha256"],
                "train_labels_csv_sha256":"b596f4cbb12107ecac987e3a61123af40408a6bb83719a1dbb5cc3ce70968e93"},
            "split":{"schema":"train_dev_confirmation_v1","seed":2026,"indices_sha256":identity["split_sha256"],
                "train_indices_sha256":identity["train_indices_sha256"]}}
        preflight.write(folder/"training_manifest.json",document)
        return document

    def test_official_identity_and_training_only_allowlist_are_hard_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            document = self.official_fixture(folder)
            paths, identity = preflight.official_inputs(folder)
            self.assertEqual(identity,preflight.DATA_IDENTITY)
            self.assertEqual(len(paths),5)
            for field in ("archive", "split", "train"):
                changed = copy.deepcopy(document)
                if field == "archive": changed["source_provenance"]["official_train_npz_sha256"] = "0"*64
                elif field == "split": changed["split"]["indices_sha256"] = "0"*64
                else: changed["split"]["train_indices_sha256"] = "0"*64
                preflight.write(folder/"training_manifest.json",changed)
                with self.assertRaises(ValueError): preflight.official_inputs(folder)
            changed = copy.deepcopy(document)
            changed["files_sha256"]["test.csv"] = "0"*64
            preflight.write(folder/"training_manifest.json",changed)
            with self.assertRaises(ValueError): preflight.official_inputs(folder)

    def report_fixture(self):
        manifest = {"recipe":"wrn16_5", "seed":2026,"config_sha256":preflight.CONFIGS[2026][1],"effective_config":preflight.configuration(2026),
            "source_sha256":{"source.py":"0"*64},"source_git_commit":"a"*40,"data_identity":preflight.DATA_IDENTITY}
        manifest["effective_config_sha256"] = preflight.json_sha(manifest["effective_config"])
        env = {"device":"cuda","gpu":"Unit fixture only","amp_dtype":"torch.bfloat16"}
        report = {"schema":preflight.REPORT_SCHEMA,"status":"complete","engineering_only":True,"synthetic_data":False,
            "accuracy_evaluated":False,"formal_training_admitted":False,"hidden_images_loaded":False,"confirmation_evaluated":False,
            "training_precision":"bf16","native_bf16_without_emulation":True,"grad_scaler_enabled":False,
            "recipe":"wrn16_5","future_trainer":"frozen","epochs":160,"planned_formal_attempted_updates":50080,"disabled_scaler_scale":None,
            "master_parameter_dtype":"torch.float32","ema_parameter_dtype":"torch.float32",
            "seed":2026,"parameter_count":4289754,"trainable_parameter_count":4289754,"training_pool_examples":40000,
            "evaluation_examples":0,"batch_size":128,"warmup_updates":3,"timed_updates":20,"successful_updates":23,
            "skipped_updates":0,"source_examples_seen":2944,"unique_source_examples":2944,"bn_modules":13,"bn_forward_counts":[23]*13,
            "ema_buffer_copies":0,"preflight_state_discarded":True,"input_manifest_sha256":"b"*64,"environment":env,
            "environment_sha256":preflight.json_sha(env),"peak_gpu_memory_bytes":1024,"step_seconds":[.1]*23,
            "median_step_seconds":.1,"conservative_pipeline_seconds":1.5*.1*313*160+300,"setup_seconds":1.,"full_step_seconds":sum([.1]*23),
            "ordered_training_indices_sha256":"c"*64,"ordered_official_ids_sha256":"d"*64,
            "steps":[{"updates":i,"all_parameters_finite":True,"bn_single_forward":True,"loss":1.,"gradient_norm":1.,
                "lr":.1*i/(5*313),"parameter_probe_max_delta":.01,"ema_probe_max_delta":.01,"mixed":False,
                "logits_dtype":"torch.bfloat16"} for i in range(1,24)]}
        report.update({k:manifest[k] for k in ("config_sha256","effective_config_sha256","source_sha256","source_git_commit","data_identity")})
        return report, manifest

    def test_complete_typed_report_and_fixed_160_round_guard(self):
        report, manifest = self.report_fixture()
        preflight.validate_report(report,manifest,"b"*64)
        for key, value in (("successful_updates",22),("synthetic_data",True),("parameter_count",True),
                ("training_precision","fp16"),("formal_training_admitted",True),("ema_buffer_copies",23),
                ("conservative_pipeline_seconds",10),("environment_sha256","0"*64),("bn_forward_counts",[24]*13)):
            changed = {**report,key:value}
            with self.assertRaises(ValueError): preflight.validate_report(changed,manifest,"b"*64)

    def test_shadow_300_report_cannot_reuse_160_counts_guard_or_scaler_protocol(self):
        report, manifest = self.report_fixture()
        manifest.update(recipe="wrn34_3_300_bf16",effective_config=preflight.configuration(2026,recipe="wrn34_3_300_bf16"),
                        config_sha256=preflight.WRN300_CONFIGS[2026][1])
        manifest["effective_config_sha256"] = preflight.json_sha(manifest["effective_config"])
        report.update(recipe=manifest["recipe"],future_trainer="shadow",epochs=300,planned_formal_attempted_updates=93900,
            disabled_scaler_scale=1.,parameter_count=4166298,trainable_parameter_count=4166298,bn_modules=31,bn_forward_counts=[23]*31,
            conservative_pipeline_seconds=1.5*.1*313*300+300,config_sha256=manifest["config_sha256"],effective_config_sha256=manifest["effective_config_sha256"])
        preflight.validate_report(report,manifest,"b"*64)
        for key,value in (("epochs",160),("planned_formal_attempted_updates",50080),("disabled_scaler_scale",65536.),
                ("bn_modules",13),("future_trainer","frozen"),("conservative_pipeline_seconds",1.5*.1*313*160+300)):
            with self.assertRaises(ValueError): preflight.validate_report({**report,key:value},manifest,"b"*64)

    def test_nonfinite_gradients_wrong_schedule_or_zero_update_proof_are_rejected(self):
        report, manifest = self.report_fixture()
        for field,value in (("gradient_norm",float("inf")),("parameter_probe_max_delta",0),
                ("ema_probe_max_delta",0),("lr",.1),("logits_dtype","torch.float16"),("updates",True)):
            changed = copy.deepcopy(report)
            changed["steps"][0][field] = value
            with self.assertRaises(ValueError): preflight.validate_report(changed,manifest,"b"*64)

    def test_direct_worker_is_rejected_without_importing_torch(self):
        before = "torch" in sys.modules
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
            preflight.validate_worker(Path(temporary),"a"*64)
        self.assertEqual("torch" in sys.modules,before)

    def test_guarded_worker_checks_lease_nonce_bootstrap_source_deadline_and_os_membership(self):
        from research import profile_augmix
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root/"results/research/fresh"
            directory = folder/("process_tree_"+"a"*32)
            directory.mkdir(parents=True)
            guard = root/"research/process_tree_guard.py"
            guard.parent.mkdir()
            shutil.copyfile(preflight.ROOT/"research/process_tree_guard.py",guard)
            (root/"train.py").write_text("unit fixture")
            config_path = root/preflight.CONFIGS[2026][0]
            config_path.parent.mkdir(parents=True)
            config_path.write_text("unit fixture")
            data_dir = root/"data"
            data_dir.mkdir()
            data_manifest = data_dir/"training_manifest.json"
            data_manifest.write_text("unit fixture")
            sources = {"train.py":preflight.sha256(root/"train.py")}
            config = preflight.configuration(2026)
            inputs = {str(p.resolve()):preflight.sha256(p) for p in (data_manifest,config_path,root/"train.py")}
            manifest = {"schema":preflight.INPUT_SCHEMA,"recipe":"wrn16_5","seed":2026,"run":"fresh","data_dir":str(data_dir),
                "source_sha256":sources,"source_git_commit":"a"*40,"config_sha256":preflight.CONFIGS[2026][1],
                "effective_config":config,"effective_config_sha256":preflight.json_sha(config),
                "data_identity":preflight.DATA_IDENTITY,"input_sha256":inputs}
            preflight.write(folder/"inputs.json",manifest)
            digest = preflight.sha256(folder/"inputs.json")
            now = datetime.now(timezone.utc)
            receipt = {"schema":preflight.SCHEMA,"status":"benchmarking","backend":"local","mode":"preflight","precision":"bf16","run":"fresh","recipe":"wrn16_5",
                "started_at":now.isoformat(),"reserved_seconds":180,"deadline_unix":now.timestamp()+180,
                "pid":11,"process_created_at":1.5,"child_pid":12,"child_created_at":2.5,
                "input_manifest_sha256":digest,"source_sha256":sources,"config_sha256":manifest["config_sha256"],"input_sha256":inputs,
                "confirmation_evaluated":False,"submitted":False}
            lease = {"pid":11,"process_created_at":1.5}
            spec = {"schema":"owned_python_process_tree_v1","nonce":"a"*32,"host_pid":11,"host_created_at":1.5,
                "cwd":str(root),"script":str(Path(preflight.__file__).resolve()),"script_sha256":preflight.sha256(preflight.__file__),
                "guard_source_sha256":preflight.PINS["research/process_tree_guard.py"],"arguments":["--_worker",str(folder),"--_manifest-sha",digest],
                "deadline_unix":now.timestamp()+170,"job_name":"Local\\SaberAugMix_"+"a"*32 if os.name == "nt" else None}
            ready = {"schema":spec["schema"],"nonce":spec["nonce"],"pid":12,"created_at":2.5,"script_sha256":spec["script_sha256"],
                "method":"windows_job_v1" if os.name == "nt" else "posix_session_v1"}
            process = SimpleNamespace(pid=12,create_time=lambda:2.5,cmdline=lambda:[sys.executable,str(guard),"--bootstrap",str(directory/"spec.json")])
            parent = SimpleNamespace(pid=11,create_time=lambda:1.5)
            with ExitStack() as stack:
                stack.enter_context(patch.object(preflight,"ROOT",root))
                stack.enter_context(patch.object(preflight,"source_snapshot",return_value=sources))
                stack.enter_context(patch.object(preflight,"configuration",return_value=config))
                stack.enter_context(patch.object(preflight,"official_inputs",return_value=([data_manifest],preflight.DATA_IDENTITY)))
                stack.enter_context(patch.object(preflight.os,"getppid",return_value=11))
                stack.enter_context(patch.object(preflight.psutil,"Process",side_effect=lambda pid=None:process if pid is None else parent))
                boundary = stack.enter_context(patch.object(profile_augmix,"verify_os_boundary"))
                def save():
                    for path,document in ((folder/"supervisor.json",receipt),(root/"results/research/local_lease.json",lease),
                            (directory/"spec.json",spec),(directory/"ready.json",ready)):
                        preflight.write(path,document)
                save()
                self.assertEqual(preflight.validate_worker(folder,digest),manifest)
                boundary.assert_called_once()
                for target,key,value in ((lease,"cleanup_requires_reconciliation",True),(lease,"pid",99),(spec,"nonce","b"*32),
                        (spec,"script_sha256","b"*64),(spec,"deadline_unix",now.timestamp()+200),(ready,"pid",99),
                        (receipt,"child_created_at",99),(receipt,"reserved_seconds",181)):
                    old = target.get(key)
                    target[key] = value
                    save()
                    with self.assertRaises(ValueError): preflight.validate_worker(folder,digest)
                    if old is None: target.pop(key)
                    else: target[key] = old
                save()
                boundary.side_effect = ValueError("not a real OS member")
                with self.assertRaises(ValueError): preflight.validate_worker(folder,digest)


class TorchNumericalContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import runtime
        import torch
        import train
        cls.torch, cls.core = torch, train
        torch.set_num_threads(1)

    def tiny(self):
        t = self.torch
        return t.nn.Sequential(t.nn.Conv2d(3,4,3,padding=1,bias=False), t.nn.BatchNorm2d(4),t.nn.ReLU(),
            t.nn.AdaptiveAvgPool2d(1),t.nn.Flatten(),t.nn.Linear(4,10))

    def test_actual_model_parameter_and_bn_counts(self):
        from model import build_model,count_parameters
        for name,recipe in preflight.RECIPES.items():
            config = preflight.configuration(2026,recipe=name)
            model = build_model({k:config[k] for k in ("depth","widen_factor","dropout","num_classes")})
            self.assertEqual(count_parameters(model),recipe["parameters"])
            self.assertEqual(count_parameters(model,True),recipe["parameters"])
            self.assertEqual(sum(isinstance(m,self.torch.nn.BatchNorm2d) for m in model.modules()),recipe["bn_modules"])

    def compare_original_steps(self, trainer):
        """Execute the frozen original for-body independently, not a copy of ours."""
        import random
        import numpy as np
        t, core = self.torch, self.core
        recipe = "wrn16_5" if trainer == "frozen" else "wrn34_3_300_bf16"
        config = {**preflight.configuration(2026,recipe=recipe),"batch_size":4,"mixup_probability":1.}
        core.seed_all(42)
        original = self.tiny()
        initial = copy.deepcopy(original.state_dict())
        source = "train.py" if trainer == "frozen" else "research/train_shadow.py"
        tree = ast.parse((preflight.ROOT/source).read_text())
        train = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == "train")
        epoch = next(n for n in train.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id == "epoch")
        batch = next(n for n in epoch.body if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple))
        reference_code = compile(ast.Module(body=batch.body,type_ignores=[]),"frozen_train_batch","exec")
        results = []
        for reference in (True,False):
            core.seed_all(2026)
            model = self.tiny()
            model.load_state_dict(initial)
            ema = copy.deepcopy(model).eval().requires_grad_(False)
            opt = t.optim.SGD(model.parameters(),lr=config["lr"],momentum=config["momentum"],weight_decay=config["weight_decay"],nesterov=True)
            scaler = t.amp.GradScaler("cpu",enabled=False) if trainer == "shadow" else None
            images = t.randint(0,256,(16,32,32,3),dtype=t.uint8)
            labels = t.arange(16)%10
            order = t.randperm(16)
            namespace = dict(torch=t,F=t.nn.functional,np=np,random=random,config=config,device=t.device("cpu"),
                schedule=core.schedule,augment=core.augment,model=model,ema=ema,opt=opt,model_params=list(model.parameters()),
                ema_params=list(ema.parameters()),train_x=images,train_y=labels,order=order,epoch=1,steps_per_epoch=313,
                total_steps=config["epochs"]*313,updates=0,correct=t.zeros(()),loss_sum=t.zeros(()),seen=0,
                dtype=t.float32,scaler=scaler,epoch_updates=0,epoch_skipped=0,skipped=0)
            if trainer == "shadow":
                from research.train_shadow import optimizer_step
                namespace["optimizer_step"] = optimizer_step
            for index in range(4):
                if reference:
                    namespace.update(batch=index,start=index*4)
                    exec(reference_code,namespace)
                else:
                    preflight.core_step(core,t,model,ema,opt,images,labels,order,index,config,index,trainer,scaler)
            results.append((copy.deepcopy(model.state_dict()),copy.deepcopy(ema.state_dict()),copy.deepcopy(opt.state_dict()),
                            random.getstate(),np.random.get_state(),t.get_rng_state().clone()))
        left,right = results
        for index in (0,1):
            for key in left[index]: self.assertTrue(t.equal(left[index][key],right[index][key]),key)
        self.assertEqual(left[2]["param_groups"],right[2]["param_groups"])
        for key in left[2]["state"]:
            self.assertTrue(t.equal(left[2]["state"][key]["momentum_buffer"],right[2]["state"][key]["momentum_buffer"]))
        self.assertEqual(left[3],right[3])
        self.assertEqual(left[4][0],right[4][0])
        self.assertTrue(np.array_equal(left[4][1],right[4][1]))
        self.assertEqual(left[4][2:],right[4][2:])
        self.assertTrue(t.equal(left[5],right[5]))
        self.assertEqual(left[0]["1.num_batches_tracked"].item(),4)
        self.assertEqual(left[1]["1.num_batches_tracked"].item(),0)

    def test_instrumented_steps_match_original_core_ast_parameters_sgd_ema_rng_and_bn(self):
        self.compare_original_steps("frozen")

    def test_shadow_disabled_scaler_steps_match_original_shadow_ast(self):
        self.compare_original_steps("shadow")


if __name__ == "__main__":
    unittest.main()
