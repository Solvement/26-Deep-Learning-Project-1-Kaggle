"""Synthetic byte/JSON lineage contracts; no tensors or competition inputs."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch
from research import scratch_kd_lineage as lineage


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


class PureContracts(unittest.TestCase):
    def test_import_has_no_numeric_runtime_or_filesystem_effects(self):
        code="import sys;from research import scratch_kd_lineage;assert not {'torch','numpy','runtime','PIL','nbformat','nbclient'}&set(sys.modules)"
        subprocess.check_output([sys.executable,"-c",code],cwd=lineage.ROOT)

    def fixture(self, root):
        for name in {**lineage.CORE, **lineage.METHOD}:
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(lineage.ROOT/name,target)
        checkpoint=root/lineage.CHECKPOINT_REL
        checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b"synthetic teacher metadata fixture; no tensor")
        teacher_sha=lineage.sha256(checkpoint)
        provenance={"code":{"git_commit":None},"train_examples":40000,"validation_examples":5000,
            "parameter_count":4166298,"data_signature":lineage.SIGNATURE}
        prov=checkpoint.parent/"provenance.json";write(prov,provenance)
        training={"schema":"research_shadow_training_v1","source_sha256":lineage.METHOD,"precision":"fp16",
            "grad_scaler_enabled":True,"training_seed":2026,"split_seed":2026,
            "confirmation_evaluated":False,"hidden_images_loaded":False,"external_data_loaded":False}
        context={"device_type":"cuda","amp_dtype":"torch.bfloat16","torch_version":"2.12.0+cu130","training_core_sha256":lineage.CORE}
        config=lineage.expected_config(root)
        benchmark={"schema":"official_development_stress_v3","status":"complete","examples":5000,
            "training_examples":40000,"checkpoint_sha256":teacher_sha,"checkpoint_epoch":298,
            "checkpoint_variant":"model","parameter_count":4166298,"model_config":lineage.MODEL,
            "config":config,"data_signature":lineage.SIGNATURE,"bn_recalibrated":False,
            "confirmation_evaluated":False,"hidden_images_loaded":False,"external_data_loaded":False,
            "evaluation_context":context,"research_training_integrity":{"schema":"research_shadow_integrity_v1",
                "training":training,"successful_updates":93864,"skipped_updates":36,
                "provenance_document_sha256":lineage.sha256(prov)}}
        bench=root/lineage.BENCHMARK_REL;write(bench,benchmark)
        evidence={str(root/n):digest for n,digest in {**lineage.CORE,**lineage.METHOD}.items()}
        evidence.update({str(checkpoint):teacher_sha,str(prov):lineage.sha256(prov),str(bench):lineage.sha256(bench)})
        for index in range(20):
            path=root/f"fixture_evidence/{index}.json";write(path,{"synthetic":index});evidence[str(path)]=lineage.sha256(path)
        data=root/"data";data.mkdir()
        official=[]
        for index in range(5):
            q=data/f"synthetic{index}.json";write(q,{"not_an_array":index});official.append(q)
        frozen={"schema":"research_candidate_freeze_v2","status":"frozen","config":config,
            "model_config":lineage.MODEL,"parameter_count":4166298,"selected_epoch":298,"weight_variant":"model",
            "policy":"center80_shift20","data_signature":lineage.SIGNATURE,"confirmation_evaluated":False,
            "repeat_prediction_authorized":False,"full_refit_supported":False,"source_git_commit":None,
            "checkpoint_path":str(checkpoint),"benchmark_path":str(bench),"data_dir":str(data),"evaluation_context":context,
            "primary_decision":{"seed":2026,"checkpoint_sha256":teacher_sha,"benchmark_sha256":lineage.sha256(bench),
                "policy":"center80_shift20","decided_at":"2020-01-01T00:00:00+00:00"},
            "frozen_at":"2020-01-01T00:00:01+00:00","evidence_sha256":evidence}
        freeze=root/lineage.FREEZE_REL;write(freeze,frozen)
        return frozen,benchmark,official

    def boundary(self, root, official):
        from research import incumbent_compare,preflight_core_control
        stack=ExitStack()
        stack.enter_context(patch.object(lineage,"FREEZE_SHA",lineage.sha256(root/lineage.FREEZE_REL)))
        stack.enter_context(patch.object(lineage,"TEACHER_SHA",lineage.sha256(root/lineage.CHECKPOINT_REL)))
        stack.enter_context(patch.object(lineage,"BENCHMARK_SHA",lineage.sha256(root/lineage.BENCHMARK_REL)))
        stack.enter_context(patch.object(preflight_core_control,"official_inputs",return_value=(official,preflight_core_control.DATA_IDENTITY)))
        stack.enter_context(patch.object(incumbent_compare,"validate_document"))
        return stack

    def test_valid_complete30_reference_and_any_metadata_substitution_rejects(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);f,b,official=self.fixture(root)
            with self.boundary(root,official):
                reference=lineage.build_teacher_reference(root=root)
                self.assertEqual(len(reference),8);self.assertEqual(len(reference["evidence_sha256"]),30)
                self.assertEqual(lineage.validate_teacher_reference(reference,root=root),reference)
                for key,value in (("schema","renamed_v1"),("checkpoint_sha256","a"*64),("evidence_sha256",{}),
                        ("checkpoint_path",str(root/"seed2027/best.pt"))):
                    with self.subTest(key=key),self.assertRaises(ValueError):
                        lineage.validate_teacher_reference({**reference,key:value},root=root)
                with self.assertRaises(ValueError):lineage.validate_teacher_reference({**reference,"extra":True},root=root)
                changed=root/"fixture_evidence/3.json";changed.write_text("changed")
                with self.assertRaises(ValueError):lineage.build_teacher_reference(root=root)

    def test_re_signed_fixture_cannot_change_primary_seed_variant_or_null_commit(self):
        for key,value in (("source_git_commit","a"*40),("selected_epoch",300),("weight_variant","ema"),
                ("parameter_count",True),("confirmation_evaluated",True),("repeat_prediction_authorized",True)):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);f,b,official=self.fixture(root);f[key]=value;write(root/lineage.FREEZE_REL,f)
                with self.boundary(root,official),self.assertRaises(ValueError):lineage.build_teacher_reference(root=root)

    def test_original_fp16_teacher_skips_cannot_be_relabelled_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);f,b,official=self.fixture(root)
            b["research_training_integrity"].update(successful_updates=93900,skipped_updates=0)
            bench=root/lineage.BENCHMARK_REL;write(bench,b);digest=lineage.sha256(bench)
            f["primary_decision"]["benchmark_sha256"]=digest;f["evidence_sha256"][str(bench)]=digest
            write(root/lineage.FREEZE_REL,f)
            with self.boundary(root,official),self.assertRaises(ValueError):lineage.build_teacher_reference(root=root)

    def test_wrong_data_context_or_missing_teacher_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);f,b,official=self.fixture(root)
            with self.boundary(root,official),self.assertRaises(ValueError):
                lineage.build_teacher_reference(root/"other_data",root)
            f["evidence_sha256"].pop(next(iter(f["evidence_sha256"])));write(root/lineage.FREEZE_REL,f)
            with self.boundary(root,official),self.assertRaises(ValueError):lineage.build_teacher_reference(root=root)


if __name__=="__main__":unittest.main()
