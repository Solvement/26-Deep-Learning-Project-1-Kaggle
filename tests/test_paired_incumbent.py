"""Pure contracts for the new paired identity, inputs, results and cleanup."""
import ast
from contextlib import ExitStack, redirect_stdout
import copy
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research import paired_incumbent as pair
from tests import test_incumbent_compare as fixtures


class PairedIncumbentContracts(unittest.TestCase):
    def owner_fixture(self, root):
        folder = root/"results/research/unit_pair"; folder.mkdir(parents=True)
        guard = root/"research/process_tree_guard.py"; guard.parent.mkdir(parents=True)
        guard.write_bytes((pair.ROOT/"research/process_tree_guard.py").read_bytes())
        nonce = "a"*32; directory = folder/("process_tree_"+nonce); directory.mkdir()
        current = 1793635200.
        receipt = dict(schema=pair.SCHEMA,status="benchmarking",mode="paired-incumbent",backend="local",run=folder.name,
                       input_manifest_sha256="b"*64,started_at=datetime.fromtimestamp(current-1,timezone.utc).isoformat(),
                       reserved_seconds=180,deadline_unix=current+170,pid=101,process_created_at=10.,child_pid=202,child_created_at=20.)
        spec = dict(schema="owned_python_process_tree_v1",nonce=nonce,host_pid=101,host_created_at=10.,cwd=str(root),
                    script=str(Path(pair.__file__).resolve()),script_sha256=pair.sha(pair.__file__),
                    guard_source_sha256=pair.PINS["research/process_tree_guard.py"],
                    arguments=["--_worker",str(folder),"--_manifest-sha","b"*64],deadline_unix=current+169,
                    job_name="Local\\SaberAugMix_"+nonce if os.name == "nt" else None)
        ready = dict(schema=spec["schema"],nonce=nonce,pid=202,created_at=20.,script_sha256=spec["script_sha256"],
                     method="windows_job_v1" if os.name == "nt" else "posix_session_v1")
        pair.write(directory/"spec.json",spec); pair.write(directory/"ready.json",ready)
        lease = root/"results/research/local_lease.json"; pair.write(lease,dict(pid=101,process_created_at=10.))
        process = SimpleNamespace(pid=202,create_time=lambda:20.,cmdline=lambda:[sys.executable,str(guard),"--bootstrap",str(directory/"spec.json")])
        parent = SimpleNamespace(pid=101,create_time=lambda:10.)
        return folder,receipt,spec,ready,process,parent,current,directory,lease

    def test_own_actual_guard_and_lease_contract_before_torch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); folder,receipt,spec,ready,process,parent,current,directory,lease = self.owner_fixture(root)
            with patch.object(pair,"ROOT",root),patch.object(pair.time,"time",return_value=current),patch.object(pair.guard_checks,"verify_os_boundary") as verify:
                pair.validate_owner(folder,"b"*64,{"run":folder.name},receipt,process,parent)
                verify.assert_called_once_with(spec,ready,process)
                for changed in ({**receipt,"process_created_at":11.},{**receipt,"child_created_at":21.},
                                {**receipt,"reserved_seconds":181},{**receipt,"deadline_unix":current-1},
                                {**receipt,"schema":"research_core_control_profile_supervisor_v1"}):
                    with self.assertRaises(ValueError): pair.validate_owner(folder,"b"*64,{"run":folder.name},changed,process,parent)
                pair.write(lease,dict(pid=101,process_created_at=10.,process_tree_cleanup_verified=False))
                with self.assertRaises(ValueError): pair.validate_owner(folder,"b"*64,{"run":folder.name},receipt,process,parent)

    def test_old_script_or_wrong_nonce_argv_or_bootstrap_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); folder,receipt,spec,ready,process,parent,current,directory,lease = self.owner_fixture(root)
            with patch.object(pair,"ROOT",root),patch.object(pair.time,"time",return_value=current),patch.object(pair.guard_checks,"verify_os_boundary"):
                for key,value in (("script",str(root/"research/profile_core_control.py")),("nonce","c"*32),
                                  ("arguments",["--_worker",str(folder),"--_manifest-sha","c"*64]),
                                  ("guard_source_sha256","c"*64),("deadline_unix",current+200)):
                    pair.write(directory/"spec.json",{**spec,key:value})
                    with self.assertRaises(ValueError): pair.validate_owner(folder,"b"*64,{"run":folder.name},receipt,process,parent)
                pair.write(directory/"spec.json",spec)
                bad = SimpleNamespace(pid=202,create_time=lambda:20.,cmdline=lambda:[sys.executable,str(root/"direct.py")])
                with self.assertRaises(ValueError): pair.validate_owner(folder,"b"*64,{"run":folder.name},receipt,bad,parent)

    def test_worker_calls_authentication_before_any_torch_numpy_or_runtime_import(self):
        before = set(sys.modules)
        with tempfile.TemporaryDirectory() as temp,self.assertRaises(ValueError): pair.worker(Path(temp),"b"*64)
        self.assertFalse({"torch","numpy","runtime"} & (set(sys.modules)-before))
        tree = ast.parse(Path(pair.__file__).read_text())
        worker = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == "worker")
        self.assertIsInstance(worker.body[0],ast.Assign)
        self.assertEqual(worker.body[0].value.func.id,"validate_worker")
        for node in tree.body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                name = node.module if isinstance(node,ast.ImportFrom) else node.names[0].name
                self.assertNotIn(name.split(".")[0],("torch","numpy","runtime","train","data","model"))

    def test_frozen_sources_and_accounting_are_torch_free(self):
        before = set(sys.modules); sources = pair.source_snapshot(); account = pair.control.accounting()
        self.assertEqual(sources["research/paired_compare_v2.py"],pair.PINS["research/paired_compare_v2.py"])
        self.assertEqual(set(account) & {"now","used_seconds"},{"now","used_seconds"})
        self.assertFalse({"torch","runtime","numpy"} & (set(sys.modules)-before))

    def test_actual_core_shadow_sam_resizemix_source_closures_are_fully_pinned(self):
        """Read real source declarations via AST; never import a Torch trainer."""
        before = set(sys.modules)
        core_tree = ast.parse((pair.ROOT/"train.py").read_text())
        provenance = next(n for n in core_tree.body if isinstance(n,ast.FunctionDef) and n.name == "code_provenance")
        core_map = next(n.value for n in provenance.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.DictComp))
        closures = {"core":ast.literal_eval(core_map.generators[0].iter)}
        for method in ("shadow","sam","resizemix"):
            tree = ast.parse((pair.ROOT/f"research/train_{method}.py").read_text())
            literals = {}
            for node in tree.body:
                if isinstance(node,ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0],ast.Name):
                    name = node.targets[0].id
                    if name in ("SOURCE_KEY","SOURCE_KEYS"):
                        value = node.value
                        literals[name] = (tuple(literals[e.id] if isinstance(e,ast.Name) else ast.literal_eval(e) for e in value.elts)
                                          if isinstance(value,ast.Tuple) else ast.literal_eval(value))
            closures[method] = literals["SOURCE_KEYS"]
        for method, names in closures.items():
            with self.subTest(method=method):
                self.assertTrue(names)
                self.assertTrue(set(names) <= set(pair.PINS),f"Unpinned actual {method} SOURCE_KEYS: {set(names)-set(pair.PINS)}")
                for name in names:
                    self.assertEqual(pair.sha(pair.ROOT/name),pair.PINS[name])
        self.assertFalse({"torch","runtime","numpy"} & (set(sys.modules)-before))

    def input_fixture(self, root):
        fixture = fixtures.IncumbentComparisonTests()
        outputs = []
        for name, incumbent in (("candidate",False),("incumbent",True)):
            folder = root/name; folder.mkdir(); checkpoint = folder/"best.pt"; checkpoint.write_bytes(name.encode())
            document = fixture.fixture(incumbent=incumbent); document["checkpoint_sha256"] = pair.sha(checkpoint)
            for filename,key,value in (("metrics.json","training_metrics_sha256",{"status":"complete"}),
                                       ("history.json","training_history_sha256",[{"epoch":300}])):
                pair.write(folder/filename,value); document[key] = pair.sha(folder/filename)
            pair.write(folder/"config.json",document["config"])
            benchmark = folder/"benchmark.json"; pair.write(benchmark,document)
            outputs.append((checkpoint,benchmark,document))
        return outputs

    def test_binding_hashes_actual_checkpoint_metrics_history_config_not_just_claims(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); candidate,incumbent = self.input_fixture(root)
            with ExitStack() as stack:
                stack.enter_context(patch.object(pair,"INCUMBENT_CHECKPOINT",incumbent[0]))
                stack.enter_context(patch.object(pair.aggregate,"INCUMBENT_PATH",incumbent[1]))
                stack.enter_context(patch.object(pair.aggregate,"INCUMBENT_CHECKPOINT_SHA",pair.sha(incumbent[0])))
                stack.enter_context(patch.object(pair.aggregate,"INCUMBENT_BENCHMARK_SHA",pair.sha(incumbent[1])))
                stack.enter_context(patch.object(pair.control,"official_inputs",return_value=([],{"test":"identity"})))
                binding,identity,summary = pair.binding_for(root,candidate[0],candidate[1],{})
                self.assertIn(str(candidate[0].resolve()),binding)
                self.assertIn(str(incumbent[0].resolve()),binding)
                self.assertEqual(identity,{"test":"identity"})
                self.assertEqual(summary["policy"],pair.POLICY)
                candidate[0].write_bytes(b"different")
                with self.assertRaises(ValueError): pair.binding_for(root,candidate[0],candidate[1],{})

    def result_fixture(self):
        fixture = fixtures.IncumbentComparisonTests()
        summary = pair.aggregate.compare_documents(fixture.fixture(),fixture.fixture(incumbent=True))
        manifest = dict(comparison=summary,source_sha256={"own.py":"b"*64},input_sha256={"private":"c"*64})
        row = dict(delta_accuracy=0.,percentile_ci95=[-.01,.01],candidate_only_correct=0,baseline_only_correct=0,
                   both_correct=4800,both_wrong=200,original_image_net_wins=0,original_image_net_losses=0)
        bootstrap = dict(original_images=5000,replicates=10000,seed=pair.SEED,
                         sampling_unit="original_image_with_all_five_stress_views",confidence_level=.95,
                         clean=row,stress={**row,"both_correct":20000,"both_wrong":5000})
        result = dict(schema=pair.RESULT_SCHEMA,status="complete",candidate_policy=pair.POLICY,incumbent_policy=pair.POLICY,
                      data_signature=pair.aggregate.DATA,evaluation_context=pair.aggregate.CONTEXT,
                      candidate_checkpoint_sha256=summary["candidate"]["checkpoint_sha256"],
                      incumbent_checkpoint_sha256=pair.aggregate.INCUMBENT_CHECKPOINT_SHA,
                      source_sha256=manifest["source_sha256"],input_sha256=manifest["input_sha256"],input_manifest_sha256="a"*64,
                      comparison=summary,shape=[5000,6],correctness_dtype="uint8_binary",raw_flags_persisted=False,
                      original_id_order_sha256="d"*64,development_index_order_sha256="e"*64,
                      candidate_correctness_sha256="f"*64,incumbent_correctness_sha256="1"*64,
                      original_ids_aligned=True,bootstrap=bootstrap,automatic_promotion=False,delivery_authorized=False,
                      confirmation_evaluated=False,hidden_images_loaded=False,external_data_loaded=False)
        return result,manifest

    def test_result_requires_complete_original_id_flags_hashes_and_no_delivery_authority(self):
        result,manifest = self.result_fixture(); pair.validate_result(result,manifest,"a"*64)
        for key,value in (("shape",[25000,1]),("original_ids_aligned",False),("raw_flags_persisted",True),
                          ("delivery_authorized",True),("automatic_promotion",True),("original_id_order_sha256","bad"),
                          ("evaluation_context",{"amp_dtype":"torch.float16"})):
            with self.assertRaises(ValueError): pair.validate_result({**result,key:value},manifest,"a"*64)

    def test_result_rejects_view_bootstrap_wrong_counts_delta_and_nonfinite_intervals(self):
        result,manifest = self.result_fixture()
        for key,value in (("original_images",25000),("replicates",9999),("seed",3),("sampling_unit","views")):
            changed = copy.deepcopy(result); changed["bootstrap"][key] = value
            with self.assertRaises(ValueError): pair.validate_result(changed,manifest,"a"*64)
        for key,value in (("both_correct",True),("both_wrong",201),("delta_accuracy",.1),("percentile_ci95",[float("nan"),.1])):
            changed = copy.deepcopy(result); changed["bootstrap"]["clean"][key] = value
            with self.assertRaises(ValueError): pair.validate_result(changed,manifest,"a"*64)

    def parent_patches(self, stack, root, used=0):
        policy = dict(stops_at="2026-11-05T00:00:00-05:00",concurrent_local_gpu_jobs=1,
                      local_gpu_seconds_per_utc_day=21600,local_job_timeout_seconds=10800)
        (root/"research").mkdir(); pair.write(root/"research/campaign.json",policy)
        current = datetime(2026,11,2,tzinfo=timezone.utc)
        stack.enter_context(patch.object(pair,"ROOT",root))
        stack.enter_context(patch.object(pair,"source_snapshot",return_value={}))
        stack.enter_context(patch.object(pair,"binding_for",return_value=({}, {}, {})))
        stack.enter_context(patch.object(pair.subprocess,"check_output",return_value="a"*40))
        stack.enter_context(patch.object(pair.control,"accounting",return_value={"now":lambda:current,"used_seconds":lambda *a:used}))
        stack.enter_context(patch.object(pair.time,"time",return_value=current.timestamp()))
        stack.enter_context(patch.object(pair,"validate_result"))

    def test_unknown_cleanup_retains_canonical_lease_and_fresh_run_evidence(self):
        from research.process_tree_guard import TreeCleanupError
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root = Path(temp); self.parent_patches(stack,root)
            def fail(command,log,remaining,receipt_path,receipt,**kwargs):
                receipt.update(child_pid=999,process_tree_cleanup_verified=False)
                raise TreeCleanupError("synthetic unknown cleanup")
            stack.enter_context(patch("research.process_tree_guard.run_guarded",side_effect=fail))
            with self.assertRaises(TreeCleanupError): pair.parent("unit",root/"best.pt",root/"benchmark.json",root)
            lease = pair.read(root/"results/research/local_lease.json")
            self.assertTrue(lease["cleanup_requires_reconciliation"])
            self.assertFalse(lease["process_tree_cleanup_verified"])
            self.assertEqual(pair.read(root/"results/research/unit/supervisor.json")["status"],"cleanup_unverified")

    def test_budget_exhaustion_and_foreign_lease_never_launch_or_reclaim(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root = Path(temp); self.parent_patches(stack,root,used=21600)
            with patch("research.process_tree_guard.run_guarded") as launch:
                with self.assertRaises(ValueError): pair.parent("unit",root/"best.pt",root/"benchmark.json",root)
                launch.assert_not_called(); self.assertFalse((root/"results/research/local_lease.json").exists())
            lease = root/"results/research/local_lease.json"; lease.parent.mkdir(parents=True); lease.write_text('{"pid": 42}')
            with self.assertRaises(ValueError): pair.parent("unit",root/"best.pt",root/"benchmark.json",root)
            self.assertEqual(lease.read_text(),'{"pid": 42}')

    def test_no_args_and_incomplete_worker_arguments_never_run(self):
        with patch.object(pair,"parent") as parent,patch.object(pair,"worker") as worker,redirect_stdout(io.StringIO()):
            pair.main([]); parent.assert_not_called(); worker.assert_not_called()
            with self.assertRaises(ValueError): pair.main(["--_manifest-sha","a"*64])


if __name__ == "__main__":
    unittest.main()
