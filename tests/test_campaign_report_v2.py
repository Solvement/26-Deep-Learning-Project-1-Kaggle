"""Pure v2 admission/schema tests; no kernel, numerical library or PDF execution."""
from contextlib import redirect_stdout,ExitStack
import copy
from datetime import datetime,timedelta,timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch
import build_campaign_report_v2 as report


class PureContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.folder=self.root/"results/research/example";self.folder.mkdir(parents=True)
        self.started=datetime.now(timezone.utc)-timedelta(hours=2);self.finished=self.started+timedelta(hours=1)
        self.record={"status":"complete","backend":"local","run":"example","pid":11,"process_created_at":1.5,
            "child_pid":12,"child_created_at":2.5,"watchdog_pid":13,"started_at":self.started.isoformat(),"finished_at":self.finished.isoformat()}
        self.save(self.folder/"supervisor.json",self.record)

    def save(self,path,value):
        Path(path).parent.mkdir(parents=True,exist_ok=True);report.write(path,value);return Path(path)

    def audit(self,table=None,**kwargs):return report.campaign_inactive(self.root,table={} if table is None else table,**kwargs)

    def native(self,status="complete",exit_code=0):
        script=self.root/"research/train_fixture.py";script.parent.mkdir(exist_ok=True);script.write_text("# synthetic controlled script\n")
        directory=self.folder/("process_tree_"+"a"*32);directory.mkdir()
        spec={"schema":"owned_python_process_tree_v1","nonce":"a"*32,"host_pid":11,"host_created_at":1.5,
            "script":str(script),"script_sha256":report.sha(script),"guard_source_sha256":report.old.FROZEN["research/process_tree_guard.py"]}
        child={"schema":spec["schema"],"nonce":spec["nonce"],"pid":12,"created_at":2.5,"script_sha256":spec["script_sha256"]}
        proof={"guard_directory":str(directory),"nonce":spec["nonce"],"cleanup_verified":True,"completion_authenticated":True,
            "completion_exit_code":exit_code,"child_returncode":exit_code,"child_pid":12,"child_created_at":2.5,
            "guard_source_sha256":spec["guard_source_sha256"],"timed_out":False}
        record={**self.record,"status":status,"process_tree_cleanup_verified":True,"process_tree_guard_results":[proof]}
        for path,value in ((directory/"spec.json",spec),(directory/"ready.json",child),(directory/"completion.json",{**child,"exit_code":exit_code}),
            (directory/"host_result.json",proof),(self.folder/"supervisor.json",record)):self.save(path,value)
        return record,directory,proof

    def test_01_import_is_pure(self):
        code="import sys;import build_campaign_report_v2;assert not {'torch','numpy','PIL','runtime','nbclient','nbformat','reportlab','pypdf','matplotlib','kaggle'}&set(sys.modules)"
        subprocess.check_output([sys.executable,"-c",code],cwd=report.ROOT,text=True)

    def test_02_default_adapter_scope_does_not_claim_kd_or200_support(self):
        descriptor=report.descriptor();self.assertEqual(descriptor["sha256"],report.DEFAULT_ADAPTER_SHA)
        self.assertIn("no WRN16-200 or KD support",descriptor["scope"])

    def test_03_original_adapter_cannot_be_overwritten_or_relabeled(self):
        with self.assertRaises(ValueError):report.descriptor(report.DEFAULT_ADAPTER,"a"*64,report.old.PROJECTION_SCHEMA,self.root)
        with self.assertRaises(ValueError):report.descriptor(report.DEFAULT_ADAPTER,report.DEFAULT_ADAPTER_SHA,"audited_campaign_report_projection_v2",self.root)

    def test_04_explicit_new_adapter_requires_all_three_fields(self):
        for args in (("scripts/campaign_report_evidence_v2.py",None,None),(None,"a"*64,None),(None,None,"audited_campaign_report_projection_v2")):
            with self.assertRaises(ValueError):report.descriptor(*args,root=self.root)

    def test_05_new_adapter_source_is_distinct_contained_and_versioned(self):
        item=report.descriptor("scripts/campaign_report_evidence_v10.py","a"*64,"audited_campaign_report_projection_v10",self.root)
        self.assertEqual(item["source"],"scripts/campaign_report_evidence_v10.py")
        for source in ("other/adapter.py","scripts/adapter.py","scripts/../outside.py"):
            with self.assertRaises(ValueError):report.descriptor(source,"a"*64,"audited_campaign_report_projection_v2",self.root)

    def test_06_adapter_closure_binds_actual_declared_local_reader_sources(self):
        script=self.root/"scripts/campaign_report_evidence_v2.py";script.parent.mkdir();script.write_text("from scripts import reader\n")
        helper=script.with_name("reader.py");helper.write_text("VALUE=1\n")
        item=report.descriptor(script,report.sha(script),"audited_campaign_report_projection_v2",self.root)
        closure=report.adapter_closure(item,self.root)
        self.assertEqual(set(closure),{"scripts/campaign_report_evidence_v2.py","scripts/reader.py"})
        helper.write_text("VALUE=2\n");self.assertNotEqual(report.adapter_closure(item,self.root),closure)

    def test_07_adapter_numerical_import_is_rejected_before_loading_it(self):
        script=self.root/"scripts/campaign_report_evidence_v2.py";script.parent.mkdir();script.write_text("def later():\n import torch\n")
        item=report.descriptor(script,report.sha(script),"audited_campaign_report_projection_v2",self.root)
        with self.assertRaisesRegex(ValueError,"numerical"):report.adapter_closure(item,self.root)

    def test_08_adapter_hidden_dynamic_import_is_not_a_closed_pure_reader(self):
        script=self.root/"scripts/campaign_report_evidence_v2.py";script.parent.mkdir();script.write_text("__import__('unknown')\n")
        item=report.descriptor(script,report.sha(script),"audited_campaign_report_projection_v2",self.root)
        with self.assertRaisesRegex(ValueError,"dynamic"):report.adapter_closure(item,self.root)

    def test_09_adapter_raw_sha_must_match(self):
        script=self.root/"scripts/campaign_report_evidence_v2.py";script.parent.mkdir();script.write_text("VALUE=1\n")
        item=report.descriptor(script,"a"*64,"audited_campaign_report_projection_v2",self.root)
        with self.assertRaisesRegex(ValueError,"raw source"):report.adapter_closure(item,self.root)

    def test_10_recorded_absent_pid_is_inactive_without_inventing_creation_time(self):
        self.assertEqual(report.inactive_identity(11,None,self.finished,{}),"pid_absent")

    def test_11_matching_live_identity_blocks(self):
        with self.assertRaises(ValueError):report.inactive_identity(11,1.5,self.finished,{11:{"created_at":1.5}})

    def test_12_reused_pid_is_not_the_old_instance(self):
        self.assertEqual(report.inactive_identity(11,1.5,self.finished,{11:{"created_at":9.5}}),"recorded_instance_replaced")

    def test_13_missing_old_stamp_requires_current_instance_after_finish(self):
        self.assertEqual(report.inactive_identity(13,None,self.finished,{13:{"created_at":self.finished.timestamp()+30}}),"instance_created_after_recorded_finish")

    def test_14_missing_stamp_same_historical_window_is_unknown(self):
        with self.assertRaises(ValueError):report.inactive_identity(13,None,self.finished,{13:{"created_at":self.started.timestamp()+20}})

    def test_15_protected_target_identity_is_unknown_and_blocks(self):
        with self.assertRaisesRegex(ValueError,"unreadable"):report.inactive_identity(11,1.5,self.finished,{11:{"created_at":None}})

    def test_16_native_dead_or_zombie_identity_is_inactive(self):
        self.assertEqual(report.inactive_identity(11,1.5,self.finished,{11:{"created_at":1.5,"state":"Z"}}),"native_dead_process")

    def test_17_any_unknown_canonical_lease_blocks_before_process_query(self):
        self.save(self.root/"results/research/local_lease.json",{"status":"unknown"})
        with patch.object(report,"process_table") as query:
            with self.assertRaisesRegex(ValueError,"lease"):report.campaign_inactive(self.root)
            query.assert_not_called()

    def test_18_historical_failed_terminal_legacy_job_is_not_eternally_busy(self):
        self.save(self.folder/"supervisor.json",{**self.record,"status":"failed"})
        result=self.audit();self.assertIs(result["local_campaign_workers_inactive"],True)
        self.assertIn("legacy_terminal",result["records"][0]["cleanup_basis"])

    def test_19_nonterminal_canonical_job_blocks_even_when_pid_absent(self):
        self.save(self.folder/"supervisor.json",{**self.record,"status":"training"})
        with self.assertRaisesRegex(ValueError,"terminal"):self.audit()

    def test_20_raw_training_history_needs_hash_bound_interrupted_attempt(self):
        raw=self.save(self.folder/"supervisor_raw_1.json",{**self.record,"status":"training"})
        self.save(self.folder/"supervisor_attempt_1.json",{**self.record,"status":"interrupted","original_receipt_sha256":report.sha(raw)})
        result=self.audit();self.assertEqual(result["supervisor_records"],3)
        self.assertIn("hash_bound_interrupted_attempt",{row["history_basis"] for row in result["records"]})

    def test_21_raw_history_cannot_be_silently_ignored_or_rewritten(self):
        self.save(self.folder/"supervisor_raw_1.json",{**self.record,"status":"training"})
        with self.assertRaisesRegex(ValueError,"reconciliation"):self.audit()

    def test_22_preserved_pre_cleanup_history_uses_same_canonical_terminal_job(self):
        self.save(self.folder/"supervisor_before_cleanup_reconciliation.json",{**self.record,"status":"cleanup_unverified"})
        self.assertEqual(self.audit()["supervisor_records"],2)

    def test_23_archive_identity_mismatch_is_not_a_valid_reconciliation(self):
        self.save(self.folder/"supervisor_before_cleanup_reconciliation.json",{**self.record,"pid":99,"status":"cleanup_unverified"})
        with self.assertRaisesRegex(ValueError,"same reconciled"):self.audit()

    def test_24_complete_and_failed_native_calculations_require_real_cleaned_proofs(self):
        self.native(status="failed",exit_code=1)
        result=self.audit();self.assertEqual(result["records"][0]["effective_status"],"failed")
        self.assertEqual(result["records"][0]["cleanup_basis"],"recorded_native_guard_cleanup_verified")

    def test_25_missing_tampered_or_unknown_native_cleanup_blocks(self):
        record,directory,proof=self.native()
        self.save(directory/"host_result.json",{**proof,"cleanup_verified":False})
        with self.assertRaisesRegex(ValueError,"host result"):self.audit()
        self.save(directory/"host_result.json",proof);(directory/"completion.json").unlink()
        with self.assertRaisesRegex(ValueError,"missing its child"):self.audit()
        self.save(self.folder/"supervisor.json",{**record,"process_tree_cleanup_verified":False})
        with self.assertRaisesRegex(ValueError,"unverified"):self.audit()

    def test_26_unrecorded_managed_local_command_blocks(self):
        table={90:{"pid":90,"created_at":9.,"name":"python.exe","command":f'python.exe "{self.root}/research/train_unknown.py"'}}
        with self.assertRaisesRegex(ValueError,"unrecorded"):self.audit(table)

    def test_27_graphics_browser_and_unreadable_unrelated_processes_do_not_imply_campaign_busy(self):
        table={90:{"pid":90,"created_at":None,"name":"explorer.exe","command":None},
            91:{"pid":91,"created_at":8.,"name":"chrome.exe","command":f'chrome {self.root}/train.py'},
            92:{"pid":92,"created_at":None,"name":"System","command":None}}
        self.assertEqual(self.audit(table)["global_gpu_idle"],"unverified")

    def test_28_stale_lease_missing_stamp_is_not_proven_reused_from_arbitrary_old_start(self):
        self.save(self.root/"results/research/local_lease_stale_1.json",{"pid":90})
        with self.assertRaises(ValueError):self.audit({90:{"created_at":self.started.timestamp()}})

    def test_29_only_exact_owned_v2_cpu_lease_may_be_present(self):
        owner={"schema":report.LEASE,"pid":101,"process_created_at":1.,"nonce":"owned"}
        self.save(self.root/"results/research/local_lease.json",owner)
        self.assertIs(self.audit(owned_lease=owner)["local_campaign_workers_inactive"],True)
        with self.assertRaises(ValueError):self.audit(owned_lease={**owner,"nonce":"foreign"})
        self.save(self.root/"results/research/cloud_lease.json",{"status":"unknown"})
        with self.assertRaises(ValueError):self.audit(owned_lease=owner)

    def test_30_unknown_history_filename_and_excluding_training_tree_are_refused(self):
        self.save(self.folder/"unrecognized_supervisor.json",self.record)
        with self.assertRaisesRegex(ValueError,"Unknown campaign"):self.audit()
        with self.assertRaises(ValueError):report.history_inventory(self.root,self.folder)

    def test_31_cloud_unknown_remote_allocation_is_not_rewritten_as_local_gpu_or_never_allocated(self):
        self.save(self.folder/"supervisor.json",{**self.record,"backend":"colab","runtime_stop_verified":False,"cleanup_outcome":"allocation_identity_unknown_no_stop"})
        result=self.audit();self.assertEqual(result["remote_gpu_allocation_status"],"not_inferred")
        self.assertEqual(result["records"][0]["scope"],"cloud_host_processes_only_remote_allocation_not_inferred")

    def test_32_nonempty_or_failed_global_nvidia_observations_are_never_execution_gates(self):
        for output,code in (("90\n91\n[Not Supported]\n",0),("",1)):
            with patch.object(report.subprocess,"run",return_value=SimpleNamespace(stdout=output,returncode=code)):
                result=report.global_observation();self.assertEqual(result["global_gpu_idle"],"unverified");self.assertIs(result["use_as_execution_gate"],False)
                self.assertNotIn("stdout",result)

    def test_33_current_actual_source_closure_still_pins_both_old_wrappers_and_adapter(self):
        source=report.source_snapshot(report.descriptor())
        self.assertEqual(source["build_campaign_report.py"],report.OLD_SHA)
        self.assertEqual(source[report.DEFAULT_ADAPTER],report.DEFAULT_ADAPTER_SHA)
        self.assertIn("build_campaign_report_v2.py",source);self.assertIn(report.PLAN,source)

    def test_34_prepare_does_not_query_processes_claim_lease_or_execute_old_worker(self):
        primary=report.old.read(report.ROOT/"notebooks/Project1_Results_Executed.ipynb")
        projection={"schema":report.old.PROJECTION_SCHEMA,"recorded_at":"synthetic fixture","baseline":{},"rows":[],"official_submissions":[],"limitations":[],"engineering":[]}
        inputs={"primary":primary,"primary_manifest":{"report_pages":5},"projection":projection,"sources":{},"snapshot_sha256":report.old.value_sha(projection),
            "campaign_manifest_sha256":"a"*64,"adapter":report.descriptor()}
        with patch.object(report,"load_inputs",return_value=inputs),patch.object(report,"campaign_inactive") as audit,patch.object(report,"claim_lease") as claim,patch.object(report,"worker") as worker:
            result=report.prepare("unused",self.root/"outputs/new-prepare",report.descriptor(),self.root)
        audit.assert_not_called();claim.assert_not_called();worker.assert_not_called()
        self.assertEqual(result["schema"],report.SCHEMA);self.assertIs(result["publication_ready"],False);self.assertIs(result["kernel_started"],False)
        notebook=report.read(self.root/"outputs/new-prepare/Project1_Campaign_Draft.ipynb")
        self.assertEqual(report.old.preserved_original_cells(primary,notebook),report.old.original_cells(primary))

    def test_35_cli_defaults_to_v2_prepare_only(self):
        with patch.object(report,"prepare",return_value={"status":"prepared"}) as prepare,patch.object(report,"execute") as execute,redirect_stdout(io.StringIO()):
            report.main(["--campaign-manifest","unused.json","--output-dir","unused"])
        prepare.assert_called_once();execute.assert_not_called()

    def test_36_unknown_dependency_cannot_hide_a_lazy_numerical_worker(self):
        script=self.root/"scripts/campaign_report_evidence_v2.py";script.parent.mkdir();script.write_text("from scripts import reader\n")
        script.with_name("reader.py").write_text("def worker():\n import torch\n")
        item=report.descriptor(script,report.sha(script),"audited_campaign_report_projection_v2",self.root)
        with self.assertRaisesRegex(ValueError,"numerical"):report.adapter_closure(item,self.root)

    def test_37_actual_reviewed_db2879_metadata_closure_is_bound_without_importing_workers(self):
        item=report.descriptor("scripts/campaign_report_evidence_v2.py","db2879de14c9bfbf3ff717cda355be4f19de187856dbb0a83f3ce6227750dff8","audited_campaign_report_projection_v2")
        source=report.adapter_closure(item)
        self.assertEqual(source["research/run_scratch_kd.py"],report.MIXED_SOURCE_PINS["research/run_scratch_kd.py"])
        self.assertIn("runtime.py",source);self.assertIn("scripts/campaign_report_evidence_v2.py",source)

    def test_38_auxiliary_duplicate_state_preserves_all_pid_members_but_canonical_duplicates_fail(self):
        path=self.root/"results/research/automation_state.json"
        path.write_text('{"job":{"pid":91,"process_created_at":1.5,"pid":92,"process_created_at":2.5}}')
        result=self.audit();self.assertEqual({item["pid"] for item in result["structured_state_identities"]},{91,92})
        self.assertEqual(len(result["structured_state_identities"]),4)
        (self.folder/"supervisor.json").write_text('{"status":"complete","status":"failed"}')
        with self.assertRaisesRegex(ValueError,"Duplicate"):self.audit()

    def test_39_full_pdf_presentation_keeps_all_eleven_rows_and_engineering_separate(self):
        rows=[{"label":f"row{i}","clean_accuracy":.96,"stress_accuracy":.8,"clean_nll":.1} for i in range(11)]
        projection={"rows":rows,"engineering_preflights":[{"method":"KD alpha0 hard control","median":.22108195,"admitted":False}]}
        measured,engineering=report.pdf_rows(projection)
        self.assertEqual(len(measured),11);self.assertIn("row10",measured[10][0]);self.assertEqual(len(engineering),1)
        self.assertIn("alpha0",engineering[0])
        self.assertEqual(report.checked_pages(5,2),7);self.assertEqual(report.checked_pages(5,3),8)
        for counts in ((6,3),(5,4),(True,2)):
            with self.assertRaises(ValueError):report.checked_pages(*counts)

    def test_40_v2_new_cells_embed_and_print_engineering_without_changing_original_six(self):
        primary=report.old.read(report.ROOT/"notebooks/Project1_Results_Executed.ipynb")
        projection={"schema":"audited_campaign_report_projection_v2","recorded_at":"synthetic-only","baseline":{},"rows":[],"official_submissions":[],
            "limitations":[],"engineering":[],"engineering_preflights":[{"method":"KD alpha0","formal_admitted":False}]}
        cells=report.research_cells(projection)
        namespace={}
        with patch.dict(os.environ),redirect_stdout(io.StringIO()) as output:exec("".join(cells[0]["source"]),namespace)
        self.assertIn("ENGINEERING PREFLIGHT ONLY",output.getvalue());self.assertIn("not positive-KD",output.getvalue())
        notebook=report.compose(primary,projection)
        self.assertEqual(report.old.preserved_original_cells(primary,notebook),report.old.original_cells(primary))
        self.assertEqual([cell["execution_count"] for cell in notebook["cells"] if cell["cell_type"]=="code"],[1,2,3,4,5,6,None,None])

    def test_41_unique_completed_v2_native_child_must_match_receipt_and_method(self):
        folder=self.root/"outputs/own";folder.mkdir(parents=True);request=self.save(folder/"execution_request.json",{"synthetic":True})
        directory=folder/("process_tree_"+"b"*32);directory.mkdir()
        method="windows_job_v1" if os.name=="nt" else "posix_session_v1"
        spec={"schema":"owned_python_process_tree_v1","nonce":"b"*32,"script":str(Path(report.__file__).resolve()),"script_sha256":report.sha(report.__file__),
            "arguments":["--_kernel",str(request),"--request-sha",report.sha(request)],"host_pid":101,"host_created_at":1.5,
            "guard_source_sha256":report.old.FROZEN["research/process_tree_guard.py"]}
        child={"schema":spec["schema"],"nonce":spec["nonce"],"pid":102,"created_at":2.5,"script_sha256":spec["script_sha256"],"method":method}
        proof={"nonce":spec["nonce"],"guard_directory":str(directory),"cleanup_verified":True,"timed_out":False,"completion_authenticated":True,
            "completion_exit_code":0,"child_returncode":0,"child_pid":102,"child_created_at":2.5,"method":method,"guard_source_sha256":spec["guard_source_sha256"]}
        for path,value in ((directory/"spec.json",spec),(directory/"ready.json",child),(directory/"completion.json",{**child,"exit_code":0}),(directory/"host_result.json",proof)):self.save(path,value)
        receipt={"pid":101,"process_created_at":1.5,"child_pid":102,"child_created_at":2.5,"process_tree_cleanup_verified":True,"process_tree_guard_results":[proof]}
        self.assertEqual(len(report.completed_boundary(folder,receipt)),5)
        for key,value in (("child_pid",103),("child_pid",True),("child_created_at",True),("child_created_at",float("nan"))):
            with self.subTest(key=key),self.assertRaises(ValueError):report.completed_boundary(folder,{**receipt,key:value})
        self.save(directory/"ready.json",{**child,"method":"other"})
        with self.assertRaisesRegex(ValueError,"method"):report.completed_boundary(folder,receipt)

    def test_42_own_pre_kernel_boundary_rejects_old_script_bad_parent_nonce_argv_and_cpu_env(self):
        import psutil
        from research import process_tree_guard,profile_augmix
        folder=self.root/"outputs/owned";folder.mkdir(parents=True);directory=folder/("process_tree_"+"c"*32);directory.mkdir()
        current=datetime.now(timezone.utc);owner={"schema":report.LEASE,"pid":101,"process_created_at":1.5,"started_at":current.isoformat(),"nonce":"owner"}
        request=self.save(folder/"execution_request.json",{"schema":report.REQUEST,"lease_owner":owner,"deadline_unix":current.timestamp()+180})
        digest=report.sha(request);self.save(self.root/"results/research/local_lease.json",owner)
        guard=self.root/"research/process_tree_guard.py";guard.parent.mkdir(exist_ok=True);guard.write_bytes((report.ROOT/"research/process_tree_guard.py").read_bytes())
        spec={"schema":"owned_python_process_tree_v1","nonce":"c"*32,"host_pid":101,"host_created_at":1.5,"cwd":str(self.root),
            "script":str(Path(report.__file__).resolve()),"script_sha256":report.sha(report.__file__),"arguments":["--_kernel",str(request),"--request-sha",digest],
            "guard_source_sha256":report.old.FROZEN["research/process_tree_guard.py"],"deadline_unix":current.timestamp()+170,
            "job_name":"Local\\SaberAugMix_"+"c"*32 if os.name=="nt" else None}
        ready={"schema":spec["schema"],"nonce":spec["nonce"],"pid":102,"created_at":2.5,"script_sha256":spec["script_sha256"],
            "method":"windows_job_v1" if os.name=="nt" else "posix_session_v1"}
        process=SimpleNamespace(pid=102,create_time=lambda:2.5,ppid=lambda:101,cmdline=lambda:[sys.executable,str(guard),"--bootstrap",str(directory/"spec.json")])
        def save():self.save(directory/"spec.json",spec);self.save(directory/"ready.json",ready)
        with ExitStack() as stack:
            stack.enter_context(patch.object(report,"ROOT",self.root));stack.enter_context(patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"-1"}))
            stack.enter_context(patch.object(psutil,"Process",return_value=process));stack.enter_context(patch.object(process_tree_guard,"matching_process",return_value=object()))
            boundary=stack.enter_context(patch.object(profile_augmix,"verify_os_boundary"));save()
            self.assertEqual(report.owned_boundary(folder,digest)[0]["schema"],report.REQUEST);boundary.assert_called_once()
            for target,key,value in ((spec,"script",str(report.ROOT/"build_campaign_report.py")),(spec,"arguments",["--old-worker"]),(spec,"nonce","d"*32),(ready,"pid",103),(ready,"method","other")):
                previous=target[key];target[key]=value;save()
                with self.subTest(key=key),self.assertRaises(ValueError):report.owned_boundary(folder,digest)
                target[key]=previous
            save()
            with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"0"}),self.assertRaises(ValueError):report.owned_boundary(folder,digest)
            process.ppid=lambda:999
            with self.assertRaises(ValueError):report.owned_boundary(folder,digest)


if __name__=="__main__":unittest.main()
