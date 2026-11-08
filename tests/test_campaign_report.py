"""Pure campaign-report contracts: no kernel, PDF, plotting or Torch execution."""
import contextlib
import copy
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import build_campaign_report as report


class CampaignReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for name in (*report.FROZEN, *report.SOURCE_NAMES):
            path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("synthetic source fixture\n")
        self.primary = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"language_info": {"name": "python"}},
            "cells": [report.new_cell("Original study\n", "original-title", "markdown")]}
        for index in range(1,7):
            cell = report.new_cell(f"print('original {index}')\n", f"original-{index}")
            cell.update(execution_count=index, metadata={"actual_old_session": index},
                outputs=[{"output_type": "stream", "name": "stdout", "text": [f"original output {index}\n"]}])
            self.primary["cells"].append(cell)
        report.write(self.root/"notebooks/Project1_Results_Executed.ipynb", self.primary)
        old_manifest = {"notebook_executed": True, "report_pages": 5, "sha256": {
            "Project1_Results_Executed.ipynb": report.sha(self.root/"notebooks/Project1_Results_Executed.ipynb"),
            "ky2738_Project1_Report.pdf": report.sha(self.root/report.PRIMARY_DIR/"ky2738_Project1_Report.pdf")}}
        report.write(self.root/report.PRIMARY_DIR/"report_manifest.json", old_manifest)
        self.pins = {name: report.sha(self.root/name) for name in report.FROZEN}
        (self.root/"research").mkdir(exist_ok=True)
        report.write(self.root/"research/campaign.json", {"stops_at": "2026-11-05T00:00:00-05:00"})
        self.campaign_manifest = self.root/"private_campaign_inputs.json"
        report.write(self.campaign_manifest, {"schema": "synthetic_private_input_fixture"})
        self.projection = {"schema": report.PROJECTION_SCHEMA, "recorded_at": "2026-11-03T00:00:00Z",
            "source_manifest_sha256": report.sha(self.campaign_manifest), "repository": "https://github.com/Solvement/example",
            "baseline": {"clean_accuracy": .96, "clean_nll": .12, "stress_accuracy": .77, "policy": "flip"},
            "rows": [], "official_submissions": [], "original_primary_public": .871, "original_best_public": .874,
            "best_new_public": .878, "best_observed_public": .878, "unscored_receipts_count": 0,
            "limitations": ["Synthetic fixture; not a measured competition result."], "engineering": [], "sources": {}}
        self.adapter = ModuleType("scripts.campaign_report_evidence")
        self.adapter.load_campaign = Mock(side_effect=lambda path: copy.deepcopy(self.projection))
        self.stack = contextlib.ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(report, "FROZEN", self.pins))
        self.stack.enter_context(patch.dict(sys.modules, {self.adapter.__name__: self.adapter}))

    def actual_new_cells(self):
        cells = report.research_cells(self.projection)
        for index, cell in enumerate(cells,1):
            cell.update(execution_count=index, outputs=[{"output_type":"stream", "name":"stdout", "text":["synthetic unit-test output\n"]}])
        return cells

    def test_prepare_is_json_only_and_preserves_every_original_cell_object(self):
        target = self.root/"outputs/prepare-new"
        with patch.object(report, "assert_idle") as idle, patch.object(report, "worker") as kernel, patch.object(report, "render_research_pdf") as pdf:
            manifest = report.prepare(self.campaign_manifest,target,self.root)
        idle.assert_not_called(); kernel.assert_not_called(); pdf.assert_not_called()
        draft = report.read(target/"Project1_Campaign_Draft.ipynb")
        cells = [cell for cell in draft["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(cells[:6], report.original_cells(self.primary))
        self.assertEqual([cell["execution_count"] for cell in cells], [1,2,3,4,5,6,None,None])
        self.assertIn("unexecuted draft", "".join(draft["cells"][-3]["source"]))
        self.assertIs(manifest["kernel_started"], False); self.assertIs(manifest["publication_ready"], False)
        self.assertFalse((target/"ky2738_Project1_Report.pdf").exists())
        self.assertEqual(manifest["snapshot_sha256"], report.value_sha(self.projection))
        self.assertEqual(manifest["snapshot_sha256"], report.sha(target/report.SNAPSHOT_NAME))

    def test_new_sources_are_independent_of_old_namespace_and_private_paths(self):
        self.projection["limitations"].append("Parameters ≤5M; CI ±0.2pp\nSecond public line.")
        cells = report.research_cells(self.projection)
        report.public_check(cells)
        for cell in cells:
            namespace = {}
            with patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()) as output:
                exec("".join(cell["source"]), namespace)
            self.assertEqual(namespace["campaign"], self.projection)
            self.assertNotIn("ROOT", namespace)
            self.assertNotIn(str(self.root), "".join(cell["source"]))
            self.assertTrue(output.getvalue())
        self.assertNotIn("nbclient", "".join(cells[0]["source"]))

    def test_composite_keeps_actual_two_session_counts_without_renumbering(self):
        original = copy.deepcopy(self.primary); cells = self.actual_new_cells()
        session = {"status":"complete", "execution_counts":[1,2], "errors":0}
        notebook = report.compose(self.primary,self.projection,cells,session)
        self.assertEqual(self.primary, original)
        combined = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(combined[:6], report.original_cells(original))
        self.assertEqual([cell["execution_count"] for cell in combined], [1,2,3,4,5,6,1,2])
        self.assertEqual(report.cell_hashes(report.preserved_original_cells(self.primary,notebook)),report.cell_hashes(report.original_cells(self.primary)))
        self.assertIs(notebook["metadata"]["campaign_execution_sessions"]["all_eight_newly_executed"], False)
        changed=copy.deepcopy(notebook);changed["cells"][1]["metadata"]={"changed":True}
        with self.assertRaisesRegex(ValueError,"preserve all six"):report.preserved_original_cells(self.primary,changed)
        cells[0]["source"] = ["print('changed')"]
        with self.assertRaisesRegex(ValueError,"sources differ"):
            report.compose(self.primary,self.projection,cells,session)

    def test_kernel_input_uses_string_source_without_altering_archival_cell_objects(self):
        cells=report.research_cells(self.projection); before=copy.deepcopy(cells)
        execution=report.kernel_cells(cells)
        self.assertEqual(cells,before)
        self.assertTrue(all(isinstance(cell["source"],str) for cell in execution))
        self.assertEqual([cell["source"] for cell in execution],["".join(cell["source"]) for cell in before])

    def test_renumbered_bool_error_and_non_png_new_outputs_fail_closed(self):
        for change in (lambda cells:cells[0].update(execution_count=7), lambda cells:cells[0].update(execution_count=True),
            lambda cells:cells[1].update(outputs=[{"output_type":"error","ename":"fixture"}]),
            lambda cells:cells[0].update(outputs=[{"output_type":"display_data","data":{"image/png":"bm90IHBuZw=="},"metadata":{}}])):
            cells=self.actual_new_cells(); change(cells)
            with self.assertRaises(ValueError): report.validate_new_cells(cells)

    def test_original_raw_source_and_manifest_hash_changes_stop_before_output(self):
        for name in ("build_report.py","scripts/research_report_evidence.py","notebooks/Project1_Results_Executed.ipynb",
                     report.PRIMARY_DIR+"/report_manifest.json",report.PRIMARY_DIR+"/ky2738_Project1_Report.pdf"):
            with self.subTest(name=name):
                path=self.root/name; raw=path.read_bytes(); path.write_bytes(raw+b"\n")
                target=self.root/"outputs/rejected"
                with self.assertRaisesRegex(ValueError,"Frozen"):
                    report.prepare(self.campaign_manifest,target,self.root)
                self.assertFalse(target.exists()); path.write_bytes(raw)

    def test_projection_source_hash_and_privacy_are_checked(self):
        self.projection["source_manifest_sha256"]="a"*64
        with self.assertRaisesRegex(ValueError,"projection/source"): report.load_inputs(self.campaign_manifest,self.root)
        self.projection["source_manifest_sha256"]=report.sha(self.campaign_manifest)
        for secret in ("C:/Users/person/private", "\\\\host\\private", "/home/person/token", "/opt/private", "//host/private", "see /home/person/private", "KGAT_example", "Bearer credential"):
            with self.subTest(secret=secret):
                self.projection["engineering"]=[secret]
                with self.assertRaises(ValueError): report.load_inputs(self.campaign_manifest,self.root)
        report.public_check({"repository":"https://github.com/Solvement/example","ref":"docs/aggregate.json"})

    def test_output_scope_and_frozen_directories_cannot_be_reused(self):
        for target in (self.root/"outputs", self.root/"notebooks/new", self.root/report.PRIMARY_DIR/"new",
                       self.root/"outputs/research_report_preview_20261101_b3/new"):
            with self.subTest(target=str(target)),self.assertRaises(ValueError): report.output_directory(target,self.root)
        existing=self.root/"outputs/already-present";existing.mkdir()
        with self.assertRaisesRegex(ValueError,"must be new"):report.output_directory(existing,self.root)

    def test_active_unknown_gpu_and_lease_block_without_starting_any_kernel(self):
        path=self.root/"results/research/local_lease.json";path.parent.mkdir(parents=True)
        report.write(path,{"status":"unknown"})
        with patch.object(report.subprocess,"run") as run:
            with self.assertRaisesRegex(ValueError,"lease"):report.assert_idle(self.root)
            run.assert_not_called()
        path.unlink()
        for stdout,code in (("1234\n",0),("[Not Supported]\n",0),("",1)):
            with patch.object(report.subprocess,"run",return_value=SimpleNamespace(stdout=stdout,returncode=code)):
                with self.assertRaises(ValueError):report.assert_idle(self.root)
        with patch.object(report.subprocess,"run",return_value=SimpleNamespace(stdout="",returncode=0)):
            report.assert_idle(self.root)

    def test_cleanup_unknown_retains_owned_lease_and_foreign_owner_is_untouched(self):
        path=self.root/"results/research/local_lease.json";path.parent.mkdir(parents=True)
        owner={"schema":report.LEASE_SCHEMA,"pid":1,"nonce":"owned"};report.write(path,owner)
        report.release_cpu_lease(path,owner,{"child_pid":22,"process_tree_cleanup_verified":False})
        self.assertIs(report.read(path)["cleanup_requires_reconciliation"],True)
        foreign={"schema":report.LEASE_SCHEMA,"pid":2,"nonce":"foreign"};report.write(path,foreign)
        report.release_cpu_lease(path,owner,{"process_tree_cleanup_verified":True})
        self.assertEqual(report.read(path),foreign)
        report.write(path,owner)
        report.release_cpu_lease(path,owner,{"process_tree_cleanup_verified":True})
        self.assertFalse(path.exists())

    def test_report_closeout_deadline_does_not_change_experiment_stop(self):
        self.assertEqual(report.report_deadline(self.root),datetime(2026,11,5,5,30,tzinfo=timezone.utc))
        self.assertEqual(report.read(self.root/"research/campaign.json")["stops_at"],"2026-11-05T00:00:00-05:00")
        report.write(self.root/"research/campaign.json",{"stops_at":"2026-11-06T00:00:00-05:00"})
        with self.assertRaisesRegex(ValueError,"is fixed"):report.report_deadline(self.root)

    def test_actual_pdf_page_count_contract_keeps_two_research_pages_and_eight_total(self):
        self.assertEqual(report.checked_page_count(5,2),7)
        self.assertEqual(report.checked_page_count(6,2),8)
        for primary,research in ((7,2),(5,3),(True,2),(5,True),(0,2)):
            with self.subTest(primary=primary,research=research),self.assertRaises(ValueError):
                report.checked_page_count(primary,research)

    def test_native_completion_requires_authenticated_matching_closed_proof(self):
        folder=self.root/"outputs/native-fixture";folder.mkdir()
        request_path=folder/"execution_request.json";report.write(request_path,{"synthetic":True})
        directory=folder/("process_tree_"+"a"*32);directory.mkdir()
        spec={"schema":"owned_python_process_tree_v1","nonce":"a"*32,"script":str(Path(report.__file__).resolve()),
            "script_sha256":report.sha(report.__file__),"arguments":["--_kernel",str(request_path),"--request-sha",report.sha(request_path)],
            "host_pid":101,"host_created_at":1.5,"guard_source_sha256":self.pins["research/process_tree_guard.py"]}
        proof={"schema":"owned_python_process_tree_v1","nonce":"a"*32,"guard_directory":str(directory),
            "cleanup_verified":True,"timed_out":False,"completion_authenticated":True,"completion_exit_code":0,
            "child_returncode":0,"guard_source_sha256":self.pins["research/process_tree_guard.py"],"child_pid":102,"child_created_at":2.5}
        child={"schema":"owned_python_process_tree_v1","nonce":spec["nonce"],"pid":102,"created_at":2.5,"script_sha256":spec["script_sha256"]}
        for name,value in (("spec.json",spec),("ready.json",child),("completion.json",{**child,"exit_code":0}),("host_result.json",proof)):report.write(directory/name,value)
        receipt={"pid":101,"process_created_at":1.5,"process_tree_cleanup_verified":True,"process_tree_guard_results":[proof]}
        self.assertEqual(len(report.completed_kernel_guard(folder,receipt)),5)
        for key,value in (("completion_authenticated",False),("timed_out",True),("completion_exit_code",True)):
            with self.subTest(key=key):
                bad=copy.deepcopy(receipt);bad["process_tree_guard_results"][0][key]=value
                with self.assertRaises(ValueError):report.completed_kernel_guard(folder,bad)
        report.write(directory/"completion.json",{**child,"nonce":"b"*32,"exit_code":0})
        with self.assertRaises(ValueError):report.completed_kernel_guard(folder,receipt)

    def test_cli_defaults_to_prepare_and_execution_requires_explicit_flag(self):
        with patch.object(report,"prepare",return_value={"status":"prepared"}) as prepare,patch.object(report,"execute") as execute,contextlib.redirect_stdout(io.StringIO()):
            report.main(["--campaign-manifest",str(self.campaign_manifest),"--output-dir",str(self.root/"outputs/cli")])
        prepare.assert_called_once();execute.assert_not_called()


@unittest.skipUnless(os.environ.get("CI") == "true", "Serialization dependency check runs only in existing Linux CI")
class CampaignNotebookSerializationTests(unittest.TestCase):
    def test_real_nbformat_keeps_string_kernel_input_and_archival_line_array_source(self):
        import nbformat
        projection={"schema":report.PROJECTION_SCHEMA,"recorded_at":"synthetic-fixture","baseline":{},"rows":[],
            "official_submissions":[],"limitations":["Unicode ≤5M and newline\nare public."],"engineering":[]}
        generated=report.research_cells(projection)
        notebook=nbformat.v4.new_notebook(cells=[nbformat.from_dict(cell) for cell in report.kernel_cells(generated)])
        self.assertTrue(all(isinstance(cell.source,str) for cell in notebook.cells))
        serialized=json.loads(nbformat.writes(notebook))
        self.assertEqual([cell["source"] for cell in serialized["cells"]],[cell["source"] for cell in generated])
        nbformat.validate(notebook)


if __name__ == "__main__":
    unittest.main()
