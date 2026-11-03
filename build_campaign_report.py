"""Prepare or explicitly execute a transparent, composite campaign report.

Preparation uses only standard-library JSON/source checks and never starts a
kernel. Execution preserves the frozen six cells, executes only two independent
public-snapshot cells in a fresh CPU kernel, and appends two research PDF pages.
Publication always requires a subsequent human/agent visual and privacy review.
"""
from __future__ import annotations

import argparse
import base64
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent
SCHEMA = "campaign_report_artifact_manifest_v1"
REQUEST_SCHEMA = "campaign_report_cpu_request_v1"
LEASE_SCHEMA = "campaign_report_cpu_reservation_v1"
PROJECTION_SCHEMA = "audited_campaign_report_projection_v1"
PRIMARY_DIR = "outputs/primary_delivery_20261031"
FROZEN = {
    "build_report.py": "c9bccac8670c705810bc1c4b1bfdef336b8247ca80bead984b5a1d780eb79e28",
    "scripts/research_report_evidence.py": "888c8c509d5dc3034847bcdb9b8b8b213babe1e04e336345a94bdd920306d0ef",
    "notebooks/Project1_Results_Executed.ipynb": "1de5bb657623461404a173479a0a519bbd4e7ee99976300e23178c41894387db",
    PRIMARY_DIR+"/report_manifest.json": "e6a3ed08b3b0c365828608764031d6de4bb629171d39df929d84a21a72ba045c",
    PRIMARY_DIR+"/ky2738_Project1_Report.pdf": "c0e697bc1c668ebb4baf434a693a7856c6f3371c5c5476f5d2b7e986494e69a9",
    "research/process_tree_guard.py": "5a5f79365308a987330bb6a91064173e7deb63da8583294585f254ed7f8b3d80",
    "research/profile_augmix.py": "82af320b33b9778c2c32877e272d9e87a8265b70a313001d4775f267f7917c66",
}
SOURCE_NAMES = ("build_campaign_report.py", "scripts/campaign_report_evidence.py", "research/campaign-report-plan.md", "research/campaign.json")
NB_NAME = "Project1_Campaign_Executed.ipynb"
SNAPSHOT_NAME = "campaign_evidence.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def value_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON value")))


def write(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)+"\n", encoding="utf-8")


def write_snapshot(path, projection):
    Path(path).write_bytes(canonical(projection).encode("utf-8"))
    require(sha(path) == value_sha(projection), "External and embedded public snapshot bytes differ")


def now():
    return datetime.now(timezone.utc)


def public_check(value):
    """A second privacy boundary after the adapter's allowlisted projection."""
    if isinstance(value, dict):
        for key, item in value.items():
            public_check(key); public_check(item)
    elif isinstance(value, list):
        for item in value:
            public_check(item)
    elif isinstance(value, str):
        require(not re.search(r"(?:(?<![A-Za-z0-9])[A-Za-z]:[\\/]|(?<!\\)\\\\[^\\\s]+\\(?!\\)[^\\\s]+|file://|KGAT_[A-Za-z0-9]|Bearer\s+)", value), "Private path or credential-like text in public output")
        require(not value.startswith("/") and not re.search(r"/(?:home|Users|tmp|mnt|workspace)/", value), "Private absolute path in public output")
    elif isinstance(value, float):
        require(math.isfinite(value), "Nonfinite public scalar")
    else:
        require(value is None or type(value) in (bool, int), "Only public JSON values are supported")


def source_snapshot(root=ROOT):
    root = Path(root).resolve()
    require(all(sha(root/name) == digest for name, digest in FROZEN.items()), "Frozen primary/builder/adapter/guard changed")
    return {**FROZEN, **{name: sha(root/name) for name in SOURCE_NAMES}}


def original_cells(notebook):
    cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    require(len(cells) == 6 and all(type(cell.get("execution_count")) is int for cell in cells) and
        [cell.get("execution_count") for cell in cells] == list(range(1, 7)), "Original six actual execution counts required")
    require(all(isinstance(cell.get("outputs"), list) and cell["outputs"] and
        all(output.get("output_type") != "error" for output in cell["outputs"]) for cell in cells), "Original actual outputs must be complete and error-free")
    return cells


def preserved_original_cells(primary, composite):
    original = original_cells(primary)
    combined = [cell for cell in composite.get("cells", []) if cell.get("cell_type") == "code"]
    require(len(combined) == 8 and combined[:6] == original, "Composite must preserve all six original code-cell objects exactly")
    return combined[:6]


def load_inputs(campaign_manifest, root=ROOT):
    root = Path(root).resolve(); manifest = Path(campaign_manifest).resolve()
    sources = source_snapshot(root)
    primary_manifest = read(root/PRIMARY_DIR/"report_manifest.json")
    notebook_path = root/"notebooks/Project1_Results_Executed.ipynb"
    require(primary_manifest.get("notebook_executed") is True and type(primary_manifest.get("report_pages")) is int and
        1 <= primary_manifest["report_pages"] <= 6 and primary_manifest.get("sha256", {}).get("Project1_Results_Executed.ipynb") == sha(notebook_path) and
        primary_manifest.get("sha256", {}).get("ky2738_Project1_Report.pdf") == sha(root/PRIMARY_DIR/"ky2738_Project1_Report.pdf"), "Actual frozen primary manifest differs")
    primary = read(notebook_path); original_cells(primary)
    from scripts.campaign_report_evidence import load_campaign
    projection = load_campaign(manifest)
    require(projection.get("schema") == PROJECTION_SCHEMA and projection.get("source_manifest_sha256") == sha(manifest), "Audited campaign projection/source required")
    public_check(projection)
    return {"sources": sources, "primary": primary, "primary_manifest": primary_manifest,
        "projection": projection, "snapshot_sha256": value_sha(projection), "campaign_manifest_path": str(manifest),
        "campaign_manifest_sha256": sha(manifest)}


def new_cell(source, identity, kind="code"):
    result = {"cell_type": kind, "id": identity, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def research_cells(projection):
    payload = canonical(projection); digest = value_sha(projection)
    header = ("import hashlib, json, os\nos.environ['CUDA_VISIBLE_DEVICES'] = '-1'\n"+f"_payload = {payload!r}\n_expected_sha256 = {digest!r}\n"+
        "assert hashlib.sha256(_payload.encode('utf-8')).hexdigest() == _expected_sha256\n"+
        "campaign = json.loads(_payload)\n")
    first = header + (
        "print('EXPLORATORY CAMPAIGN SNAPSHOT:', campaign['recorded_at'])\n"
        "print('Snapshot SHA-256:', _expected_sha256)\n"
        "print('Fixed baseline:', json.dumps(campaign['baseline'], sort_keys=True))\n"
        "for row in campaign['rows']:\n    print('AGGREGATE:', json.dumps(row, sort_keys=True))\n"
        "for note in campaign.get('limitations', []):\n    print('LIMITATION:', note)\n"
        "for note in campaign.get('engineering', []):\n    print('ENGINEERING:', note)\n"
        "plot_rows = [r for r in campaign['rows'] if isinstance(r.get('clean_delta'), (int,float)) and isinstance(r.get('stress_delta'), (int,float))]\n"
        "if plot_rows:\n"
        "    import io\n    import matplotlib\n    matplotlib.use('Agg')\n"
        "    import matplotlib.pyplot as plt\n    from IPython.display import Image, display\n"
        "    fig, axes = plt.subplots(1, 2, figsize=(10, max(3, .48*len(plot_rows))), sharey=True)\n"
        "    for axis, metric in zip(axes, ('clean','stress')):\n"
        "        for y,row in enumerate(plot_rows):\n"
        "            interval = row.get(metric+'_ci95')\n"
        "            if interval is not None:\n                axis.hlines(y, 100*interval[0], 100*interval[1], color='#167e75')\n"
        "            axis.plot(100*row[metric+'_delta'], y, 'o', color='#173346')\n"
        "        axis.axvline(0, color='grey', linewidth=.8)\n"
        "        axis.set(title=metric.capitalize(), xlabel='Difference vs fixed flip baseline (pp)')\n"
        "        axis.set_yticks(range(len(plot_rows)), [r['label'] for r in plot_rows])\n"
        "    fig.suptitle('Reused development: conditional intervals when available')\n"
        "    fig.tight_layout()\n    buffer = io.BytesIO()\n"
        "    fig.savefig(buffer, format='png', dpi=160, bbox_inches='tight')\n"
        "    display(Image(data=buffer.getvalue()))\n    plt.close(fig)\n"
        "print('Incumbent comparisons, if present, are separate row fields and are not the plotted baseline intervals.')\n")
    second = header + (
        "for key in ('original_primary_public','original_best_public','best_new_public','best_observed_public'):\n"
        "    print(key+':', campaign.get(key))\n"
        "print('Unscored receipt count:', campaign.get('unscored_receipts_count'))\n"
        "for receipt in campaign.get('official_submissions', []):\n    print('OFFICIAL SCORED RECEIPT:', json.dumps(receipt, sort_keys=True))\n"
        "print('Sources:', json.dumps(campaign.get('sources', {}), sort_keys=True))\n"
        "print('A development result, prediction or intent is not a scored submission. Private scores remain unknown unless an official receipt supplies them.')\n"
        "print('The old six cells retain their original session counts and outputs. Only these two campaign cells were executed in the new CPU session.')\n")
    return [new_cell(first, "campaign-development"), new_cell(second, "campaign-official-scoring")]


def cell_hashes(cells):
    return [{"canonical_cell_object_sha256": value_sha(cell), "source_lines_canonical_sha256": value_sha(cell["source"]),
        "outputs_canonical_sha256": value_sha(cell["outputs"]), "execution_count": cell["execution_count"]} for cell in cells]


def kernel_cells(cells):
    """nbclient consumes string source; JSON archival objects retain line arrays."""
    result = copy.deepcopy(cells)
    for cell in result:
        require(cell.get("cell_type") == "code" and isinstance(cell.get("source"), list) and
            all(isinstance(line,str) for line in cell["source"]), "Explicit generated code-line arrays required")
        cell["source"] = "".join(cell["source"])
    return result


def validate_new_cells(cells):
    require(len(cells) == 2 and all(type(cell.get("execution_count")) is int for cell in cells) and
        [cell.get("execution_count") for cell in cells] == [1, 2], "Fresh two-cell session must retain actual counts 1,2")
    require(all(cell.get("cell_type") == "code" and isinstance(cell.get("outputs"), list) and cell["outputs"] and
        all(output.get("output_type") != "error" for output in cell["outputs"]) for cell in cells), "Two actual error-free output sets required")
    for cell in cells:
        public_check(cell)
        for output in cell["outputs"]:
            for key, value in output.get("data", {}).items():
                if key == "image/png":
                    raw = base64.b64decode("".join(value) if isinstance(value, list) else value, validate=True)
                    require(raw.startswith(b"\x89PNG\r\n\x1a\n"), "Actual PNG output signature required")


def compose(primary, projection, executed=None, session=None):
    before = copy.deepcopy(original_cells(primary))
    additions = research_cells(projection) if executed is None else copy.deepcopy(executed)
    if executed is not None:
        validate_new_cells(additions)
        expected = research_cells(projection)
        require([cell["source"] for cell in additions] == [cell["source"] for cell in expected], "Executed campaign sources differ from audited snapshot")
    notebook = copy.deepcopy(primary)
    execution_note = ("The following two campaign cells are an unexecuted draft; preparation starts no kernel. "
        "Explicit execution will use a separate fresh CPU kernel and retain its actual counts 1,2. " if executed is None else
        "The following two campaign cells used a separate fresh CPU kernel and retain that session's actual counts 1,2. ")
    intro = new_cell("## Campaign reporting: two explicit execution sessions\n\n"
        "The original six code cells preserve their original source, outputs, metadata and execution counts 1-6. "
        + execution_note + "This composite notebook is not eight newly executed cells. Both new cells embed the same audited public JSON snapshot; "
        "they can be rerun independently without private data, checkpoints or the original cell namespace. "
        "The external copy is `campaign_evidence.json` beside this notebook. Synthetic stress is not hidden-test accuracy.\n",
        "campaign-session-note", "markdown")
    notebook["cells"] += [intro, *additions]
    notebook.setdefault("metadata", {})["campaign_execution_sessions"] = {
        "composite": True, "all_eight_newly_executed": False,
        "original_notebook_sha256": FROZEN["notebooks/Project1_Results_Executed.ipynb"],
        "original_six_cells": cell_hashes(before), "snapshot_file": SNAPSHOT_NAME,
        "snapshot_sha256": value_sha(projection), "new_session": session or {"status": "not_executed"}}
    preserved_original_cells(primary, notebook)
    public_check(notebook)
    return notebook


def output_directory(path, root=ROOT):
    root = Path(root).resolve(); target = Path(path).resolve(); outputs = root/"outputs"
    require(target != outputs and target.is_relative_to(outputs), "A new child of project outputs is required")
    require(not target.is_relative_to(root/PRIMARY_DIR) and not any(p.name.startswith("research_report_preview_") for p in target.parents), "Frozen and reviewed output directories are protected")
    require(not target.exists(), "Report output directory must be new")
    return target


def base_manifest(inputs, status):
    return {"schema": SCHEMA, "status": status, "recorded_at": now().isoformat(), "publication_ready": False,
        "visual_and_privacy_review": "pending_root_review", "composite_notebook": True,
        "all_eight_cells_newly_executed": False, "original_six_cells_preserved_exactly": True,
        "original_six_comparison": "Exact parsed JSON cell-object equality; canonical object hashes are separate from the original notebook raw-byte SHA-256.",
        "frozen_sources_sha256": inputs["sources"], "snapshot_sha256": inputs["snapshot_sha256"],
        "campaign_source_manifest_sha256": inputs["campaign_manifest_sha256"],
        "original_six_cells": cell_hashes(original_cells(inputs["primary"])),
        "original_pdf_pages_recorded": inputs["primary_manifest"]["report_pages"],
        "gpu_training_seconds_claimed": 0, "private_scores_inferred": False}


def prepare(campaign_manifest, output_dir, root=ROOT):
    inputs = load_inputs(campaign_manifest, root); target = output_directory(output_dir, root)
    target.mkdir(parents=True)
    write_snapshot(target/SNAPSHOT_NAME, inputs["projection"])
    notebook = compose(inputs["primary"], inputs["projection"])
    write(target/"Project1_Campaign_Draft.ipynb", notebook)
    manifest = base_manifest(inputs, "prepared")
    manifest.update(kernel_started=False, pdf_rendered=False, new_session={"status": "not_executed"},
        artifacts_sha256={name: sha(target/name) for name in (SNAPSHOT_NAME, "Project1_Campaign_Draft.ipynb")})
    write(target/"report_manifest.json", manifest)
    return manifest


def report_deadline(root=ROOT):
    policy = read(Path(root)/"research/campaign.json")
    stop = datetime.fromisoformat(policy["stops_at"])
    require(stop.tzinfo is not None and stop.astimezone(timezone.utc) == datetime(2026,11,5,5,tzinfo=timezone.utc),
        "Original November 5 05:00 UTC experiment stop is fixed")
    from datetime import timedelta
    return stop+timedelta(minutes=30)


def assert_idle(root=ROOT):
    require(not (Path(root)/"results/research/local_lease.json").exists(), "An active or unknown local lease blocks report execution")
    response = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10)
    require(response.returncode == 0, "GPU idle state is unverified")
    lines = [line.strip() for line in response.stdout.splitlines() if line.strip()]
    require(not lines, "GPU compute processes or an unknown GPU response block execution")


def claim_cpu_lease(root, nonce):
    import psutil
    path = Path(root)/"results/research/local_lease.json"; path.parent.mkdir(parents=True, exist_ok=True)
    owner = {"schema": LEASE_SCHEMA, "purpose": "CPU report only; no GPU-training accounting", "nonce": nonce,
        "pid": os.getpid(), "process_created_at": psutil.Process().create_time(), "started_at": now().isoformat()}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(owner, stream); stream.flush(); os.fsync(stream.fileno())
    return path, owner


def release_cpu_lease(path, owner, receipt):
    from research.process_tree_guard import may_release_lease
    path = Path(path)
    if path.exists() and read(path) == owner:
        if may_release_lease(receipt): path.unlink()
        else: write(path, {**owner, "cleanup_requires_reconciliation": True, "process_tree_cleanup_verified": False})


def completed_kernel_guard(folder, receipt):
    from research.process_tree_guard import _valid_child_record
    require(receipt.get("process_tree_cleanup_verified") is True and len(receipt.get("process_tree_guard_results", [])) == 1,
        "Exactly one cleaned native kernel proof required")
    proof = receipt["process_tree_guard_results"][0]; directory = Path(proof["guard_directory"]).resolve()
    require(directory.parent == Path(folder).resolve() and directory.name == "process_tree_"+proof["nonce"] and
        proof.get("cleanup_verified") is True and proof.get("timed_out") is False and proof.get("completion_authenticated") is True and
        type(proof.get("completion_exit_code")) is int and proof["completion_exit_code"] == 0 and
        type(proof.get("child_returncode")) is int and proof["child_returncode"] == 0 and
        proof.get("guard_source_sha256") == FROZEN["research/process_tree_guard.py"] and read(directory/"host_result.json") == proof,
        "Native kernel completion/cleanup proof differs")
    spec = read(directory/"spec.json"); request_path = Path(folder)/"execution_request.json"
    require(spec.get("script") == str(Path(__file__).resolve()) and spec.get("script_sha256") == sha(__file__) and
        spec.get("arguments") == ["--_kernel", str(request_path), "--request-sha", sha(request_path)] and
        spec.get("nonce") == proof["nonce"] and spec.get("host_pid") == receipt["pid"] and
        spec.get("host_created_at") == receipt["process_created_at"] and spec.get("guard_source_sha256") == proof["guard_source_sha256"],
        "Native kernel proof is not bound to this owner/request/source")
    for name in ("ready.json", "completion.json"):
        record = read(directory/name)
        require(_valid_child_record(record, spec, proof["child_pid"], proof["child_created_at"]), "Native child identity/nonce/source differs")
        if name == "completion.json": require(type(record.get("exit_code")) is int and record["exit_code"] == 0, "Native completion exit must be zero")
    return {path.relative_to(folder).as_posix(): sha(path) for path in
        (request_path, *(directory/name for name in ("spec.json", "ready.json", "completion.json", "host_result.json")))}


def worker(request_path, request_sha):
    """Authenticate the exact native boundary before importing notebook packages."""
    import psutil
    from research.process_tree_guard import _valid_child_record, matching_process
    from research.profile_augmix import verify_os_boundary
    request_path = Path(request_path).resolve(); folder = request_path.parent
    require(request_path.name == "execution_request.json" and folder.is_relative_to(ROOT/"outputs") and sha(request_path) == request_sha, "Controlled unchanged CPU request required")
    request = read(request_path); process = psutil.Process()
    require(request.get("schema") == REQUEST_SCHEMA and request.get("sources_sha256") == source_snapshot() and
        os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and time.time() < request["deadline_unix"], "CPU environment/source/deadline required")
    owner = read(ROOT/"results/research/local_lease.json")
    require(owner == request["lease_owner"] and owner.get("schema") == LEASE_SCHEMA and
        matching_process(owner["pid"], owner["process_created_at"]) is not None, "Exact live CPU report lease owner required")
    started = datetime.fromisoformat(owner["started_at"])
    require(started.tzinfo is not None and type(request.get("deadline_unix")) in (int,float) and
        math.isfinite(request["deadline_unix"]) and request["deadline_unix"] <= min(started.timestamp()+240, report_deadline().timestamp()),
        "CPU request cannot extend its 240-second or campaign-closeout boundary")
    directories = list(folder.glob("process_tree_*")); require(len(directories) == 1, "Exactly one native report boundary required")
    spec = read(directories[0]/"spec.json"); ready = read(directories[0]/"ready.json")
    require(spec.get("script") == str(Path(__file__).resolve()) and spec.get("script_sha256") == sha(__file__) and
        spec.get("arguments") == ["--_kernel", str(request_path), "--request-sha", request_sha] and
        spec.get("host_pid") == owner["pid"] and spec.get("host_created_at") == owner["process_created_at"] and
        spec.get("guard_source_sha256") == FROZEN["research/process_tree_guard.py"] and
        type(spec.get("deadline_unix")) in (int,float) and time.time() < spec["deadline_unix"] <= request["deadline_unix"] and
        _valid_child_record(ready, spec, process.pid, process.create_time()), "Exact self source/argv/nonce/parent native proof required")
    verify_os_boundary(spec, ready, process)
    projection = read(folder/SNAPSHOT_NAME)
    require(value_sha(projection) == request["snapshot_sha256"], "Bound public snapshot changed")
    cells = research_cells(projection)
    from nbclient import NotebookClient
    import nbformat
    from jupyter_client.kernelspec import KernelSpecManager
    require(KernelSpecManager().get_kernel_spec("python3").env.get("CUDA_VISIBLE_DEVICES") in (None,"-1"),
        "Kernel specification may not override CPU-only CUDA visibility")
    notebook = nbformat.v4.new_notebook(cells=[nbformat.from_dict(cell) for cell in kernel_cells(cells)])
    notebook.metadata["kernelspec"] = {"name": "python3", "language": "python", "display_name": "Python 3"}
    started = now().isoformat()
    NotebookClient(notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(folder)}}).execute()
    actual = json.loads(nbformat.writes(notebook)); validate_new_cells(actual["cells"])
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and time.time() < request["deadline_unix"] and
        source_snapshot() == request["sources_sha256"], "CPU/source/deadline changed during notebook execution")
    write(folder/"campaign_two_cells_executed.ipynb", actual)
    session = {"schema": "campaign_two_cell_execution_session_v1", "session_id": request["nonce"],
        "status": "complete", "started_at": started, "finished_at": now().isoformat(), "fresh_kernel": True,
        "executed_code_cells": 2, "execution_counts": [cell["execution_count"] for cell in actual["cells"]],
        "errors": 0, "cuda_visible_devices": "-1", "old_cells_executed_in_this_session": False,
        "new_cells": cell_hashes(actual["cells"]), "executed_notebook_sha256": sha(folder/"campaign_two_cells_executed.ipynb")}
    write(folder/"new_session.json", session)


def percentage(value):
    return "Pending" if value is None else f"{100*value:.2f}%"


def render_research_pdf(projection, target):
    """Independent two-page renderer; no old static research-method narrative."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    style = ParagraphStyle("campaign", fontName="Helvetica", fontSize=8, leading=10, spaceAfter=5)
    title = ParagraphStyle("campaign-title", parent=style, fontName="Helvetica-Bold", fontSize=17, leading=20, spaceAfter=10)
    story = []
    def p(text, heading=False): story.append(Paragraph(text, title if heading else style))
    def table(rows, widths):
        wrapped = [[Paragraph(escape(str(value)), style) for value in row] for row in rows]
        value = Table(wrapped, colWidths=[part*inch for part in widths], repeatRows=1, hAlign="LEFT")
        value.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e8f1f2")),
            ("VALIGN", (0,0), (-1,-1), "TOP"), ("BOTTOMPADDING", (0,0), (-1,-1), 3),
            ("TOPPADDING", (0,0), (-1,-1), 3), ("LINEBELOW", (0,0), (-1,-1), .3, colors.HexColor("#cbd8df"))]))
        story.append(value); story.append(Spacer(1, 6))
    p("Five-day campaign: measured development", True)
    p("This appendix is exploratory and reuses the fixed official 5,000-image development partition. "
      "The original five report pages and six executed notebook cells remain the frozen original study. "
      "Its one-time confirmation was not reused. Five synthetic stress views are not CIFAR-C or hidden-test accuracy.")
    baseline = projection["baseline"]
    p("Fixed flip baseline: "+escape(json.dumps(baseline, sort_keys=True)))
    p("Primary gates follow each predeclared protocol. Original baseline, incumbent mean and per-case floor flags "
      "remain separate in the snapshot; a primary gate does not replace independent-repeat or delivery qualification.")
    rows = projection["rows"]; visible = rows[:8]
    values = [["Recipe / training", "Clean / NLL / stress", "Primary gate / repeat / delivery"]]
    for row in visible:
        loss = row.get("clean_nll"); loss = "Pending" if loss is None else f"{loss:.4f}"
        gate = "Passed" if row.get("gate_passed") is True else "Failed" if row.get("gate_passed") is False else "Pending"
        values.append([f"{row['label']}; {row.get('epochs','?')} epochs; {row.get('precision','?')}; {row.get('parameters','?')} parameters",
            f"{percentage(row.get('clean_accuracy'))} / {loss} / {percentage(row.get('stress_accuracy'))}; e{row.get('checkpoint_epoch','?')} {row.get('checkpoint_variant','?')}; {row.get('policy','?')}",
            f"{gate}; {row.get('qualification_status','')}; {row.get('repeat_status','not_verified')}; {row.get('delivery_status','not_promoted')}"])
    table(values, [2.15, 2.25, 2.55])
    if len(rows) > len(visible):
        p(f"PDF displays the first {len(visible)} of {len(rows)} audited rows in snapshot order. "
          "The complete snapshot and notebook retain every row, including: "+escape(", ".join(row["label"] for row in rows[len(visible):]))+".")
    p("Baseline-flip paired intervals, where available, use 10,000 original-image cluster bootstrap replicates; "
      "all five stress views stay together. Incumbent-policy comparisons are separate row fields. "
      "Conditional intervals after development reuse/postselection do not establish independent confirmation, seed variability or hidden performance.")
    p("Snapshot SHA-256: "+value_sha(projection))
    story.append(PageBreak())
    p("Campaign: official outcomes and limits", True)
    table([["Evidence", "Public accuracy"], ["Original primary", percentage(projection.get("original_primary_public"))],
        ["Best original", percentage(projection.get("original_best_public"))], ["Best newly scored research", percentage(projection.get("best_new_public"))],
        ["Best observed overall", percentage(projection.get("best_observed_public"))]], [4.95, 2.0])
    receipts = projection.get("official_submissions", []); values = [["Research run", "Reference", "Public / private"]]
    for receipt in receipts[:8]:
        values.append([receipt["run"], receipt["submission_ref"], f"{percentage(receipt.get('public_score'))} / {percentage(receipt.get('private_score'))}"])
    if not receipts: values.append(["No new scored receipt", "None", "Pending / unavailable"])
    table(values, [3.25, 1.45, 2.25])
    p(f"{len(receipts)} scored research receipts; {projection.get('unscored_receipts_count',0)} unscored records. "
      "The PDF displays the first eight receipts; the notebook and snapshot retain all. A prepared CSV, intent or partial repeat is not a score. "
      "Public best and original primary are distinct; private results are unavailable unless supplied by an official receipt.")
    for note in projection.get("limitations", [])[:5]: p(escape(str(note)))
    for note in projection.get("engineering", [])[:3]: p("Recorded scope: "+escape(str(note)))
    p("Only the two new campaign notebook cells execute in a fresh CPU session. Original counts 1-6 and outputs are preserved; "
      "new counts remain their actual 1,2. All methods, failures, pairing references and source fingerprints are retained in the public snapshot.")
    refs = projection.get("references", [])
    if refs: p("Additional method references: "+escape("; ".join(json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item) for item in refs)))
    p("Recorded at "+escape(str(projection["recorded_at"]))+". Public source: "+escape(str(projection.get("repository", ""))))
    SimpleDocTemplate(str(target), pagesize=letter, leftMargin=.55*inch, rightMargin=.55*inch,
        topMargin=.55*inch, bottomMargin=.55*inch, title="Five-day campaign appendix", author="Kevin Yang").build(story)
    from pypdf import PdfReader
    require(len(PdfReader(str(target)).pages) == 2, "Research appendix must occupy exactly two pages")


def combine_pdf(primary_pdf, research_pdf, output, expected_primary_pages):
    from pypdf import PdfReader, PdfWriter
    from reportlab.pdfgen import canvas
    primary, research = PdfReader(str(primary_pdf)), PdfReader(str(research_pdf))
    require(len(primary.pages) == expected_primary_pages, "Actual frozen PDF page count differs from its manifest")
    checked_page_count(len(primary.pages),len(research.pages))
    writer = PdfWriter(); pages = [*primary.pages, *research.pages]
    for index, page in enumerate(pages, 1):
        buffer = io.BytesIO(); layer = canvas.Canvas(buffer, pagesize=(float(page.mediabox.width), float(page.mediabox.height)))
        layer.setFillColorRGB(1,1,1); layer.rect(0, 0, float(page.mediabox.width), 32, fill=1, stroke=0)
        layer.setFillColorRGB(.1,.2,.27); layer.setFont("Helvetica", 7)
        layer.drawString(40, 19, "Saber | Kevin Yang (ky2738) | original study + campaign appendix")
        layer.drawRightString(float(page.mediabox.width)-40, 19, f"{index} / {len(pages)}")
        layer.save(); buffer.seek(0); page.merge_page(PdfReader(buffer).pages[0]); writer.add_page(page)
    with Path(output).open("wb") as stream: writer.write(stream)
    require(len(PdfReader(str(output)).pages) == len(pages) <= 8, "Final actual PDF exceeds eight pages")
    return len(pages)


def checked_page_count(primary_pages, research_pages):
    require(type(primary_pages) is int and 1 <= primary_pages <= 6 and type(research_pages) is int and research_pages == 2 and
        primary_pages+research_pages <= 8, "Actual primary/research/total PDF page counts exceed the audited allocation")
    return primary_pages+research_pages


def render_pages(pdf, folder, deadline):
    command = shutil.which("pdftoppm"); require(command is not None, "PDF rendering requires existing pdftoppm")
    folder.mkdir()
    remaining = deadline-time.time(); require(remaining > 0, "CPU report reservation exhausted before page rendering")
    subprocess.run([command, "-r", "120", "-png", str(pdf), str(folder/"page")], check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=min(120, remaining))
    return sorted(folder.glob("page-*.png"))


def execute(campaign_manifest, output_dir, root=ROOT):
    root = Path(root).resolve(); inputs = load_inputs(campaign_manifest, root); target = output_directory(output_dir, root)
    assert_idle(root)
    deadline = min(time.time()+240, report_deadline(root).timestamp())
    require(deadline-time.time() > 65, "Insufficient bounded CPU/report closeout time")
    target.mkdir(parents=True)
    nonce = uuid.uuid4().hex; lease, owner = claim_cpu_lease(root, nonce)
    receipt = {"schema": "campaign_report_cpu_supervisor_v1", "status": "preparing", "cpu_only": True,
        "gpu_training_seconds_claimed": 0, "pid": owner["pid"], "process_created_at": owner["process_created_at"],
        "started_at": now().isoformat(), "deadline_unix": deadline, "sources_sha256": inputs["sources"]}
    began = time.perf_counter()
    try:
        write_snapshot(target/SNAPSHOT_NAME, inputs["projection"])
        request = {"schema": REQUEST_SCHEMA, "nonce": nonce, "lease_owner": owner, "deadline_unix": deadline,
            "sources_sha256": inputs["sources"], "snapshot_sha256": inputs["snapshot_sha256"]}
        write(target/"execution_request.json", request); write(target/"supervisor.json", receipt)
        from research.process_tree_guard import run_guarded
        prior = os.environ.get("CUDA_VISIBLE_DEVICES"); os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            receipt["status"] = "executing_two_cpu_cells"
            run_guarded([sys.executable, str(Path(__file__).resolve()), "--_kernel", str(target/"execution_request.json"),
                "--request-sha", sha(target/"execution_request.json")], target/"kernel.log", deadline-time.time()-60,
                target/"supervisor.json", receipt, cwd=root)
        finally:
            if prior is None: os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else: os.environ["CUDA_VISIBLE_DEVICES"] = prior
        kernel_proof = completed_kernel_guard(target, receipt)
        actual = read(target/"campaign_two_cells_executed.ipynb"); session = read(target/"new_session.json")
        validate_new_cells(actual["cells"])
        require(session.get("schema") == "campaign_two_cell_execution_session_v1" and session.get("status") == "complete" and
            session.get("session_id") == nonce and session.get("fresh_kernel") is True and
            session.get("old_cells_executed_in_this_session") is False and session.get("cuda_visible_devices") == "-1" and
            type(session.get("executed_code_cells")) is int and session["executed_code_cells"] == 2 and
            session.get("execution_counts") == [1,2] and type(session.get("errors")) is int and session["errors"] == 0 and
            session.get("new_cells") == cell_hashes(actual["cells"]) and session.get("executed_notebook_sha256") == sha(target/"campaign_two_cells_executed.ipynb"), "Actual two-cell session evidence differs")
        notebook = compose(inputs["primary"], inputs["projection"], actual["cells"], session)
        write(target/NB_NAME, notebook)
        render_research_pdf(inputs["projection"], target/"campaign_appendix.pdf")
        pages = combine_pdf(root/PRIMARY_DIR/"ky2738_Project1_Report.pdf", target/"campaign_appendix.pdf",
            target/"ky2738_Project1_Report.pdf", inputs["primary_manifest"]["report_pages"])
        images = render_pages(target/"ky2738_Project1_Report.pdf", target/"page_renderings", deadline)
        require(len(images) == pages and time.time() <= deadline and source_snapshot(root) == inputs["sources"] and
            sha(inputs["campaign_manifest_path"]) == inputs["campaign_manifest_sha256"], "Actual page rendering/source/deadline audit failed")
        manifest = base_manifest(inputs, "complete")
        manifest.update(kernel_started=True, new_session=session, research_pdf_pages=2, report_pages=pages,
            rendered_pdf_pages=len(images), original_six_cells=cell_hashes(preserved_original_cells(inputs["primary"],notebook)),
            kernel_guard_evidence_sha256=kernel_proof,
            artifacts_sha256={name:sha(target/name) for name in (NB_NAME,SNAPSHOT_NAME,"campaign_appendix.pdf","ky2738_Project1_Report.pdf","new_session.json")},
            page_renderings_sha256={path.name:sha(path) for path in images})
        write(target/"report_manifest.json", manifest); receipt.update(status="complete", artifact="report_manifest.json", artifact_sha256=sha(target/"report_manifest.json"))
        return manifest
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        receipt.update(finished_at=now().isoformat(), elapsed_cpu_report_seconds=time.perf_counter()-began)
        write(target/"supervisor.json", receipt)
        release_cpu_lease(lease, owner, receipt)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--_kernel":
        require(len(argv) == 4 and argv[2] == "--request-sha", "Exact private CPU worker arguments required")
        return worker(argv[1], argv[3])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    actions = parser.add_mutually_exclusive_group(); actions.add_argument("--prepare", action="store_true"); actions.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = execute(args.campaign_manifest, args.output_dir) if args.execute else prepare(args.campaign_manifest, args.output_dir)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
