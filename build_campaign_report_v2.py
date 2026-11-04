"""CPU campaign report v2: campaign inactivity, not WDDM/global GPU idleness.

No numerical/render dependency is imported by preparation or process auditing.
Only explicit execution claims the canonical CPU lease and starts its own
authenticated, bounded worker. Existing report sources/artifacts stay frozen.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
import copy
import io
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parent
OLD_SHA="2be241e4b6d8f8723aee2733871a2073e5ec3e81ac3b0b9b40f0b3980c264849"
DEFAULT_ADAPTER="scripts/campaign_report_evidence.py"
DEFAULT_ADAPTER_SHA="d1f069f6bbecb5d3129729a07d1de11caf6a9094f291e80553481dd7a39ad2e1"
if hashlib.sha256((ROOT/"build_campaign_report.py").read_bytes()).hexdigest()!=OLD_SHA:
    raise ValueError("Frozen original campaign wrapper changed")
import build_campaign_report as old

SCHEMA="campaign_report_artifact_manifest_v2"
SUPERVISOR="campaign_report_cpu_supervisor_v2"
REQUEST="campaign_report_cpu_request_v2"
LEASE="campaign_report_cpu_reservation_v2"
PLAN="research/campaign-report-v2-plan.md"
TERMINAL={"complete","failed","interrupted","cancelled","aborted","rejected"}
FORBIDDEN_IMPORTS={"torch","numpy","PIL","runtime","data","model","train","nbclient","nbformat","reportlab","pypdf","matplotlib","kaggle"}
# Reviewed metadata controllers also contain dormant numerical workers. These
# exact source bytes may be hashed as dependencies; this does not permit their
# numerical workers or top-level numerical imports to run in the report host.
MIXED_SOURCE_PINS={
 "research/run_scratch_kd.py":"61ef86a809338d87cdbd4784b2df1917aa232217708151462ce3b2ea148f3ab7",
 "research/benchmark.py":"d30acccd66a18f8de198824a57b565cb8e16cd7044fa1cec6608007c80d37f16",
 "train.py":"4b69ef97c9bfd0a3e780520afb49123397ce1841944b21828c615ad41f6c21be",
 "data.py":"4bc8316c36bf4c229fc6b8d9327fcb37ebdbd5c5d414697b916633081f2fe50b",
 "runtime.py":"75849c8d87868cd2cd4de9e4657e516d3e7848f9bafd023d5d83c4fd473182e4",
 "model.py":"79b8af827af874385a7609ec9da7823df604144ae3896fd11fb956274531b848",
 "research/train_scratch_kd.py":"bd2b3c9c0f97ad9aa3d499a2033b3bba6c7b8c6805018dc9a17ef6aec55507a9",
 "research/paired_compare_v2.py":"7218d65bd0f33ae6e3563763730d6ce161a9e90131cdfb21c9feb10ad5bb5199",
 "research/benchmark_resizemix.py":"64508b91668b84627aecdce16f644c1512b94c9a6628470e5e4b2b8702590c40",
 "research/train_resizemix.py":"6fc49c6b76af09a753cb784b367c39e1be4ddb47b918e5cee8c2d3dfff298ae8",
 "research/train_shadow.py":"9fdf08ef341ca09202b452f488c092a6a239c2c46096a344f0540e5db6a9c061",
 "research/augmentations.py":"718530d2c56c556eb65508915126e0f5f0964a9c03d4937df74d3e3798551873",
 "research/resizemix.py":"042d909df03063abba4cf4e8cb9401b2f48c247734741780d97c9c0b30cbced1",
 "research/paired_compare.py":"034c850037b71ef151ad1f3f168afb3c475f170ad2a436d70e406e329fba5e65",
 "research/benchmark_sam.py":"ec7d93ff8aa05e658f0fb44170d43a76b1f8d0a711fe7320b82a9717a83a5376",
 "research/train_sam.py":"6c172b127ee0749e6973903c192156bcc2f0bb88c8c6c22b74f8a0d1e089d8f1",
 "research/sam.py":"682171d56134e82e14cd538bee67b7eec8dffa63c3c788b77e4636e5a09ad01c",
 "research/benchmark_shadow.py":"aab840a944125f45ef070099e70270103c62a5acabf081ead7f26cfdb6f7d6c9",
 "research/scratch_kd.py":"7bb624fd4fa2e6f3989b03e36c659ff69488b4945799c353ac329004e626c81b",
 "research/preflight_core_control.py":"85d24f743f547ff6914fc8e03cf3388394d3c70b4c7e7ebcf7d282948d16a761",
 "research/profile_augmix.py":"82af320b33b9778c2c32877e272d9e87a8265b70a313001d4775f267f7917c66",
 "research/train_augmix.py":"4b975f8132f0608d7088a1d66a13008b702dac0387abe9ed8ff8cc15735689b0",
 "research/augmix.py":"9786de01bcbe170d99f40d7681220732946348e933297b7bd600be8932e04f5f",
 "research/run_augmix_local.py":"61aee851bb6f2bbff757b83d80c8710c1437c330810f8408505227b31d9fe776",
 "research/run_local.py":"6c42cfd08aaad6b39c1acd23ed788044a68c31b047a677a0218308a1dd15a069"}
require,sha,read,write,now=old.require,old.sha,old.read,old.write,old.now


def descriptor(source=None,digest=None,schema=None,root=ROOT):
    root=Path(root).resolve()
    if source is None and digest is None and schema is None:
        return {"source":DEFAULT_ADAPTER,"sha256":DEFAULT_ADAPTER_SHA,"schema":old.PROJECTION_SCHEMA,
            "scope":"Frozen d1 adapter; no WRN16-200 or KD support is inferred."}
    require(source is not None and digest is not None and schema is not None,"Explicit new adapter requires source/rawSHA/schema together")
    path=Path(source);path=(root/path).resolve() if not path.is_absolute() else path.resolve()
    require(path.parent==root/"scripts" and path.suffix==".py" and re.fullmatch(r"[0-9a-f]{64}",digest or "") and
        re.fullmatch(r"audited_campaign_report_projection_v[1-9][0-9]*",schema or ""),"Explicit reviewed report-adapter interface required")
    if path==root/DEFAULT_ADAPTER:
        require(digest==DEFAULT_ADAPTER_SHA and schema==old.PROJECTION_SCHEMA,"The frozen d1 adapter cannot be relabeled or overwritten")
    else:require(re.fullmatch(r"campaign_report_evidence_v(?:[2-9]|[1-9][0-9]+)\.py",path.name),"Future adapters require a distinct versioned source")
    return {"source":path.relative_to(root).as_posix(),"sha256":digest,"schema":schema,
        "scope":"Explicit source pin/schema only; independent review, CI and actual integration remain required."}


def adapter_closure(adapter,root=ROOT):
    """Bind all static local sources; audit metadata imports, not dormant workers."""
    root=Path(root).resolve();selected=(root/adapter["source"]).resolve();pending=[selected];result={};trees={}
    def modules(node):
        if isinstance(node,ast.Import):return [alias.name for alias in node.names]
        if isinstance(node,ast.ImportFrom):
            return [node.module or ""]+[(node.module+"." if node.module else "")+alias.name for alias in node.names]
        return []
    def local(module,source,node):
        base=root
        if isinstance(node,ast.ImportFrom) and node.level:
            base=source.parent
            for _ in range(node.level-1):base=base.parent
            require(base.is_relative_to(root),"Adapter relative import escapes project source")
        path=(base/Path(*module.split("."))).with_suffix(".py")
        return path.resolve() if path.is_file() else None
    def eager_nodes(node):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.Lambda)):return
        yield node
        for child in ast.iter_child_nodes(node):yield from eager_nodes(child)
    while pending:
        path=pending.pop().resolve();name=path.relative_to(root).as_posix()
        if name in result:continue
        require(path.is_file() and not path.is_symlink(),"Actual non-symlink adapter source required")
        result[name]=sha(path);tree=ast.parse(path.read_text(encoding="utf-8-sig"));trees[path]=tree
        mixed=name in MIXED_SOURCE_PINS
        if mixed:require(result[name]==MIXED_SOURCE_PINS[name],"Reviewed mixed-purpose metadata dependency changed")
        for node in ast.walk(tree):
            for module in modules(node):
                if path==selected or not mixed:require(module.split(".")[0] not in FORBIDDEN_IMPORTS,"Unreviewed JSON dependency imports a numerical/runtime/API package")
                candidate=local(module,path,node)
                if candidate is not None:pending.append(candidate)
            if (path==selected or not mixed) and isinstance(node,ast.Call):
                require(not (isinstance(node.func,ast.Name) and node.func.id in {"exec","eval","__import__"}) and
                    not (isinstance(node.func,ast.Attribute) and node.func.attr in {"import_module","spec_from_file_location"}),
                    "Adapter dependencies must not hide dynamic source imports")
    # Every local module imported directly by the selected reader can execute
    # metadata code. Its top-level dependency closure must be pure. Function-
    # local worker imports are hash-bound above but are not executed by parsing.
    pending=[selected,*[p for node in ast.walk(trees[selected]) for module in modules(node) if (p:=local(module,selected,node)) is not None]];checked=set()
    while pending:
        path=pending.pop()
        if path in checked:continue
        checked.add(path)
        for node in eager_nodes(trees[path]):
            for module in modules(node):
                require(module.split(".")[0] not in FORBIDDEN_IMPORTS,"Metadata module top-level imports a numerical/runtime/API package")
                candidate=local(module,path,node)
                if candidate is not None:pending.append(candidate)
            if isinstance(node,ast.Call):
                require(not (isinstance(node.func,ast.Name) and node.func.id in {"exec","eval","__import__"}) and
                    not (isinstance(node.func,ast.Attribute) and node.func.attr in {"import_module","spec_from_file_location"}),"Metadata module has a top-level dynamic source import")
    require(result[adapter["source"]]==adapter["sha256"],"Actual adapter raw source differs from explicit pin")
    return result


def source_snapshot(adapter,root=ROOT):
    root=Path(root).resolve();pins={**old.FROZEN,"build_campaign_report.py":OLD_SHA,
        DEFAULT_ADAPTER:DEFAULT_ADAPTER_SHA,"research/campaign-report-plan.md":"6e5a5bed9c0a6f219a25ee56b3bca80bb4ecf16914cde7e7afeede5b8a71b625"}
    require(all(sha(root/name)==value for name,value in pins.items()),"Frozen original report/guard sources or primary artifacts changed")
    return {**pins,**adapter_closure(adapter,root),**{name:sha(root/name) for name in
        ("build_campaign_report_v2.py",PLAN,"research/campaign.json")}}


def load_inputs(manifest_path,adapter,root=ROOT):
    root=Path(root).resolve();manifest_path=Path(manifest_path).resolve();sources=source_snapshot(adapter,root)
    primary=read(root/"notebooks/Project1_Results_Executed.ipynb");old.original_cells(primary)
    primary_manifest=read(root/old.PRIMARY_DIR/"report_manifest.json")
    require(primary_manifest.get("notebook_executed") is True and type(primary_manifest.get("report_pages")) is int and
        1<=primary_manifest["report_pages"]<=6 and primary_manifest["sha256"]["Project1_Results_Executed.ipynb"]==sha(root/"notebooks/Project1_Results_Executed.ipynb") and
        primary_manifest["sha256"]["ky2738_Project1_Report.pdf"]==sha(root/old.PRIMARY_DIR/"ky2738_Project1_Report.pdf"),"Actual frozen primary manifest required")
    imports_before=set(sys.modules)
    module=importlib.import_module(adapter["source"][:-3].replace("/","."))
    require(Path(module.__file__).resolve()==root/adapter["source"],"Selected adapter module does not match its reviewed source")
    projection=module.load_campaign(manifest_path)
    require(not (FORBIDDEN_IMPORTS & (set(sys.modules)-imports_before)),"Actual report metadata loading imported a numerical/runtime/API module")
    require(projection.get("schema")==adapter["schema"] and projection.get("source_manifest_sha256")==sha(manifest_path) and
        isinstance(projection.get("rows"),list) and isinstance(projection.get("official_submissions"),list) and
        isinstance(projection.get("baseline"),dict),"Actual selected-adapter projection/common presentation contract differs")
    require(isinstance(projection.get("engineering_preflights",[]),list) and
        all(isinstance(item,dict) for item in projection.get("engineering_preflights",[])),"Explicit engineering preflights must be a public list of aggregate objects")
    old.public_check(projection);require(source_snapshot(adapter,root)==sources,"Adapter source closure changed during projection")
    return {"sources":sources,"primary":primary,"primary_manifest":primary_manifest,"projection":projection,
        "snapshot_sha256":old.value_sha(projection),"campaign_manifest_path":str(manifest_path),"campaign_manifest_sha256":sha(manifest_path),"adapter":adapter}


def _stamp(value):
    require(isinstance(value,str),"Recorded aware process/job time required")
    result=datetime.fromisoformat(value);require(result.tzinfo is not None,"Recorded aware process/job time required");return result


def process_table():
    """Read native identities/commands into memory; never publish raw commands."""
    if os.name=="nt":
        script=("$taskRows=@(Get-CimInstance Win32_Process | ForEach-Object {"
            "[pscustomobject]@{pid=[int]$_.ProcessId;created_at=$(if ($_.CreationDate) {"
            "([DateTimeOffset]$_.CreationDate.ToUniversalTime()).ToUnixTimeMilliseconds()/1000.0} else {$null});"
            "name=$_.Name;command=$_.CommandLine}});ConvertTo-Json -InputObject $taskRows -Compress -Depth 3")
        response=subprocess.run(["powershell.exe","-NoProfile","-NonInteractive","-Command",script],capture_output=True,
            encoding="utf-8",errors="replace",timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
        require(response.returncode==0,"Native process inventory is unavailable")
        rows=json.loads(response.stdout)
    else:
        proc=Path("/proc");require(proc.is_dir(),"Native process inventory is unsupported")
        boot=next(float(line.split()[1]) for line in (proc/"stat").read_text().splitlines() if line.startswith("btime "))
        hz=os.sysconf("SC_CLK_TCK");rows=[]
        for directory in proc.iterdir():
            if not directory.name.isdigit():continue
            try:
                raw=(directory/"stat").read_text();parts=raw[raw.rfind(")")+2:].split()
                rows.append({"pid":int(directory.name),"created_at":boot+float(parts[19])/hz,"state":parts[0],
                    "name":raw[raw.find("(")+1:raw.rfind(")")],"command":(directory/"cmdline").read_bytes().replace(b"\0",b" ").decode("utf-8",errors="replace")})
            except FileNotFoundError:continue
            except PermissionError:rows.append({"pid":int(directory.name),"created_at":None,"command":None,"name":None})
    require(isinstance(rows,list),"Native process inventory must be a list")
    table={}
    for row in rows:
        require(type(row.get("pid")) is int and row["pid"]>=0 and row["pid"] not in table,"Ambiguous native PID inventory")
        table[row["pid"]]=row
    return table


def inactive_identity(pid,created,finished,table):
    require(type(pid) is int and pid>0,"Actual positive recorded process ID required")
    if created is not None:require(type(created) in (int,float) and math.isfinite(created) and created>0,"Recorded process creation time is invalid")
    row=table.get(pid)
    if row is None:return "pid_absent"
    actual=row.get("created_at")
    require(type(actual) in (int,float) and math.isfinite(actual) and actual>0,"Campaign process identity is unreadable/unknown")
    if created is not None and abs(actual-created)>=.01:return "recorded_instance_replaced"
    if created is None and actual>finished.timestamp()+.01:return "instance_created_after_recorded_finish"
    if row.get("state") in {"Z","X"}:return "native_dead_process"
    raise ValueError("A recorded campaign process is active or its historical identity remains unknown")


def history_inventory(root=ROOT,exclude_output=None):
    root=Path(root).resolve();paths=list((root/"results/research").rglob("*supervisor*.json"))
    if (root/"outputs").is_dir():paths+=list((root/"outputs").rglob("supervisor*.json"))
    if exclude_output is not None:
        excluded=Path(exclude_output).resolve();require(excluded.is_relative_to(root/"outputs") and excluded!=root/"outputs","Only the exact owned CPU-report output can be excluded")
        paths=[path for path in paths if not path.resolve().is_relative_to(excluded)]
    result={}
    for path in sorted(paths):
        require(not path.is_symlink(),"Supervisor symlinks are unsupported")
        result[path.resolve()]=read(path)
    require(result,"Campaign supervisor history is missing")
    return result


def effective_receipt(path,record,records):
    if path.name=="supervisor.json" or re.fullmatch(r"supervisor_attempt_[0-9]+\.json",path.name):return record,"original_terminal_receipt"
    if re.fullmatch(r"supervisor_raw_[0-9]+\.json",path.name):
        matches=[value for candidate,value in records.items() if candidate.parent==path.parent and
            re.fullmatch(r"supervisor_attempt_[0-9]+\.json",candidate.name) and value.get("original_receipt_sha256")==sha(path) and value.get("status") in TERMINAL]
        require(len(matches)==1,"Raw supervisor lacks its exact terminal reconciliation")
        effective=matches[0];basis="hash_bound_interrupted_attempt"
    elif path.name in {"supervisor_before_cleanup_reconciliation.json","supervisor_before_allocation_reconciliation.json"}:
        canonical=path.with_name("supervisor.json");require(canonical in records,"Historical receipt lacks its canonical record")
        effective=records[canonical];basis="preserved_pre_reconciliation_history"
    else:raise ValueError("Unknown campaign supervisor history format")
    require(all(record.get(k)==effective.get(k) for k in ("pid","process_created_at","started_at","run","session","input_bundle_sha256")),
        "Archived receipt does not identify the same reconciled job")
    return effective,basis


def native_cleanup(record,folder,root):
    flag=record.get("process_tree_cleanup_verified")
    require(record.get("cleanup_requires_reconciliation") is not True and record.get("status")!="cleanup_unverified","Campaign cleanup remains unknown")
    if flag is None:return [],{},"legacy_terminal_and_recorded_identity_check_only"
    require(flag is True,"Native campaign cleanup is unverified")
    proofs=record.get("process_tree_guard_results");require(isinstance(proofs,list) and proofs,"Native cleanup flag needs its actual guard proofs")
    identities=[];bindings={}
    for proof in proofs:
        directory=Path(proof.get("guard_directory","")).resolve()
        require(directory.parent==folder and directory.name=="process_tree_"+str(proof.get("nonce")) and
            re.fullmatch(r"[0-9a-f]{32}",proof.get("nonce","") or "") and proof.get("cleanup_verified") is True and
            proof.get("guard_source_sha256")==old.FROZEN["research/process_tree_guard.py"],"Owned native cleanup proof is incomplete")
        host=directory/"host_result.json";spec_path=directory/"spec.json"
        require(read(host)==proof,"Archived native host result differs from its receipt")
        spec=read(spec_path)
        require(spec.get("schema")=="owned_python_process_tree_v1" and spec.get("nonce")==proof["nonce"] and
            spec.get("host_pid")==record.get("pid") and abs(spec.get("host_created_at",0)-record.get("process_created_at",-1))<.01 and
            spec.get("guard_source_sha256")==proof["guard_source_sha256"],"Native cleanup owner/nonce/source differs")
        script=Path(spec.get("script","")).resolve()
        require(script.is_relative_to(root) and script.suffix==".py" and script.is_file() and sha(script)==spec.get("script_sha256"),
            "Archived native controlled source changed or is unknown")
        bindings[script.relative_to(root).as_posix()]=sha(script)
        bindings.update({host.relative_to(root).as_posix():sha(host),spec_path.relative_to(root).as_posix():sha(spec_path)})
        if record["status"]=="complete":
            require(proof.get("completion_authenticated") is True and type(proof.get("completion_exit_code")) is int and proof["completion_exit_code"]==0,
                "Complete native job lacks authenticated successful completion")
        if proof.get("child_pid") is not None:identities.append((proof["child_pid"],proof.get("child_created_at"),"native_child"))
        for name in ("ready.json","completion.json"):
            path=directory/name
            if not path.exists():
                require(record["status"]!="complete","Complete native job is missing its child records")
                continue
            value=read(path);bindings[path.relative_to(root).as_posix()]=sha(path)
            require(value.get("schema")=="owned_python_process_tree_v1" and value.get("nonce")==proof["nonce"] and
                value.get("script_sha256")==spec.get("script_sha256") and value.get("pid")==proof.get("child_pid") and
                type(value.get("created_at")) in (int,float) and type(proof.get("child_created_at")) in (int,float) and
                abs(value["created_at"]-proof["child_created_at"])<.01,"Native child source/nonce/identity differs")
            if name=="completion.json" and record["status"]=="complete":
                require(type(value.get("exit_code")) is int and value["exit_code"]==0,"Actual complete native child exit differs")
            if value.get("watchdog_pid") is not None:identities.append((value["watchdog_pid"],value.get("watchdog_created_at"),"native_watchdog"))
    return identities,bindings,"recorded_native_guard_cleanup_verified"


def canonical_leases(root=ROOT):
    base=Path(root)/"results/research"
    return sorted(path.resolve() for path in base.glob("*lease.json"))


def state_history(root,table,finish_bounds):
    """Check structured historical IDs, including legacy children overwritten later."""
    class Members(list):pass
    evidence={};rows=[]
    def walk(value,location,path):
        if isinstance(value,Members):
            occurrences={}
            for key,pid in value:
                number=occurrences.get(key,0);occurrences[key]=number+1
                item_location=location+"/"+key+f"#{number}"
                if (key=="pid" or key.endswith("_pid")) and type(pid) is int and pid>0:
                    prefix=key[:-4] if key!="pid" else ""
                    if prefix.startswith(("remote_","guest_")):continue
                    candidates=(["process_created_at","created_at"] if not prefix else
                        [prefix+suffix for suffix in ("_created_at","_create_time","_creation_time","_created_at_observed","_actual_creation_time_verified")])
                    created_values=[item for name,item in value if name in candidates and type(item) in (int,float)] or [None]
                    dates=[finish_bounds[pid]] if pid in finish_bounds else []
                    for name,item in value:
                        if name in {"finished_at","ended_at"} and isinstance(item,str):
                            try:dates.append(_stamp(item))
                            except ValueError:dates.append(now())
                    finished=max(dates) if dates else now()
                    for created in created_values:
                        decision=inactive_identity(pid,created,finished,table)
                        rows.append({"state_source":path.relative_to(root).as_posix(),"json_location":item_location,
                            "pid":pid,"recorded_created_at":created,"state":decision})
                if isinstance(pid,list):walk(pid,item_location,path)
        elif isinstance(value,list):
            for index,item in enumerate(value):walk(item,location+f"/{index}",path)
    for path in sorted((root/"results/research").glob("automation_state*.json")):
        require(not path.is_symlink(),"Automation-state history symlinks are unsupported")
        raw=path.read_bytes()
        value=json.loads(raw.decode("utf-8-sig"),object_pairs_hook=Members,
            parse_constant=lambda value:(_ for _ in ()).throw(ValueError("Nonfinite auxiliary history JSON")))
        evidence[path.relative_to(root).as_posix()]=hashlib.sha256(raw).hexdigest();walk(value,"",path)
    return rows,evidence


def campaign_inactive(root=ROOT,table=None,owned_lease=None,exclude_output=None,allowed_processes=()):
    """Audit every recorded local identity; preserve remote/global uncertainty."""
    root=Path(root).resolve();leases=canonical_leases(root)
    expected=root/"results/research/local_lease.json"
    if owned_lease is None:require(not leases,"An active/unknown canonical campaign lease blocks CPU reporting")
    else:require(leases==[expected] and read(expected)==owned_lease and owned_lease.get("schema")==LEASE,"Only the exact current CPU report lease may remain")
    records=history_inventory(root,exclude_output);table=process_table() if table is None else table
    evidence={path.relative_to(root).as_posix():sha(path) for path in records};rows=[];identity_count=0;finish_bounds={}
    for path,record in records.items():
        effective,basis=effective_receipt(path,record,records)
        require(effective.get("status") in TERMINAL,"Campaign job has not reached a terminal/reconciled status")
        started,finished=_stamp(effective.get("started_at")),_stamp(effective.get("finished_at"))
        require(started<=finished<=now(),"Campaign job timestamps are incomplete or in the future")
        remote=(effective.get("backend")=="colab" or effective.get("resource_policy")=="existing_Colab_entitlement_only_no_purchases")
        extra,proofs,cleanup=native_cleanup(effective,path.parent,root);evidence.update(proofs)
        identities=[(record[k],record.get(stamp),k) for k,stamp in (("pid","process_created_at"),("child_pid","child_created_at"),("watchdog_pid","watchdog_created_at")) if record.get(k) is not None]
        identities+=extra;require(identities,"Campaign receipt has no inspectable host process identity")
        decisions=[]
        for pid,created,role in identities:
            decisions.append({"role":role,"pid":pid,"recorded_created_at":created,"state":inactive_identity(pid,created,finished,table)})
            finish_bounds[pid]=max(finished,finish_bounds.get(pid,finished))
        identity_count+=len(decisions)
        rows.append({"receipt":path.relative_to(root).as_posix(),"original_status":record.get("status"),"effective_status":effective["status"],
            "history_basis":basis,"cleanup_basis":cleanup,"scope":"cloud_host_processes_only_remote_allocation_not_inferred" if remote else "local_campaign_processes",
            "identities":decisions})
    allowed={pid:created for pid,created in allowed_processes}
    root_text=str(root).replace("\\","/").casefold()
    for pid,row in table.items():
        command=row.get("command");name=(row.get("name") or "").casefold()
        if not isinstance(command,str) or not name.startswith(("python","py.exe")):continue
        normalized=command.replace("\\","/").casefold()
        managed=root_text in normalized and re.search(r"(?:^|[/\s\"])(?:train[^/\s\"]*|benchmark[^/\s\"]*|predict[^/\s\"]*|run_[^/\s\"]*|preflight_[^/\s\"]*|profile_[^/\s\"]*|process_tree_guard|watchdog)\.py(?:[\s\"]|$)",normalized)
        if managed:
            require(pid in allowed and type(row.get("created_at")) in (int,float) and abs(row["created_at"]-allowed[pid])<.01,
                "An unclosed/unrecorded local campaign command is active")
    stale=list((root/"results/research").glob("local_lease_stale_*.json"))
    for path in stale:
        record=read(path)
        inactive_identity(record.get("pid"),record.get("process_created_at"),now(),table);evidence[path.relative_to(root).as_posix()]=sha(path)
    states,state_sources=state_history(root,table,finish_bounds);evidence.update(state_sources)
    return {"schema":"campaign_local_process_inactivity_audit_v2","status":"complete","recorded_at":now().isoformat(),
        "local_campaign_workers_inactive":True,"global_gpu_idle":"unverified","remote_gpu_allocation_status":"not_inferred",
        "supervisor_records":len(rows),"recorded_identity_checks":identity_count+len(states),"records":rows,
        "structured_state_identities":states,"evidence_sha256":evidence}


def global_observation():
    result={"global_gpu_idle":"unverified","use_as_execution_gate":False}
    try:
        response=subprocess.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=10)
        result.update(query_returncode=response.returncode,nonempty_rows=len([line for line in response.stdout.splitlines() if line.strip()]),
            stdout_sha256=hashlib.sha256(response.stdout.encode()).hexdigest())
    except (OSError,subprocess.TimeoutExpired) as error:result["observation_error_type"]=type(error).__name__
    return result


def manifest_for(inputs,status):
    result=old.base_manifest(inputs,status);result.update(schema=SCHEMA,adapter=inputs["adapter"],
        execution_contract="Campaign-local workers inactive and exclusive CPU lease; global/remote GPU idle is not inferred.",global_gpu_idle="unverified")
    return result


def research_cells(projection):
    cells=old.research_cells(projection)
    cells[0]["source"]+=("\nfor preflight in campaign.get('engineering_preflights', []):\n"
        "    print('ENGINEERING PREFLIGHT ONLY; NOT MODEL ACCURACY:', json.dumps(preflight, sort_keys=True))\n"
        "if campaign.get('engineering_preflights'):\n"
        "    print('An alpha-zero hard-loss control timing is not positive-KD speed, accuracy, or formal admission.')\n").splitlines(keepends=True)
    return cells


def compose(primary,projection,executed=None,session=None):
    notebook=old.compose(primary,projection)
    additions=research_cells(projection) if executed is None else copy.deepcopy(executed)
    if executed is not None:
        old.validate_new_cells(additions)
        require([cell["source"] for cell in additions]==[cell["source"] for cell in research_cells(projection)],"Actual v2 executed sources differ from selected projection")
    notebook["cells"][-2:]=additions
    note=("The two appended v2 campaign cells are unexecuted draft cells; preparation starts no kernel." if executed is None else
        "Only the two appended v2 campaign cells executed in a separate fresh CPU session with their actual counts 1,2.")
    notebook["cells"][-3]["source"]=("## Two transparent execution sessions\n\n"+note+" The six original complete cell objects retain their original source, outputs, metadata and counts 1-6. "
        "This is not eight newly executed cells. Both new cells embed the audited public snapshot and run without the original cell namespace. "
        "The identical external snapshot is campaign_evidence.json beside this Notebook. Engineering preflights do not supply model accuracy.\n").splitlines(keepends=True)
    metadata=notebook["metadata"]["campaign_execution_sessions"]
    metadata.update(builder_version=2,new_session=session or {"status":"not_executed"})
    old.preserved_original_cells(primary,notebook);old.public_check(notebook);return notebook


def checked_pages(primary_pages,research_pages):
    require(type(primary_pages) is int and type(research_pages) is int and primary_pages>0 and research_pages in (2,3) and primary_pages+research_pages<=8,
        "Actual v2 primary/research PDF pages exceed the eight-page limit")
    return primary_pages+research_pages


def pdf_rows(projection):
    """Pure presentation projection: retain every measured row and preflight."""
    rows=[]
    for row in projection["rows"]:
        loss=row.get("clean_nll");loss="Pending" if loss is None else f"{loss:.4f}"
        gate="Passed" if row.get("gate_passed") is True else "Failed" if row.get("gate_passed") is False else "Pending"
        rows.append([f"{row['label']}; {row.get('epochs','?')} epochs; {row.get('precision','?')}; {row.get('parameters','?')} parameters",
            f"{old.percentage(row.get('clean_accuracy'))} / {loss} / {old.percentage(row.get('stress_accuracy'))}; e{row.get('checkpoint_epoch','?')} {row.get('checkpoint_variant','?')}",
            f"{gate}; {row.get('qualification_status','')}; {row.get('repeat_status','not_verified')}; {row.get('delivery_status','not_promoted')}"])
    preflights=[json.dumps(value,sort_keys=True,ensure_ascii=True) for value in projection.get("engineering_preflights",[])]
    return rows,preflights


def render_research_pdf(projection,target):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
    style=ParagraphStyle("campaign-v2",fontName="Helvetica",fontSize=7.2,leading=8.5,spaceAfter=4)
    title=ParagraphStyle("campaign-v2-title",parent=style,fontName="Helvetica-Bold",fontSize=17,leading=20,spaceAfter=9)
    story=[]
    def p(text,heading=False):story.append(Paragraph(text,title if heading else style))
    def table(rows,widths):
        value=Table([[Paragraph(escape(str(cell)),style) for cell in row] for row in rows],colWidths=[w*inch for w in widths],repeatRows=1,hAlign="LEFT")
        value.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e8f1f2")),("VALIGN",(0,0),(-1,-1),"TOP"),
            ("BOTTOMPADDING",(0,0),(-1,-1),2),("TOPPADDING",(0,0),(-1,-1),2),("LINEBELOW",(0,0),(-1,-1),.3,colors.HexColor("#cbd8df"))]))
        story.append(value);story.append(Spacer(1,5))
    measured,preflights=pdf_rows(projection)
    p("Campaign: all measured development results",True)
    p("The original report is the frozen primary study. Follow-on work reuses the official 5,000 development images; it does not reuse the one-time confirmation. "
        "Five synthetic stress views are not CIFAR-C or hidden-test accuracy. Gates retain each recorded protocol and separate original/incumbent/case criteria.")
    baseline=projection["baseline"]
    p(f"Fixed original flip baseline: clean {old.percentage(baseline.get('clean_accuracy'))}, NLL {baseline.get('clean_nll')}, stress {old.percentage(baseline.get('stress_accuracy'))}. "
        f"All {len(measured)} measured rows are listed; engineering-only preflights appear separately on the next page.")
    table([["Recipe / training", "Clean / NLL / stress", "Primary gate / repeat / delivery"],*measured],[2.15,2.15,2.65])
    p("Conditional intervals use original-image clusters with all five stress views together. Baseline-flip intervals and incumbent-weighted intervals remain separate "
        "in the complete snapshot and Notebook. They do not establish independent confirmation, seed variance or hidden-test gains.")
    story.append(PageBreak());p("Campaign: official scores and engineering limits",True)
    table([["Evidence","Public accuracy"],*[ [label,old.percentage(projection.get(key))] for label,key in (
        ("Original primary","original_primary_public"),("Best original","original_best_public"),("Best newly scored research","best_new_public"),("Best observed overall","best_observed_public"))]],[4.95,2.0])
    receipts=[[r["run"],r["submission_ref"],f"{old.percentage(r.get('public_score'))} / {old.percentage(r.get('private_score'))}"] for r in projection["official_submissions"]]
    table([["All scored research entries","Official reference","Public / private"],*(receipts or [["None","None","Pending / unavailable"]])],[3.25,1.45,2.25])
    p(f"All {len(receipts)} canonical scored receipts are listed. Public best and original primary remain distinct; private scores stay unknown unless an official receipt supplies them. "
        "A benchmark, CSV, intent, partial repeat or engineering preflight is not a scored entry.")
    if preflights:
        p("Engineering-only preflights: no candidate accuracy or formal qualification",True)
        p("An alpha-zero hard-loss control timing is not positive-KD speed or accuracy. Failed timing guards do not authorize training, pairing or submission.")
        for row in preflights:p(escape(row))
    for note in projection.get("limitations",[]):p(escape(str(note)))
    for note in projection.get("engineering",[]):p(escape(str(note)))
    for item in projection.get("references",[]):p(escape(json.dumps(item,sort_keys=True,ensure_ascii=True)))
    p("Only the two appended campaign cells execute in a new CPU session, retaining real counts 1,2. Original six complete cell objects and counts 1-6 remain preserved. "
        "Global GPU idleness is unverified; the admission contract covers local campaign worker inactivity and an exclusive CPU lease.")
    p("Snapshot "+old.value_sha(projection)+"; recorded "+escape(str(projection["recorded_at"])))
    SimpleDocTemplate(str(target),pagesize=letter,leftMargin=.55*inch,rightMargin=.55*inch,topMargin=.55*inch,bottomMargin=.55*inch,
        title="Audited campaign appendix",author="Kevin Yang").build(story)
    from pypdf import PdfReader
    pages=len(PdfReader(str(target)).pages);require(pages in (2,3),"Actual v2 research appendix must occupy two or three pages");return pages


def combine_pdf(primary_pdf,research_pdf,target,expected_primary_pages):
    from pypdf import PdfReader,PdfWriter
    from reportlab.pdfgen import canvas
    primary,research=PdfReader(str(primary_pdf)),PdfReader(str(research_pdf))
    require(len(primary.pages)==expected_primary_pages,"Actual frozen primary page count differs")
    count=checked_pages(len(primary.pages),len(research.pages));writer=PdfWriter()
    for index,page in enumerate([*primary.pages,*research.pages],1):
        buffer=io.BytesIO();layer=canvas.Canvas(buffer,pagesize=(float(page.mediabox.width),float(page.mediabox.height)))
        layer.setFillColorRGB(1,1,1);layer.rect(0,0,float(page.mediabox.width),32,fill=1,stroke=0)
        layer.setFillColorRGB(.1,.2,.27);layer.setFont("Helvetica",7)
        layer.drawString(40,19,"Saber | Kevin Yang (ky2738) | frozen original + audited campaign")
        layer.drawRightString(float(page.mediabox.width)-40,19,f"{index} / {count}");layer.save();buffer.seek(0)
        page.merge_page(PdfReader(buffer).pages[0]);writer.add_page(page)
    with Path(target).open("wb") as stream:writer.write(stream)
    require(len(PdfReader(str(target)).pages)==count,"Actual combined PDF page count differs");return count


def prepare(manifest_path,output_dir,adapter,root=ROOT):
    inputs=load_inputs(manifest_path,adapter,root);target=old.output_directory(output_dir,root);target.mkdir(parents=True)
    old.write_snapshot(target/old.SNAPSHOT_NAME,inputs["projection"])
    write(target/"Project1_Campaign_Draft.ipynb",compose(inputs["primary"],inputs["projection"]))
    result=manifest_for(inputs,"prepared");result.update(kernel_started=False,pdf_rendered=False,campaign_inactivity_checked=False,
        artifacts_sha256={name:sha(target/name) for name in (old.SNAPSHOT_NAME,"Project1_Campaign_Draft.ipynb")})
    write(target/"report_manifest.json",result);return result


def claim_lease(root,nonce):
    import psutil
    path=Path(root)/"results/research/local_lease.json";path.parent.mkdir(parents=True,exist_ok=True)
    owner={"schema":LEASE,"purpose":"exclusive CPU report; no GPU-training accounting","nonce":nonce,"pid":os.getpid(),
        "process_created_at":psutil.Process().create_time(),"started_at":now().isoformat()}
    with path.open("x",encoding="utf-8") as stream:json.dump(owner,stream);stream.flush();os.fsync(stream.fileno())
    return path,owner


def owned_boundary(folder,request_sha):
    import psutil
    from research.process_tree_guard import _valid_child_record,matching_process
    from research.profile_augmix import verify_os_boundary
    request_path=folder/"execution_request.json";require(sha(request_path)==request_sha,"Unchanged v2 CPU request required")
    request=read(request_path);process=psutil.Process();owner=request.get("lease_owner",{})
    require(request.get("schema")==REQUEST and owner.get("schema")==LEASE and read(ROOT/"results/research/local_lease.json")==owner and
        matching_process(owner.get("pid"),owner.get("process_created_at")) is not None and process.ppid()==owner.get("pid") and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1",
        "Exact live v2 CPU owner/lease/environment required")
    started=_stamp(owner["started_at"])
    require(type(request.get("deadline_unix")) in (int,float) and math.isfinite(request["deadline_unix"]) and
        time.time()<request["deadline_unix"]<=min(started.timestamp()+240,old.report_deadline().timestamp()),"V2 CPU closeout deadline exceeded")
    directories=list(folder.glob("process_tree_*"));require(len(directories)==1,"One own v2 native boundary required")
    directory=directories[0];spec,ready=read(directory/"spec.json"),read(directory/"ready.json");nonce=directory.name.removeprefix("process_tree_")
    require(re.fullmatch(r"[0-9a-f]{32}",nonce) and spec.get("schema")==ready.get("schema")=="owned_python_process_tree_v1" and
        spec.get("nonce")==ready.get("nonce")==nonce and spec.get("host_pid")==owner["pid"] and
        abs(spec.get("host_created_at",0)-owner["process_created_at"])<.01 and spec.get("cwd")==str(ROOT) and
        spec.get("script")==str(Path(__file__).resolve()) and spec.get("script_sha256")==ready.get("script_sha256")==sha(__file__) and
        spec.get("arguments")==["--_kernel",str(request_path),"--request-sha",request_sha] and
        ready.get("method")==("windows_job_v1" if os.name=="nt" else "posix_session_v1") and
        spec.get("job_name")==("Local\\SaberAugMix_"+nonce if os.name=="nt" else None) and
        spec.get("guard_source_sha256")==old.FROZEN["research/process_tree_guard.py"] and
        type(spec.get("deadline_unix")) in (int,float) and time.time()<spec["deadline_unix"]<=request["deadline_unix"] and
        _valid_child_record(ready,spec,process.pid,process.create_time()),"Own v2 script/argv/nonce/child/native source binding differs")
    command=process.cmdline();guard=ROOT/"research/process_tree_guard.py"
    require(len(command)==4 and Path(command[0]).resolve()==Path(sys.executable).resolve() and Path(command[1]).resolve()==guard and
        command[2]=="--bootstrap" and Path(command[3]).resolve()==directory/"spec.json","Actual v2 guarded bootstrap required")
    verify_os_boundary(spec,ready,process)
    allowed=[(owner["pid"],owner["process_created_at"]),(process.pid,process.create_time())]
    if ready.get("watchdog_pid") is not None:allowed.append((ready["watchdog_pid"],ready["watchdog_created_at"]))
    return request,allowed


def worker(request_path,request_sha):
    folder=Path(request_path).resolve().parent
    require(Path(request_path).name=="execution_request.json" and folder.is_relative_to(ROOT/"outputs"),"Controlled v2 report output required")
    request,allowed=owned_boundary(folder,request_sha)
    inputs=load_inputs(request["campaign_manifest_path"],request["adapter"])
    require(inputs["sources"]==request["sources_sha256"] and inputs["campaign_manifest_sha256"]==request["campaign_manifest_sha256"] and
        inputs["snapshot_sha256"]==request["snapshot_sha256"] and read(folder/old.SNAPSHOT_NAME)==inputs["projection"],"V2 adapter/input/snapshot source closure changed")
    audit=campaign_inactive(owned_lease=request["lease_owner"],exclude_output=folder,allowed_processes=allowed)
    require(audit["evidence_sha256"]==request["campaign_history_sha256"],"Campaign history changed after CPU admission")
    write(folder/"campaign_inactivity_worker.json",audit)
    from nbclient import NotebookClient
    import nbformat
    from jupyter_client.kernelspec import KernelSpecManager
    require(KernelSpecManager().get_kernel_spec("python3").env.get("CUDA_VISIBLE_DEVICES") in (None,"-1"),"Kernel may not override CPU CUDA visibility")
    cells=research_cells(inputs["projection"])
    notebook=nbformat.v4.new_notebook(cells=[nbformat.from_dict(cell) for cell in old.kernel_cells(cells)])
    notebook.metadata["kernelspec"]={"name":"python3","language":"python","display_name":"Python 3"}
    started=now().isoformat()
    NotebookClient(notebook,timeout=120,kernel_name="python3",resources={"metadata":{"path":str(folder)}}).execute()
    actual=json.loads(nbformat.writes(notebook));old.validate_new_cells(actual["cells"])
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and time.time()<request["deadline_unix"] and
        source_snapshot(request["adapter"])==request["sources_sha256"],"V2 CPU/source/deadline changed")
    write(folder/"campaign_two_cells_executed.ipynb",actual)
    session={"schema":"campaign_two_cell_execution_session_v2","session_id":request["nonce"],"status":"complete","started_at":started,
        "finished_at":now().isoformat(),"fresh_kernel":True,"executed_code_cells":2,"execution_counts":[c["execution_count"] for c in actual["cells"]],
        "errors":0,"cuda_visible_devices":"-1","old_cells_executed_in_this_session":False,"new_cells":old.cell_hashes(actual["cells"]),
        "executed_notebook_sha256":sha(folder/"campaign_two_cells_executed.ipynb")}
    write(folder/"new_session.json",session)


def completed_boundary(folder,receipt):
    from research.process_tree_guard import _valid_child_record
    require(receipt.get("process_tree_cleanup_verified") is True and len(receipt.get("process_tree_guard_results",[]))==1,"One cleaned actual v2 native kernel proof required")
    proof=receipt["process_tree_guard_results"][0];directory=Path(proof["guard_directory"]).resolve();request_path=folder/"execution_request.json"
    require(re.fullmatch(r"[0-9a-f]{32}",proof.get("nonce","") or "") and directory.parent==folder and directory.name=="process_tree_"+proof["nonce"] and proof.get("cleanup_verified") is True and
        proof.get("timed_out") is False and proof.get("completion_authenticated") is True and type(proof.get("completion_exit_code")) is int and
        proof["completion_exit_code"]==0 and type(proof.get("child_returncode")) is int and proof["child_returncode"]==0 and
        read(directory/"host_result.json")==proof,"Actual v2 kernel cleanup/completion differs")
    require(type(proof.get("child_pid")) is int and proof["child_pid"]>0 and type(receipt.get("child_pid")) is int and
        receipt["child_pid"]==proof["child_pid"] and type(proof.get("child_created_at")) in (int,float) and
        math.isfinite(proof["child_created_at"]) and proof["child_created_at"]>0 and
        type(receipt.get("child_created_at")) in (int,float) and math.isfinite(receipt["child_created_at"]) and
        receipt["child_created_at"]==proof["child_created_at"],"V2 receipt and unique native child identities differ")
    spec=read(directory/"spec.json")
    require(spec.get("script")==str(Path(__file__).resolve()) and spec.get("script_sha256")==sha(__file__) and
        spec.get("arguments")==["--_kernel",str(request_path),"--request-sha",sha(request_path)] and spec.get("nonce")==proof["nonce"] and
        spec.get("host_pid")==receipt["pid"] and abs(spec.get("host_created_at",0)-receipt["process_created_at"])<.01 and
        spec.get("guard_source_sha256")==proof.get("guard_source_sha256")==old.FROZEN["research/process_tree_guard.py"],"V2 completed guard is not this script/owner/request")
    for name in ("ready.json","completion.json"):
        value=read(directory/name)
        require(_valid_child_record(value,spec,proof["child_pid"],proof["child_created_at"]) and
            value.get("method")==proof.get("method")==("windows_job_v1" if os.name=="nt" else "posix_session_v1"),"V2 completed child source/nonce/identity/method differs")
        if name=="completion.json":require(type(value.get("exit_code")) is int and value["exit_code"]==0,"V2 native exit must be zero")
    return {p.relative_to(folder).as_posix():sha(p) for p in (request_path,*(directory/name for name in ("spec.json","ready.json","completion.json","host_result.json")))}


def execute(manifest_path,output_dir,adapter,root=ROOT):
    root=Path(root).resolve();inputs=load_inputs(manifest_path,adapter,root);target=old.output_directory(output_dir,root)
    audit=campaign_inactive(root);observation=global_observation();deadline=min(time.time()+240,old.report_deadline(root).timestamp())
    require(deadline-time.time()>65,"Insufficient bounded CPU-report closeout time")
    target.mkdir(parents=True);nonce=uuid.uuid4().hex;lease,owner=claim_lease(root,nonce);began=time.perf_counter()
    receipt={"schema":SUPERVISOR,"status":"preparing","cpu_only":True,"pid":owner["pid"],"process_created_at":owner["process_created_at"],
        "started_at":owner["started_at"],"deadline_unix":deadline,"gpu_training_seconds_claimed":0,"sources_sha256":inputs["sources"]}
    try:
        checked=campaign_inactive(root,owned_lease=owner,exclude_output=target)
        require(checked["evidence_sha256"]==audit["evidence_sha256"],"Campaign history changed during atomic CPU admission")
        write(target/"campaign_inactivity_before.json",checked);write(target/"global_gpu_observation.json",observation)
        old.write_snapshot(target/old.SNAPSHOT_NAME,inputs["projection"])
        request={"schema":REQUEST,"nonce":nonce,"lease_owner":owner,"deadline_unix":deadline,"adapter":adapter,
            "campaign_manifest_path":inputs["campaign_manifest_path"],"campaign_manifest_sha256":inputs["campaign_manifest_sha256"],
            "sources_sha256":inputs["sources"],"snapshot_sha256":inputs["snapshot_sha256"],"campaign_history_sha256":checked["evidence_sha256"]}
        write(target/"execution_request.json",request);write(target/"supervisor.json",receipt)
        from research.process_tree_guard import run_guarded
        previous=os.environ.get("CUDA_VISIBLE_DEVICES");os.environ["CUDA_VISIBLE_DEVICES"]="-1"
        try:
            receipt["status"]="executing_two_cpu_cells"
            run_guarded([sys.executable,str(Path(__file__).resolve()),"--_kernel",str(target/"execution_request.json"),"--request-sha",sha(target/"execution_request.json")],
                target/"kernel.log",deadline-time.time()-60,target/"supervisor.json",receipt,cwd=root)
        finally:
            if previous is None:os.environ.pop("CUDA_VISIBLE_DEVICES",None)
            else:os.environ["CUDA_VISIBLE_DEVICES"]=previous
        proof=completed_boundary(target,receipt);actual=read(target/"campaign_two_cells_executed.ipynb");session=read(target/"new_session.json")
        old.validate_new_cells(actual["cells"])
        require(session.get("schema")=="campaign_two_cell_execution_session_v2" and session.get("status")=="complete" and
            session.get("session_id")==nonce and session.get("fresh_kernel") is True and session.get("old_cells_executed_in_this_session") is False and
            session.get("cuda_visible_devices")=="-1" and type(session.get("executed_code_cells")) is int and session["executed_code_cells"]==2 and
            session.get("execution_counts")==[1,2] and type(session.get("errors")) is int and session["errors"]==0 and
            session.get("new_cells")==old.cell_hashes(actual["cells"]) and session.get("executed_notebook_sha256")==sha(target/"campaign_two_cells_executed.ipynb"),"Actual v2 two-cell session evidence differs")
        notebook=compose(inputs["primary"],inputs["projection"],actual["cells"],session);write(target/old.NB_NAME,notebook)
        research_pages=render_research_pdf(inputs["projection"],target/"campaign_appendix.pdf")
        pages=combine_pdf(root/old.PRIMARY_DIR/"ky2738_Project1_Report.pdf",target/"campaign_appendix.pdf",target/"ky2738_Project1_Report.pdf",inputs["primary_manifest"]["report_pages"])
        images=old.render_pages(target/"ky2738_Project1_Report.pdf",target/"page_renderings",deadline)
        after=campaign_inactive(root,owned_lease=owner,exclude_output=target)
        require(after["evidence_sha256"]==checked["evidence_sha256"] and len(images)==pages and time.time()<=deadline and
            source_snapshot(adapter,root)==inputs["sources"] and sha(manifest_path)==inputs["campaign_manifest_sha256"] and
            old.preserved_original_cells(inputs["primary"],notebook),"V2 final source/history/pages/deadline audit differs")
        write(target/"campaign_inactivity_after.json",after)
        result=manifest_for(inputs,"complete");result.update(kernel_started=True,new_session=session,research_pdf_pages=research_pages,report_pages=pages,rendered_pdf_pages=len(images),
            campaign_inactivity_checked=True,campaign_local_workers_inactive=True,global_gpu_observation=observation,kernel_guard_evidence_sha256=proof,
            campaign_inactivity_audit_sha256={name:sha(target/name) for name in ("campaign_inactivity_before.json","campaign_inactivity_worker.json","campaign_inactivity_after.json")},
            artifacts_sha256={name:sha(target/name) for name in (old.NB_NAME,old.SNAPSHOT_NAME,"campaign_appendix.pdf","ky2738_Project1_Report.pdf","new_session.json")},
            page_renderings_sha256={path.name:sha(path) for path in images})
        write(target/"report_manifest.json",result);receipt.update(status="complete",artifact="report_manifest.json",artifact_sha256=sha(target/"report_manifest.json"));return result
    except BaseException as error:receipt.update(status="failed",error_type=type(error).__name__);raise
    finally:
        receipt.update(finished_at=now().isoformat(),elapsed_cpu_report_seconds=time.perf_counter()-began);write(target/"supervisor.json",receipt)
        old.release_cpu_lease(lease,owner,receipt)


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if argv and argv[0]=="--_kernel":
        require(len(argv)==4 and argv[2]=="--request-sha","Exact own v2 worker argv required");return worker(argv[1],argv[3])
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--campaign-manifest",type=Path,required=True);parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--adapter-source",type=Path);parser.add_argument("--adapter-sha256");parser.add_argument("--projection-schema")
    actions=parser.add_mutually_exclusive_group();actions.add_argument("--prepare",action="store_true");actions.add_argument("--execute",action="store_true")
    args=parser.parse_args(argv);adapter=descriptor(args.adapter_source,args.adapter_sha256,args.projection_schema)
    result=execute(args.campaign_manifest,args.output_dir,adapter) if args.execute else prepare(args.campaign_manifest,args.output_dir,adapter)
    print(json.dumps(result,indent=2));return result


if __name__=="__main__":main()
