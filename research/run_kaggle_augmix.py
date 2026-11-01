"""Explicit private AugMix staging/launch; offline preparation is independent.

Default makes no account calls. Dataset staging and kernel launch each require
an explicit action, immutable source-bound intent and shared cloud lease.
Account permissions and execution are established only by real receipts.
Only this new cloud controller uses kaggle_supervisor.json, outside the frozen
local supervisor.json budget namespace; remote child proof names are unchanged.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
SCHEMA = "research_private_kaggle_augmix_plan_v1"
STOP = datetime(2026, 11, 5, 5, tzinfo=timezone.utc)
MAX_SECONDS, MAX_GPU_SECONDS = 10800, 25 * 3600
# Kaggle's NvidiaTeslaT4 option can expose two T4s. The recipe uses cuda:0
# only, but reserves both physical GPUs conservatively until actual accounting
# is demonstrated. This is distinct from the server's wall-time timeout.
GPU_RESERVATION_MULTIPLIER = 2
TRAIN_EXPORT = "kevin1220/saber-official-data-export-20261030"
RAW_CONFIG_SHA = "c7ba91cb25ff128dfe816b303cde43cd4678673e0847a9f5b5bca7c4c4541df8"
TRAINING_MANIFEST_SHA = "88c25b1069b4d14936d021c4466adace6acc76d6c98009df4b6012631e2cb68a"
BASELINE_SHA = "502979bc03301aa2f0273a7414f86799302b908a3abada622a88e4578c4b734f"
CONFIG = "research/configs/wrn22_augmix_jsd.json"
BASELINE_FILES = ("best.pt", "metrics.json", "history.json")
BLOCKERS = ("Explicit private-baseline staging receipt and audited initial GPU ledger are required before launch.",)
GUARD_FILES = {'spec.json','ready.json','watchdog_ready.json','completion.json','host_result.json'}


def require(value, message):
    if not value: raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def exclusive(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def safe_zip(archive, expected, maximum):
    names = archive.namelist()
    require(len(set(names)) == len(names) and set(names) == set(expected) and
        sum(item.file_size for item in archive.infolist()) <= maximum and all(
        not n.endswith("/") and not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts and
        "\\" not in n and ":" not in n and (archive.getinfo(n).external_attr >> 16) & 0o170000 != 0o120000
        for n in names), "Archive paths, exact members or size differ")


def source_names():
    from research.run_colab_augmix import SOURCES
    return (*SOURCES, "prepare_data.py", "research/process_tree_guard.py", CONFIG)


def object_refs(run,nonce):
    require(isinstance(nonce,str) and re.fullmatch(r'[0-9a-f]{12}',nonce) is not None,"A fresh 48-bit object nonce is required")
    stem=run.replace('_','-')[:18].rstrip('-')
    return ('kevin1220/saber-augmix-'+stem+'-'+nonce,
            'kevin1220/saber-augmix-base-'+stem+'-'+nonce)


def validate_reuse_archive(root,archive):
    """Only the fresh, fully reconciled archive of our failed v1 is eligible."""
    from research.kaggle_research_transport import REUSE_REF,REUSE_ID,REUSE_SOURCE_SHA,REUSE_TITLE,canonical_hash
    root,archive=Path(root).resolve(),Path(archive).resolve()
    require(archive.is_relative_to(root/'results/research'),"Parent archive must remain private under ignored research results")
    manifest_path=archive/'complete_reconciled_hash_manifest.json'; manifest=read(manifest_path)
    require(manifest.get('schema')=='private_parent_version_complete_archive_v1' and manifest.get('readonly') is True and
        manifest.get('account_mutations')==0 and isinstance(manifest.get('files_sha256'),dict),"Complete readonly parent archive manifest is required")
    files={}
    for name,h in manifest['files_sha256'].items():
        relative=PurePosixPath(name); path=archive/name
        require(not relative.is_absolute() and '..' not in relative.parts and '\\' not in name and ':' not in name and
            path.is_file() and not path.is_symlink() and digest(path)==h,"Original parent archive bytes or paths changed")
        files[str(path.resolve())]=h
    files[str(manifest_path)]=digest(manifest_path)
    record=read(archive/'parent_verification_reconciled.json'); assessment=read(archive/'local_complete_archive_assessment.json')
    require(record.get('schema')=='research_owned_failed_kernel_parent_reconciliation_v1' and record.get('status')=='verified_readonly_reconciled' and
        record.get('readonly') is True and record.get('account_mutations')==0 and record.get('training_started') is False and
        record.get('confirmation_evaluated') is False and record.get('hidden_predictions_generated') is False and record.get('pinned_v1_status')=='ERROR',
        "Only the archived pre-training failed v1 can be reused")
    require(record.get('preserved_original_probe_report_sha256')==digest(archive/'parent_verification.json') and
        record.get('preserved_initial_archive_manifest_sha256')==digest(archive/'archive_hash_manifest.json') and
        record.get('local_assessment_sha256')==digest(archive/'local_complete_archive_assessment.json') and
        assessment.get('all_original_known_bytes_unchanged') is True and assessment.get('all_extra_source_bytes_explained') is True and
        assessment.get('old_cache_was_partial') is True,"Partial historical cache reconciliation must remain explicit and preserved")
    metadata={}
    for label in ('current','pinned_v1'):
        value=read(archive/(label+'.metadata.json')); metadata[label]=canonical_hash(value)
        require(value.get('id')==REUSE_ID and value.get('ref')==REUSE_REF and value.get('currentVersionNumber')==1 and
            value.get('isPrivate') is True and value.get('enableInternet') is False and value.get('title')==REUSE_TITLE and
            digest(archive/(label+'.source.py'))==REUSE_SOURCE_SHA,"Archived parent positive identity/source differs")
    expected={REUSE_REF.split('/')[1]+'.log','repeat-seed2027/launch_manifest.json','repeat-seed2027/repeat_status.json',
        *('saber/'+n for n in ('train.py','model.py','data.py','runtime.py','prepare_data.py','predict.py','configs/wrn_final.json','configs/repeat_seed2027.json','__pycache__/runtime.cpython-313.pyc'))}
    outputs=record.get('fresh_complete_v1_output_sha256',{})
    require(set(outputs)==expected and record.get('fresh_complete_v1_output_count')==12 and
        all(digest(archive/'fresh_pinned_v1_outputs'/n)==h for n,h in outputs.items()),"The complete fresh v1 output set is not the old partial cache")
    quota=record.get('raw_gpu_quota',{})
    require(quota.get('time_reserved')==0 and quota.get('pay_to_scale_enabled') is False,"Parent archive did not prove inactive free quota")
    return {'schema':'explicit_owned_failed_kernel_reuse_v1','ref':REUSE_REF,'kernel_id':REUSE_ID,'parent_version':1,'expected_next_version':2,
        'title':REUSE_TITLE,'source_sha256':REUSE_SOURCE_SHA,'current_metadata_sha256':metadata['current'],'pinned_metadata_sha256':metadata['pinned_v1'],
        'v1_outputs_sha256':outputs,'archive_path':str(archive),'archive_manifest_sha256':digest(manifest_path),'archive_files_sha256':files,
        'partial_cache_reconciliation_preserved':True,'no_atomic_version_precondition':True}


def committed_bundle(root, commit):
    """Only an explicit, committed own-source closure; never data or weights."""
    from research.run_colab_augmix import CORE, validate_config
    resolved = subprocess.check_output(["git", "rev-parse", "--verify", "--end-of-options", commit+"^{commit}"], cwd=root, text=True).strip()
    require(re.fullmatch(r'[0-9a-f]{40}',resolved) is not None,"An exact committed source revision is required")
    blobs = {name: subprocess.check_output(["git", "show", resolved+":"+name], cwd=root) for name in source_names()}
    for name, blob in blobs.items():
        require(hashlib.sha256(blob).hexdigest() == digest(Path(root)/name), "Working source differs from committed package")
    require(all(hashlib.sha256(blobs[n]).hexdigest() == h for n,h in CORE.items()), "Frozen four-file core changed")
    require(hashlib.sha256(blobs[CONFIG]).hexdigest() == RAW_CONFIG_SHA, "Registered primary raw configuration changed")
    config = validate_config(json.loads(blobs[CONFIG]))
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, blob in blobs.items():
            entry = zipfile.ZipInfo(name, date_time=(2026, 10, 31, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, blob)
    return resolved, payload.getvalue(), {n:hashlib.sha256(b).hexdigest() for n,b in blobs.items()}, config


def validate_quota(raw, reserve_seconds):
    """Inputs are unrounded typed timedelta.total_seconds(), never SDK.to_dict()."""
    require(set(raw) == {"remaining_seconds", "reserved_seconds"}, "Exact raw GPU quota fields are required")
    require(all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in raw.values()), "Raw GPU quota is missing or nonfinite")
    require(raw["reserved_seconds"] == 0 and raw["remaining_seconds"] >= reserve_seconds,
        "Another GPU reservation exists or insufficient actual GPU quota")


def raw_gpu_quota(response):
    """Normalize live *typed* SDK values without its lossy timedelta serializer.

    Official SDK 0.1.38 TimeDeltaSerializer serializes delta.seconds, losing
    complete days. The CLI also rounds hours and omits time_reserved. Neither
    representation is an acceptable allocation/cleanup authority.
    """
    gpu = getattr(response,"gpu_quota",None)
    values = {}
    for field in ("time_used","time_reserved","total_time_allowed","minimum_time_allowed"):
        value = getattr(gpu,field,None)
        require(isinstance(value,timedelta),"Raw typed GPU timedelta field is missing")
        seconds = value.total_seconds()
        require(math.isfinite(seconds) and seconds >= 0,"Invalid raw GPU quota duration")
        values[field] = seconds
    require(getattr(gpu,"is_pay_to_scale_enabled",None) is False,
        "Pay-to-scale state must be explicitly disabled")
    require(values['time_used']<=values['total_time_allowed'],"GPU quota totals are inconsistent")
    return {"remaining_seconds":values['total_time_allowed']-values['time_used'],
            "reserved_seconds":values['time_reserved']}


def validate_ledger(document, reserve_seconds):
    require(document.get("schema") == "research_kaggle_gpu_ledger_v1" and document.get("limit_seconds") == MAX_GPU_SECONDS and
        document.get("initial_history_audited") is True and isinstance(document.get("entries"),list),
        "An audited cumulative 25-GPU-hour ledger is required; weekly resets do not erase spending")
    opening=document.get('opening_charge_seconds',0)
    require(type(opening) is int and 0<=opening<=MAX_GPU_SECONDS,"Invalid conservative opening charge")
    ids, charged = set(), opening
    for row in document["entries"]:
        value = row.get("reserved_seconds")
        require(isinstance(row.get("run"),str) and row["run"] not in ids and type(value) is int and 0 < value <= MAX_SECONDS*GPU_RESERVATION_MULTIPLIER and
            row.get("status") in ("reserved", "push_unknown", "running", "terminal"), "Invalid or duplicated GPU ledger entry")
        ids.add(row["run"]); charged += value
    require(charged+reserve_seconds <= MAX_GPU_SECONDS, "Cumulative 25-GPU-hour cap exhausted")
    return charged


def reserve_ledger(document, run, seconds, intent_sha256):
    """Pure ledger transition; the eventual live adapter must persist it first."""
    validate_ledger(document,seconds)
    require(all(row['run']!=run for row in document['entries']),"A reservation/unknown push cannot be retried")
    require(re.fullmatch(r'[0-9a-f]{64}',intent_sha256) is not None,"Durable intent fingerprint is required")
    result=json.loads(json.dumps(document))
    result['entries'].append({'run':run,'reserved_seconds':seconds,'status':'reserved','intent_sha256':intent_sha256})
    return result


def validate_terminal(intent, observed, raw_quota, first_observed_at, checked_at):
    """A latest-version response or cancellation request cannot release a lease."""
    require(type(intent.get('version')) is int and intent['version']>0 and
        observed.get('ref')==intent.get('ref') and observed.get('version')==intent['version'] and
        observed.get('status') in ('COMPLETE','ERROR','CANCEL_ACKNOWLEDGED'),
        "Only the exact accepted version's terminal state is cleanup evidence")
    require(type(first_observed_at) in (int,float) and type(checked_at) in (int,float) and
        math.isfinite(first_observed_at) and math.isfinite(checked_at) and checked_at-first_observed_at>=180,
        "A separate delayed zero-reservation observation is required")
    validate_quota(raw_quota,0)
    return True


def admission(plan, raw_quota, ledger, lease_path, now=None):
    now = now or datetime.now(timezone.utc)
    require(now.tzinfo is not None and now < STOP, "Campaign absolute hard stop reached")
    seconds = min(plan["max_seconds"], int((STOP-now).total_seconds())-180)
    require(300 <= seconds <= MAX_SECONDS, "Insufficient absolute deadline margin")
    require(not Path(lease_path).exists(), "Shared Colab/Kaggle cloud lease is occupied")
    reserve=seconds*GPU_RESERVATION_MULTIPLIER
    validate_quota(raw_quota,reserve)
    charged = validate_ledger(ledger,reserve)
    return {"server_session_timeout_seconds":seconds,"absolute_deadline_unix":now.timestamp()+seconds,
            "cumulative_reserved_before_seconds":charged,"reserve_seconds":reserve,
            "conservative_physical_gpu_multiplier":GPU_RESERVATION_MULTIPLIER}


def measured_budget(median_seconds, remaining_seconds):
    require(type(median_seconds) in (int,float) and math.isfinite(median_seconds) and median_seconds > 0,
        "Actual pipeline median must be finite and positive")
    needed = 1.5*median_seconds*313*160+300
    require(needed+180 <= remaining_seconds, "Measured full AugMix+JSD pipeline exceeds fixed budget; no formal training")
    return needed


REMOTE = r'''
"""Private official-training-only AugMix; bounded absolute UTC deadline."""
import base64,hashlib,io,json,math,os,pathlib,shutil,signal,subprocess,sys,threading,time,traceback,zipfile
P=json.loads(__PLAN__)
if P.get('remote_execution_authorized') is not True: raise RuntimeError('Offline-only workload template: explicit sealed launch required')
R=pathlib.Path('/kaggle/working/augmix'); S=pathlib.Path('/tmp/saber-augmix-source'); D=S/'data'
R.mkdir(exist_ok=False); S.mkdir(exist_ok=False); D.mkdir()
deadline=min(P['absolute_deadline_unix'],P['campaign_stop_unix'])
def dump(name,value): (R/name).write_text(json.dumps(value,indent=2))
def sha(path):
    with pathlib.Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def remaining():
    left=deadline-time.time()-180
    if left<=0: raise TimeoutError('Absolute AugMix deadline reached')
    return left
def hardstop():
    time.sleep(max(0,deadline-time.time()))
    # Every workload child has an independent authenticated owner watchdog.
    # Ending this owner triggers that existing guard, never a bare reused PID.
    os._exit(124)
threading.Thread(target=hardstop,daemon=True).start()
def run(args,log):
    run_guarded([sys.executable,*args],R/log,min(10800,remaining()),R/'remote_receipt.json',receipt,cwd=S)
receipt={'schema':'research_kaggle_augmix_remote_v1','status':'failed','stage':'startup','started_at_unix':time.time(),
    'deadline_unix':deadline,
    'input_plan_sha256':P['input_plan_sha256'],'source_sha256':P['source_sha256'],'confirmation_evaluated':False,
    'hidden_images_loaded':False,'external_data_loaded':False,'stage_history':[]}
def stage(name):
    receipt['stage']=name; receipt['stage_history'].append({'stage':name,'at_unix':time.time()}); remaining()
try:
    remaining()
    payload=base64.b64decode(__BUNDLE__,validate=True)
    if hashlib.sha256(payload).hexdigest()!=P['source_bundle_sha256']: raise ValueError('Source bundle changed')
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        if set(z.namelist())!=set(P['source_sha256']) or len(z.namelist())!=len(P['source_sha256']): raise ValueError('Source allowlist differs')
        for name,h in P['source_sha256'].items():
            if name.startswith('/') or '..' in pathlib.PurePosixPath(name).parts or '\\' in name or ':' in name: raise ValueError('Unsafe source path')
            raw=z.read(name)
            if hashlib.sha256(raw).hexdigest()!=h: raise ValueError('Source changed')
            target=S/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(raw)
    sys.path.insert(0,str(S)); import psutil
    from research.process_tree_guard import run_guarded
    receipt.update(owner_pid=os.getpid(),owner_created_at=psutil.Process().create_time())
    import runtime,torch
    count=torch.cuda.device_count()
    if not torch.cuda.is_available() or count not in (1,2) or any('T4' not in torch.cuda.get_device_name(i) or tuple(torch.cuda.get_device_capability(i))!=(7,5) for i in range(count)): raise RuntimeError('Native-FP16 T4 allocation is required')
    torch.cuda.set_device(0)
    dump('hardware.json',{'gpu':torch.cuda.get_device_name(),'visible_gpu_count':count,'used_gpu_index':0,'data_parallel':False,'reserved_gpu_multiplier':2,'torch':torch.__version__,'capability':list(torch.cuda.get_device_capability()),'native_amp_dtype':'float16'})
    inputs=pathlib.Path('/kaggle/input')
    if any(list(inputs.rglob(n)) for n in ('official_test.npz','test.csv','sample_submission.csv')): raise ValueError('Hidden/test input attached')
    candidates=list(inputs.rglob('official_train.npz'))
    if len(candidates)!=1: raise ValueError('Exactly one private official TRAIN export is required')
    training=candidates[0].parent
    if sha(training/'official_train.npz')!=P['data_signature']['archive_sha256']: raise ValueError('Official training archive differs')
    for name in ('official_train.npz','official_source_manifest.json','train_labels.csv'): shutil.copyfile(training/name,D/name)
    stage('prepare_official_training')
    run(['prepare_data.py','--data-dir',str(D),'--seed','2026'],'prepare.log')
    manifest=json.loads((D/'training_manifest.json').read_text())
    if manifest['split']['indices_sha256']!=P['data_signature']['split_sha256'] or manifest['split']['train_indices_sha256']!=P['training_indices_sha256'] or (manifest['split']['train_count'],manifest['split']['validation_count'])!=(40000,5000): raise ValueError('Official training IDs/split differ')
    if manifest['files_sha256']!=P['cache_sha256']: raise ValueError('Official generated cache differs')
    base_manifest=list(inputs.rglob('saber_baseline_manifest.json'))
    if len(base_manifest)!=1: raise ValueError('Exactly one private baseline payload required')
    base=base_manifest[0].parent; bm=json.loads(base_manifest[0].read_text())
    if bm!=P['baseline_manifest']: raise ValueError('Private baseline metadata changed')
    baseline=S/'baseline'; baseline.mkdir()
    for name,h in bm['files_sha256'].items():
        if sha(base/name)!=h: raise ValueError('Private baseline file changed')
        shutil.copyfile(base/name,baseline/name)
    config='research/configs/wrn22_augmix_jsd.json'
    stage('preflight')
    run(['research/train_augmix.py','--preflight','--config',config,'--output-dir',str(R/'preflight'),'--data-dir',str(D),'--device','cuda'],'preflight.log')
    from research.train_augmix import validate_preflight
    pref=json.loads((R/'preflight/engineering_benchmark.json').read_text())
    validate_preflight(pref,P['config'],P['raw_config_sha256'],P['training_source_sha256'],require_t4=True)
    needed=1.5*pref['median_step_seconds']*313*160+300
    if not math.isfinite(needed) or needed+180>deadline-time.time(): raise TimeoutError('Predeclared measured pipeline budget does not fit')
    receipt['preflight']=pref; receipt['required_training_and_benchmark_seconds']=needed
    stage('baseline_benchmark')
    run(['research/benchmark.py','--checkpoint',str(baseline/'best.pt'),'--output',str(R/'baseline/benchmark.json'),'--data-dir',str(D),'--device','cuda'],'baseline.log')
    stage('scratch_training')
    run(['research/train_augmix.py','--config',config,'--output-dir',str(R/'candidate'),'--data-dir',str(D),'--device','cuda'],'training.log')
    stage('candidate_benchmark')
    run(['research/benchmark_augmix.py','--checkpoint',str(R/'candidate/best.pt'),'--output',str(R/'candidate/benchmark.json'),'--baseline',str(R/'baseline/benchmark.json'),'--data-dir',str(D),'--device','cuda'],'benchmark.log')
    metrics=json.loads((R/'candidate/metrics.json').read_text())
    if metrics.get('status')!='complete' or metrics.get('epochs_completed')!=160 or metrics.get('attempted_updates')!=50080 or metrics['successful_updates']+metrics['skipped_updates']!=50080: raise ValueError('Formal run incomplete')
    bench=json.loads((R/'candidate/benchmark.json').read_text())
    receipt.update(status='complete',benchmark_passed=bench['promotion']['center80_shift20']['passed'])
except BaseException as error:
    receipt.update(error_type=type(error).__name__); (R/'controller.log').write_text(traceback.format_exc())
finally:
    receipt['finished_at_unix']=time.time(); dump('remote_receipt.json',receipt)
    allow=set(P['output_allowlist'])
    for proof in receipt.get('process_tree_guard_results',[]):
        nonce=proof['nonce']
        if len(nonce)!=32 or any(c not in '0123456789abcdef' for c in nonce): raise ValueError('Unsafe guard nonce')
        allow|={'process_tree_'+nonce+'/'+n for n in P['guard_output_files']}
    actual={str(p.relative_to(R)).replace('\\','/') for p in R.rglob('*') if p.is_file()}
    if not actual<=allow: raise ValueError('Unapproved private output member')
    if receipt['status']=='complete' and not set(P['required_outputs'])<=actual: raise ValueError('Required private outputs missing')
    archive=pathlib.Path('/kaggle/working/augmix_artifacts.private.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name in sorted(actual): z.write(R/name,name)
    pathlib.Path('/kaggle/working/augmix_output_manifest.json').write_text(json.dumps({'schema':'research_kaggle_augmix_outputs_v1','archive_sha256':sha(archive),'members_sha256':{n:sha(R/n) for n in actual},'remote_status':receipt['status'],'plan_sha256':P['input_plan_sha256']},indent=2))
if receipt['status']!='complete': raise SystemExit(1)
'''


def remote_program(plan, payload, deadline_unix):
    require(type(deadline_unix) in (int,float) and math.isfinite(deadline_unix) and deadline_unix <= STOP.timestamp(), "Remote deadline exceeds campaign hard stop")
    document = {**plan, "absolute_deadline_unix":deadline_unix,"campaign_stop_unix":STOP.timestamp()}
    code = REMOTE.replace("__PLAN__",repr(json.dumps(document))).replace("__BUNDLE__",repr(base64.b64encode(payload).decode()))
    compile(code,"augmix_entry.py","exec")
    return code


def prepare(root, output, run, source_commit="HEAD", max_seconds=MAX_SECONDS,reuse_owned_failed_kernel=None):
    from research.run_colab_augmix import SIGNATURE, CACHE_SHA, CORE, COMPLETE_OUTPUTS, POOL_OUTPUTS
    root, output = Path(root).resolve(), Path(output).resolve()
    require(re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,29}",run) is not None, "Run identifier must be 2..30 safe characters so all platform slugs/titles fit")
    require(type(max_seconds) is int and 300<=max_seconds<=MAX_SECONDS,"Server timeout must fit the three-hour cap")
    require(output.is_relative_to(root/'results/research'),"Private preparation must remain under ignored results/research")
    require(not output.exists(), "Offline preparation is immutable; choose a new directory")
    commit,payload,hashes,config=committed_bundle(root,source_commit)
    require(digest(root/'results/regularized/best.pt')==BASELINE_SHA,"Trusted baseline checkpoint changed")
    baseline={n:digest(root/'results/regularized'/n) for n in BASELINE_FILES}
    require(digest(root/'data/training_manifest.json')==TRAINING_MANIFEST_SHA,"Audited official training manifest fingerprint differs")
    training_manifest=read(root/'data/training_manifest.json')  # metadata only, never image/label arrays
    require(training_manifest['source_provenance']['official_train_npz_sha256']==SIGNATURE['archive_sha256'] and
        training_manifest['split']['indices_sha256']==SIGNATURE['split_sha256'] and training_manifest['files_sha256']==CACHE_SHA,
        "Previously audited official training metadata differs")
    nonce=uuid.uuid4().hex[:12]
    kernel_id,baseline_id=object_refs(run,nonce)
    reuse=validate_reuse_archive(root,reuse_owned_failed_kernel) if reuse_owned_failed_kernel is not None else None
    if reuse: kernel_id=reuse['ref']
    bm={'schema':'research_private_baseline_payload_v1','files_sha256':baseline,'official_images_included':False,'checkpoint_sha256':BASELINE_SHA}
    # New Kaggle controller logs live outside trainer directories; final audit
    # uses the unchanged typed validator and preserves every worker-pool proof.
    trainer_outputs={n for n in COMPLETE_OUTPUTS if n not in ('remote_receipt.json','preflight/preflight.log','baseline/benchmark.log','candidate/training.log','candidate/benchmark.log')}
    required=trainer_outputs|{'remote_receipt.json','hardware.json','prepare.log','preflight.log','baseline.log','training.log','benchmark.log'}
    plan={'schema':SCHEMA,'run':run,'workspace_root':str(root),'source_commit':commit,'source_sha256':hashes,'source_bundle_sha256':hashlib.sha256(payload).hexdigest(),
        'host_source_sha256':digest(Path(__file__)),'config':config,'raw_config_sha256':RAW_CONFIG_SHA,'data_signature':SIGNATURE,
        'cache_sha256':CACHE_SHA,'training_indices_sha256':training_manifest['split']['train_indices_sha256'],
        'training_source_sha256':{n:h for n,h in hashes.items() if n in CORE or n in ('research/train_augmix.py','research/augmix.py','research/train_shadow.py','research/augmentations.py')},
        'kernel_id':kernel_id,'baseline_dataset_id':baseline_id,'object_nonce':nonce,'dataset_creation_policy':'create_new_only_no_update',
        'kernel_mode':'reuse_owned_failed_v1' if reuse else 'new_unique','reuse_parent':reuse,
        'baseline_manifest':bm,'max_seconds':max_seconds,'server_session_timeout_seconds':max_seconds,
        'campaign_stop_unix':STOP.timestamp(),'conservative_gpu_reservation_multiplier':GPU_RESERVATION_MULTIPLIER,'train_export':TRAIN_EXPORT,'kernel_private':True,'internet_enabled':False,'confirmation_evaluated':False,
        'hidden_input_attached':False,'required_outputs':sorted(required),'output_allowlist':sorted(required|{'controller.log'}),'guard_output_files':sorted(GUARD_FILES),'launch_blockers':list(BLOCKERS),
        'launch_supported':True,'remote_execution_authorized':False,'dataset_staging_supported':True,'actual_gpu_run':False,'actual_account_calls':0,'prepared_at':datetime.now(timezone.utc).isoformat()}
    # Intent identity is computed without a circular self hash.
    plan['input_plan_sha256']=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    deadline=min(datetime.now(timezone.utc).timestamp()+max_seconds,STOP.timestamp())
    code=remote_program(plan,payload,deadline)
    output.mkdir(parents=True)
    private=output/'baseline_dataset'; private.mkdir()
    for name in BASELINE_FILES: shutil.copyfile(root/'results/regularized'/name,private/name)
    exclusive(private/'saber_baseline_manifest.json',bm)
    exclusive(private/'dataset-metadata.json',{'id':baseline_id,'title':baseline_id.split('/')[1],'licenses':[{'name':'other'}]})
    (output/'source.private.zip').write_bytes(payload)
    (output/'augmix_entry.py').write_text(code,encoding='utf-8')
    kernel_metadata={'id':kernel_id,'title':reuse['title'] if reuse else kernel_id.split('/')[1],'code_file':'augmix_entry.py','language':'python',
        'kernel_type':'script','is_private':True,'enable_gpu':True,'enable_internet':False,'machine_shape':'NvidiaTeslaT4',
        'competition_sources':[],'kernel_sources':[TRAIN_EXPORT],'dataset_sources':[baseline_id]}
    if reuse: kernel_metadata['id_no']=reuse['kernel_id']
    exclusive(output/'kernel-metadata.json',kernel_metadata)
    exclusive(output/'kaggle_plan.json',plan)
    exclusive(output/'prepared_files.json',{str(p.relative_to(output)).replace('\\','/'):digest(p) for p in output.rglob('*') if p.is_file()})
    return plan


def validate_prepared(folder):
    folder=Path(folder); fingerprints=read(folder/'prepared_files.json')
    require(all(digest(folder/n)==h for n,h in fingerprints.items()),"Prepared private source/baseline payload changed")
    actual={str(p.relative_to(folder)).replace('\\','/') for p in folder.rglob('*') if p.is_file()}
    preview=read(folder/'kaggle_plan.json')
    def known_output(name):
        relative=name.removeprefix('outputs/')
        return relative in set(preview['output_allowlist']) or re.fullmatch(r'process_tree_[0-9a-f]{32}/(?:spec|ready|watchdog_ready|completion|host_result)\.json',relative) is not None
    def known_reuse_check(name):
        pieces=name.split('/',2)
        if len(pieces)!=3 or pieces[0]!='reuse_checks' or not re.fullmatch(r'(?:prestage|stage|prelaunch|prepush|postpush)_[0-9a-f]{12}',pieces[1]): return False
        return pieces[2] in {'current.metadata.json','current.source.py','pinned_v1.metadata.json','pinned_v1.source.py','pinned_v1.status.json','quota.json','proof.json'} or (
            preview.get('reuse_parent') is not None and pieces[2].startswith('outputs/') and pieces[2][8:] in preview['reuse_parent']['v1_outputs_sha256'])
    operational={'baseline_stage_intent.json','baseline_stage_progress.json','baseline_receipt.json','baseline_stage_unknown.json',
        'baseline_transport_claim.json','kernel_transport_claim.json','launch_intent.json','kernel_push_progress.json','kaggle_supervisor.json','kernel_launch/augmix_entry.py','kernel_launch/kernel-metadata.json'}
    require(all(n in set(fingerprints)|{'prepared_files.json'}|operational or
        (n.startswith('download/') and Path(n).name in ('augmix_artifacts.private.zip','augmix_output_manifest.json')) or
        n=='download/'+preview['kernel_id'].split('/')[1]+'.log' or
        (n.startswith('outputs/') and known_output(n)) or known_reuse_check(n) for n in actual),"Prepared folder members changed")
    p=read(folder/'kaggle_plan.json'); require(p.get('schema')==SCHEMA and p.get('launch_supported') is True and
        p.get('remote_execution_authorized') is False and p.get('hidden_input_attached') is False,"Unverified Kaggle plan schema")
    new_kernel,new_dataset=object_refs(p['run'],p.get('object_nonce'))
    require(p['baseline_dataset_id']==new_dataset and p.get('dataset_creation_policy')=='create_new_only_no_update',"Fresh private baseline identity/create-only policy differs")
    if p.get('kernel_mode')=='reuse_owned_failed_v1':
        require(p.get('reuse_parent')==validate_reuse_archive(Path(p['workspace_root']),p['reuse_parent']['archive_path']) and
            p['kernel_id']==p['reuse_parent']['ref'],"Explicit failed parent archive changed")
    else: require(p.get('kernel_mode')=='new_unique' and p.get('reuse_parent') is None and p['kernel_id']==new_kernel,"New kernel identity policy differs")
    require(p.get('host_source_sha256')==digest(Path(__file__)),"Offline preparation/audit source changed")
    require(hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='input_plan_sha256'},sort_keys=True,separators=(',',':')).encode()).hexdigest()==p['input_plan_sha256'],"Prepared canonical plan identity differs")
    metadata=read(folder/'kernel-metadata.json')
    require(metadata.get('id')==p['kernel_id'] and metadata.get('is_private') is True and metadata.get('enable_internet') is False and
        metadata.get('title')==(p['reuse_parent']['title'] if p.get('reuse_parent') else p['kernel_id'].split('/')[1]) and 6<=len(metadata['title'])<=50 and
        metadata.get('competition_sources')==[] and metadata.get('kernel_sources')==[TRAIN_EXPORT] and
        metadata.get('dataset_sources')==[p['baseline_dataset_id']],"Only private TRAIN export and separate private baseline may be attached")
    require(metadata.get('id_no')==(p['reuse_parent']['kernel_id'] if p.get('reuse_parent') else None),"Kernel numeric locator differs from explicit reuse mode")
    dataset=read(folder/'baseline_dataset/dataset-metadata.json')
    require(dataset.get('id')==p['baseline_dataset_id'] and dataset.get('title')==p['baseline_dataset_id'].split('/')[1] and
        6<=len(dataset['title'])<=50 and dataset.get('licenses')==[{'name':'other'}],"Private baseline staging metadata changed")
    require(read(folder/'baseline_dataset/saber_baseline_manifest.json')==p['baseline_manifest'] and
        p['baseline_manifest']['files_sha256']['best.pt']==BASELINE_SHA and
        all(digest(folder/'baseline_dataset'/name)==h for name,h in p['baseline_manifest']['files_sha256'].items()),"Private original baseline differs")
    with zipfile.ZipFile(folder/'source.private.zip') as z:
        safe_zip(z,p['source_sha256'],2*1024*1024)
        require(all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in p['source_sha256'].items()),"Committed source bundle differs")
    require(digest(folder/'source.private.zip')==p['source_bundle_sha256'] and p['raw_config_sha256']==RAW_CONFIG_SHA,"Bundle/raw configuration changed")
    return p


def extract_outputs(prepared, archive_path, manifest_path, destination):
    """Validate exact private bytes before extracting to an unused directory.

    The eventual account adapter must additionally pin the downloaded kernel
    version. This offline function alone does not prove any remote job identity
    or release a cloud lease.
    """
    plan=validate_prepared(prepared); manifest=read(manifest_path); destination=Path(destination)
    require(not destination.exists(),"Downloaded outputs are immutable; choose a fresh directory")
    require(manifest.get('schema')=='research_kaggle_augmix_outputs_v1' and
        manifest.get('plan_sha256')==plan['input_plan_sha256'] and
        manifest.get('archive_sha256')==digest(archive_path) and
        isinstance(manifest.get('members_sha256'),dict),"Pinned output manifest or archive changed")
    names=set(manifest['members_sha256'])
    require('remote_receipt.json' in names,"Output receipt is missing")
    with zipfile.ZipFile(archive_path) as archive:
        safe_zip(archive,names,128*1024*1024)
        require(all(hashlib.sha256(archive.read(n)).hexdigest()==h for n,h in manifest['members_sha256'].items()),"Output member fingerprint changed")
        remote=json.loads(archive.read('remote_receipt.json'))
        require(remote.get('schema')=='research_kaggle_augmix_remote_v1' and
            remote.get('input_plan_sha256')==plan['input_plan_sha256'] and remote.get('source_sha256')==plan['source_sha256'] and
            remote.get('confirmation_evaluated') is False and remote.get('hidden_images_loaded') is False and
            remote.get('external_data_loaded') is False and remote.get('status')==manifest.get('remote_status'),"Remote typed identity or input flags differ")
        nonces={p.get('nonce') for p in remote.get('process_tree_guard_results',[])}
        require(len(nonces)<=5 and all(isinstance(n,str) and re.fullmatch(r'[0-9a-f]{32}',n) for n in nonces),"Invalid guarded-stage nonce set")
        guard_names={f'process_tree_{nonce}/{n}' for nonce in nonces for n in GUARD_FILES}
        require(names<=set(plan['output_allowlist'])|guard_names,"Output member allowlist differs")
        if remote['status']=='complete': require(set(plan['required_outputs'])<=names,"Complete job omitted required artifacts")
        destination.mkdir(parents=True)
        for name in names:
            path=destination/name; path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream: stream.write(archive.read(name))
    return plan,remote


def audit_outputs(prepared, destination, data_dir):
    """Deep CPU audit for future downloaded artifacts; never prediction or GPU.

    This explicit future action loads the verified official train/dev cache.
    Offline preparation and tests do not invoke it on actual training arrays.
    """
    import runtime
    import torch
    from research import benchmark as core, paired_compare as paired, delivery_v2 as delivery
    from research.benchmark_augmix import validate_augmix, validate_pool_document
    from research.train_augmix import validate_preflight
    prepared,destination=Path(prepared),Path(destination)
    plan=validate_prepared(prepared); candidate=destination/'candidate'
    require(all(digest(ROOT/n)==h for n,h in plan['source_sha256'].items()),"Current source differs from prepared workload")
    checkpoint=torch.load(candidate/'best.pt',map_location='cpu',weights_only=False)
    last=torch.load(candidate/'last.pt',map_location='cpu',weights_only=False)
    metrics,history,provenance,environment=[read(candidate/n) for n in ('metrics.json','history.json','provenance.json','environment.json')]
    remote=read(destination/'remote_receipt.json'); guards=remote.get('process_tree_guard_results',[])
    require(remote.get('status')=='complete' and remote.get('process_tree_cleanup_verified') is True and len(guards)==5 and
        [r.get('stage') for r in remote.get('stage_history',[])]==['prepare_official_training','preflight','baseline_benchmark','scratch_training','candidate_benchmark'],
        "Every complete stage needs actual authenticated descendant cleanup")
    guarded_scripts=['prepare_data.py','research/train_augmix.py','research/benchmark.py','research/train_augmix.py','research/benchmark_augmix.py']
    for index,proof in enumerate(guards):
        require(proof.get('method')=='posix_session_v1' and proof.get('cleanup_verified') is True and
            proof.get('completion_authenticated') is True and proof.get('completion_exit_code')==0 and
            proof.get('guard_source_sha256')==plan['source_sha256']['research/process_tree_guard.py'],"Unverified process-tree boundary")
        directory=destination/('process_tree_'+proof['nonce']); spec,ready,done,host,watch=[read(directory/n) for n in ('spec.json','ready.json','completion.json','host_result.json','watchdog_ready.json')]
        require(host==proof and spec['nonce']==ready['nonce']==done['nonce']==watch['nonce']==proof['nonce'] and
            spec['host_pid']==remote['owner_pid'] and spec['host_created_at']==remote['owner_created_at'] and
            spec['guard_source_sha256']==proof['guard_source_sha256'] and ready['pid']==done['pid']==proof['child_pid'] and
            abs(ready['created_at']-proof['child_created_at'])<.01 and done.get('exit_code')==0 and
            watch.get('pid')==ready.get('watchdog_pid') and watch.get('created_at')==ready.get('watchdog_created_at') and
            watch.get('leader_pid')==ready['pid'] and watch.get('leader_created_at')==ready['created_at'] and
            spec['deadline_unix']<=remote['deadline_unix']<=STOP.timestamp() and
            str(spec['script'])=='/tmp/saber-augmix-source/'+guarded_scripts[index] and
            done.get('script_sha256')==ready.get('script_sha256')==spec['script_sha256'] and
            spec['script_sha256']==plan['source_sha256'][str(spec['script']).split('/tmp/saber-augmix-source/',1)[1]],
            "Guard source, owner, nonce or authenticated completion differs")
        if index==1: require('--preflight' in spec['arguments'],"Engineering stage must use the actual official-image preflight")
        if index==3: require('--preflight' not in spec['arguments'] and '--resume' not in spec['arguments'],"Formal primary must start scratch independently")
    raw=(candidate/'input_config.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest()==RAW_CONFIG_SHA and read(candidate/'config.json')==plan['config']==checkpoint['config'],"Raw/effective primary configuration differs")
    integrity=validate_augmix(checkpoint,metrics,history,provenance,environment,raw,last)
    data=delivery.data_context(Path(data_dir)); signature,core_source,train_ids,dev_ids=data
    require(signature==plan['data_signature'] and environment.get('gpu')=='Tesla T4' and
        environment.get('amp_dtype')=='torch.float16' and environment.get('grad_scaler_enabled') is True,
        "Actual official data/native T4 environment differs")
    core.validate_checkpoint(checkpoint,train_ids,dev_ids,signature,metrics,core_source,history)
    evidence=read(candidate/'benchmark.json'); baseline=read(destination/'baseline/benchmark.json'); context=evidence['evaluation_context']
    def finite(value):
        if isinstance(value,float): return math.isfinite(value)
        if isinstance(value,dict): return all(finite(v) for v in value.values())
        if isinstance(value,list): return all(finite(v) for v in value)
        return True
    require(all(finite(v) for v in (metrics,history,evidence,baseline)),"Training or full benchmark contains nonfinite metrics")
    require(context.get('device_type')=='cuda' and context.get('amp_dtype')=='torch.float16',"Native CUDA FP16 benchmark required")
    paired.validate_document(evidence,checkpoint,{'checkpoint_sha256':digest(candidate/'best.pt'),
        'training_metrics_sha256':digest(candidate/'metrics.json'),'training_history_sha256':digest(candidate/'history.json')},
        signature,context,core_source,digest(ROOT/'research/benchmark.py'))
    base=prepared/'baseline_dataset'; basecp=torch.load(base/'best.pt',map_location='cpu',weights_only=False)
    paired.validate_document(baseline,basecp,{'checkpoint_sha256':BASELINE_SHA,'training_metrics_sha256':digest(base/'metrics.json'),
        'training_history_sha256':digest(base/'history.json')},signature,context,core_source,digest(ROOT/'research/benchmark.py'))
    core.validate_checkpoint(basecp,train_ids,dev_ids,signature,read(base/'metrics.json'),core_source,read(base/'history.json'))
    require(set(evidence['policies'])==set(baseline['policies'])==set(core.POLICIES) and
        evidence.get('baseline_sha256')==digest(destination/'baseline/benchmark.json') and evidence.get('fixed_primary_policy')=='center80_shift20',
        "Complete same-runtime baseline and fixed weighted policy required")
    recorded=evidence.get('research_training_integrity',{})
    require(all(recorded.get(k)==v for k,v in integrity.items()) and
        recorded.get('validator_source_sha256')==digest(ROOT/'research/benchmark_augmix.py') and
        recorded.get('core_benchmark_sha256')==digest(candidate/'benchmark.core.json') and
        recorded.get('provenance_document_sha256')==digest(candidate/'provenance.json') and
        recorded.get('environment_document_sha256')==digest(candidate/'environment.json') and
        read(candidate/'benchmark.core.json')=={k:v for k,v in evidence.items() if k not in ('research_training_integrity','fixed_primary_policy')},
        "Typed method integrity, core evidence or source fingerprints differ")
    pref=read(destination/'preflight/engineering_benchmark.json')
    validate_preflight(pref,plan['config'],RAW_CONFIG_SHA,plan['training_source_sha256'],require_t4=True)
    validate_pool_document(read(destination/'preflight/worker_pool_preflight.json'))
    require(pref['worker_pool_receipt_sha256']==digest(destination/'preflight/worker_pool_preflight.json'),"Preflight pool fingerprint differs")
    for epoch,row in enumerate(history,1):
        path=candidate/f'worker_pool_epoch{epoch:04d}.json'; validate_pool_document(read(path))
        require(row['worker_pool_receipt_sha256']==digest(path),"Formal epoch pool fingerprint differs")
    for policy,report in evidence['policies'].items():
        require(evidence['promotion'][policy]==core.promotion_gate(report,baseline['policies']['flip']),"Benchmark gate differs")
        require(all(case['examples']==5000 and case['input_pixel_sha256']==baseline['policies'][policy]['cases'][name]['input_pixel_sha256']
            for name,case in report['cases'].items()),"Full-case counts or same-runtime pixels differ")
    return {'status':'cpu_audit_complete','benchmark_passed':evidence['promotion']['center80_shift20']['passed'],
        'attempted_updates':metrics['attempted_updates'],'successful_updates':metrics['successful_updates'],
        'skipped_updates':metrics['skipped_updates'],'checkpoint_sha256':digest(candidate/'best.pt'),
        'benchmark_sha256':digest(candidate/'benchmark.json'),'cloud_cleanup_verified':False,'qualified_for_prediction':False}


class Client:
    """Official isolated interpreter; construction itself makes no API calls."""
    def __init__(self, python=None):
        candidate=Path(os.environ.get('APPDATA',''))/'uv/tools/kaggle/Scripts/python.exe'
        self.python=str(python or candidate)
    def call(self,action,payload=None):
        result=subprocess.run([self.python,str(ROOT/'research/kaggle_research_transport.py'),action],input=json.dumps(payload or {}),
            capture_output=True,encoding='utf-8',errors='replace',timeout=600 if action in ('stage','verify-baseline') else 180)
        try: response=json.loads(result.stdout)
        except ValueError: raise RuntimeError('Official operation outcome unknown; do not retry') from None
        if response.get('error_code')=='visible_identity_ambiguous':
            raise RuntimeError('visible_identity_ambiguous: complete own-kernel identity proof unavailable; no staging/push is permitted')
        if response.get('error_code')=='reuse_parent_drift':
            raise RuntimeError('reuse_parent_drift: the explicitly archived parent failed fresh positive verification; do not retry')
        require(result.returncode==0 and response.get('ok') is True,'Official operation failed/unknown; do not retry')
        return response['result']


def quota(client):
    raw=client.call('quota')
    require(raw.get('pay_to_scale_enabled') is False,"Pay-to-scale must be explicitly disabled")
    result={k:raw[k] for k in ('remaining_seconds','reserved_seconds')}
    validate_quota(result,0)
    client.last_raw_quota=raw
    return result


def initialize_ledger(root,audit_path,audit_sha256=None):
    """Only an explicit root-reviewed historical manifest can initialize 25h."""
    root,audit_path=Path(root),Path(audit_path).resolve(); audit=read(audit_path)
    require(audit_sha256 is not None and digest(audit_path)==audit_sha256 and
        audit.get('schema')=='research_kaggle_opening_budget_audit_v1' and audit.get('additional_limit_seconds')==MAX_GPU_SECONDS and
        audit.get('auditor')=='root_agent_under_existing_campaign_authorization' and
        audit.get('official_gpu_quota_reserved_seconds')==0 and audit.get('historical_account_exclusivity_claimed') is False and
        audit.get('unknown_old_account_jobs_reclaimed') is False,"An explicit hash-pinned root opening audit is required")
    used=audit.get('official_gpu_quota_time_used_seconds'); opening=audit.get('conservative_opening_charge_seconds')
    require(type(used) in (int,float) and math.isfinite(used) and used>=0 and type(opening) is int and opening==math.ceil(used),"Opening charge must conservatively include all observed account quota usage")
    require(digest(root/'research/campaign.json')==audit['campaign_sha256'] and
        digest(root/audit['quota_probe'])==audit['quota_probe_sha256'],"Audited campaign or actual raw quota probe changed")
    for name,item in audit.get('legacy_receipts',{}).items():
        require(digest(root/name)==item['sha256'],"Legacy GPU receipt changed since opening audit")
    for item in audit.get('research_receipt_scan',[]):
        require(digest(root/item['path'])==item['sha256'],"Research GPU receipt changed since opening audit")
    document={'schema':'research_kaggle_gpu_ledger_v1','limit_seconds':MAX_GPU_SECONDS,'initial_history_audited':True,
        'initial_audit_path':str(audit_path),'initial_audit_sha256':digest(audit_path),'opening_charge_seconds':opening,'entries':[]}
    validate_ledger(document,0); path=root/'results/research/kaggle_gpu_ledger.json'; path.parent.mkdir(parents=True,exist_ok=True)
    exclusive(path,document); return document


def bound_intent(root,folder,plan,kind,extra):
    import psutil
    root,folder=Path(root),Path(folder)
    value={'kind':kind,'nonce':uuid.uuid4().hex,'pid':os.getpid(),'process_created_at':psutil.Process().create_time(),
        'session':'kaggle:'+plan['kernel_id'],'backend':'kaggle','started_at':datetime.now(timezone.utc).isoformat()}
    lease=root/'results/research/cloud_lease.json'
    require(not lease.exists(),"Shared Colab/Kaggle cloud lease is occupied")
    exclusive(lease,value)
    files={str((folder/n).resolve()):h for n,h in read(folder/'prepared_files.json').items()}
    files[str((folder/'prepared_files.json').resolve())]=digest(folder/'prepared_files.json')
    files[str(Path(__file__).resolve())]=digest(__file__)
    transport=root/'research/kaggle_research_transport.py'; files[str(transport.resolve())]=digest(transport)
    if plan.get('reuse_parent'): files.update(plan['reuse_parent']['archive_files_sha256'])
    intent={'schema':'research_kaggle_durable_intent_v1',**value,'run':plan['run'],'kernel_ref':plan['kernel_id'],'lease_path':str(lease.resolve()),'files_sha256':files,
        'transport_sha256':digest(transport),'prepared_folder':str(folder.resolve()),'plan_sha256':plan['input_plan_sha256'],**extra}
    if plan.get('reuse_parent'): intent['reuse_parent']=plan['reuse_parent']
    return intent,lease


def release_matching(lease,intent):
    require(read(lease).get('nonce')==intent['nonce'],"Shared lease owner changed")
    Path(lease).unlink()


def verify_target(plan,folder,client,phase):
    if plan.get('reuse_parent'):
        path=Path(folder)/'reuse_checks'/(phase+'_'+uuid.uuid4().hex[:12])
        proof=client.call('verify-reuse-parent',{'contract':plan['reuse_parent'],'proof_folder':str(path)})
        require(proof.get('status')=='verified' and proof.get('kernel_id')==plan['reuse_parent']['kernel_id'] and
            proof.get('ref')==plan['kernel_id'] and proof.get('latest_version')==1 and proof.get('parent_complete_outputs_unchanged') is True and
            proof.get('parent_source_unchanged') is True and proof.get('parent_metadata_unchanged') is True,
            "Fresh positive failed-parent identity/source/output proof is required")
        return proof
    identity=client.call('kernel-inventory',{'ref':plan['kernel_id']})
    require(identity.get('inventory_complete') is True and identity.get('authenticated_owner')=='kevin1220' and identity.get('exists') is False,
        "A complete positively controlled own-kernel inventory is required")
    return identity


def stage_baseline(root,folder,client=None):
    root,folder=Path(root),Path(folder).resolve(); plan=validate_prepared(folder); intent_path=folder/'baseline_stage_intent.json'
    require(not intent_path.exists(),"Existing or unknown baseline staging intent forbids metadata retry")
    client=client or Client()
    require(not (root/'results/research/cloud_lease.json').exists(),"Shared cloud lease is occupied")
    identity=verify_target(plan,folder,client,'prestage')
    # An exact GET may return 403 for an uncreated private slug. Never infer
    # absence from that response. The official create/new endpoint is distinct
    # from version/update; use a fresh source-bound nonce and fail on collision.
    # Kaggle's official kagglehub handles conflict by a separate version call:
    # https://github.com/Kaggle/kagglehub/blob/main/src/kagglehub/datasets_helpers.py
    # This controller deliberately has no such fallback.
    payload={**plan['baseline_manifest']['files_sha256'],'saber_baseline_manifest.json':digest(folder/'baseline_dataset/saber_baseline_manifest.json')}
    intent,lease=bound_intent(root,folder,plan,'baseline_stage',{'dataset_ref':plan['baseline_dataset_id'],'payload_sha256':payload,
        'creation_policy':'create_new_only_no_update','object_nonce':plan['object_nonce'],
        'kernel_identity_proof':identity,
        'upload_folder':str((folder/'baseline_dataset').resolve()),'progress_path':str((folder/'baseline_stage_progress.json').resolve())})
    exclusive(intent_path,intent); args={'intent_path':str(intent_path),'intent_sha256':digest(intent_path)}
    receipt={'status':'stage_attempt_started','intent_sha256':args['intent_sha256'],'started_at':datetime.now(timezone.utc).isoformat()}
    try:
        accepted=client.call('stage',args)
        require(accepted.get('ref')==plan['baseline_dataset_id'] and accepted.get('version')==1,"Private baseline create identity differs")
        receipt['accepted']=accepted
        # Processing is read-only: retries here never repeat upload/create.
        began=time.monotonic(); verified=None
        while time.monotonic()-began<300:
            try: verified=client.call('verify-baseline',{**args,'version':1}); break
            except (ValueError,RuntimeError): time.sleep(15)
        require(verified is not None and verified.get('is_private') is True and verified.get('version')==1 and
            verified.get('ref')==plan['baseline_dataset_id'] and verified.get('files_sha256')==payload and
            verified.get('actual_download_verified') is True,"Actual private pinned baseline byte verification is required")
        receipt.update(status='complete',verified=verified,finished_at=datetime.now(timezone.utc).isoformat())
        exclusive(folder/'baseline_receipt.json',receipt)
        release_matching(lease,intent)
        return receipt
    except BaseException as error:
        receipt.update(status='stage_outcome_unknown',error_type=type(error).__name__,finished_at=datetime.now(timezone.utc).isoformat())
        exclusive(folder/'baseline_stage_unknown.json',receipt)
        raise


def launch(folder,root=ROOT,client=None,ledger_path=None,baseline_receipt=None,poll_seconds=20):
    folder,root=Path(folder).resolve(),Path(root).resolve(); plan=validate_prepared(folder)
    require(ledger_path is not None and baseline_receipt is not None,
        "Launch blocked before authentication, intent, lease or push: explicit audited ledger and private baseline receipt are required")
    ledger_path=Path(ledger_path); ledger=read(ledger_path); validate_ledger(ledger,0)
    require(digest(ledger['initial_audit_path'])==ledger['initial_audit_sha256'],"Initial audited GPU history changed")
    baseline=read(baseline_receipt); verified=baseline.get('verified',{})
    require(baseline.get('status')=='complete' and verified.get('ref')==plan['baseline_dataset_id'] and verified.get('is_private') is True and
        verified.get('version')==1 and verified.get('actual_download_verified') is True and
        verified.get('files_sha256',{}).get('best.pt')==BASELINE_SHA,"Verified private baseline staging receipt required")
    intent_path=folder/'launch_intent.json'; require(not intent_path.exists(),"A known/unknown kernel push intent forbids retries")
    require(not (root/'results/research/cloud_lease.json').exists(),"Shared Colab/Kaggle cloud lease is occupied")
    client=client or Client()
    stage_args={'intent_path':str(folder/'baseline_stage_intent.json'),'intent_sha256':digest(folder/'baseline_stage_intent.json'),'version':1}
    require(client.call('verify-baseline',stage_args)==verified,"Private baseline version or actual bytes changed before push")
    training=client.call('verify-train',{'ref':TRAIN_EXPORT})
    require(training.get('ref')==TRAIN_EXPORT and training.get('is_private') is True and type(training.get('version')) is int and training['version']>0,
        "The existing official training export must actually be private")
    train_pinned=TRAIN_EXPORT+'/'+str(training['version'])
    current=datetime.now(timezone.utc); budget=admission(plan,quota(client),ledger,root/'results/research/cloud_lease.json',current)
    quota_before_raw=client.last_raw_quota
    # A new handle is never silently updated. Existing versions, even terminal,
    # require another explicit run/plan rather than overwriting the same kernel.
    existing=verify_target(plan,folder,client,'prelaunch')
    sealed={**plan,'remote_execution_authorized':True}
    launch_folder=folder/'kernel_launch'; require(not launch_folder.exists(),"Sealed launch source is immutable")
    launch_folder.mkdir()
    (launch_folder/'augmix_entry.py').write_text(remote_program(sealed,(folder/'source.private.zip').read_bytes(),budget['absolute_deadline_unix']),encoding='utf-8')
    metadata=read(folder/'kernel-metadata.json'); metadata['dataset_sources']=[plan['baseline_dataset_id']+'/1']; metadata['kernel_sources']=[train_pinned]; exclusive(launch_folder/'kernel-metadata.json',metadata)
    intent,lease=bound_intent(root,folder,plan,'kernel_push',{'kernel_ref':plan['kernel_id'],'kernel_folder':str(launch_folder),
        'baseline_pinned_ref':plan['baseline_dataset_id']+'/1','train_export':train_pinned,'server_timeout_seconds':budget['server_session_timeout_seconds'],
        'absolute_deadline_unix':budget['absolute_deadline_unix'],'reserve_gpu_seconds':budget['reserve_seconds'],'ledger_path':str(ledger_path.resolve())})
    intent['progress_path']=str((folder/'kernel_push_progress.json').resolve())
    intent['parent_target_proof' if plan.get('reuse_parent') else 'kernel_absence_proof']=existing
    # Earlier readonly account checks did not own the cloud lease. Another
    # legitimate controller may have finalized its ledger in that interval.
    # Re-read under our exclusive lease before reserving or making an intent.
    try:
        ledger=read(ledger_path)
        require(digest(ledger['initial_audit_path'])==ledger['initial_audit_sha256'],"Initial GPU audit changed before reservation")
        validate_ledger(ledger,budget['reserve_seconds'])
        budget['cumulative_reserved_before_seconds']=validate_ledger(ledger,0)
    except BaseException:
        release_matching(lease,intent)  # No push intent or API mutation exists.
        raise
    intent['files_sha256'].update({str(p.resolve()):digest(p) for p in launch_folder.iterdir()})
    intent['files_sha256'][str(Path(baseline_receipt).resolve())]=digest(baseline_receipt)
    exclusive(intent_path,intent)
    updated=reserve_ledger(ledger,plan['run'],budget['reserve_seconds'],digest(intent_path))
    temporary=ledger_path.with_suffix('.writing'); exclusive(temporary,updated); temporary.replace(ledger_path)
    args={'intent_path':str(intent_path),'intent_sha256':digest(intent_path)}
    receipt={'schema':'research_kaggle_augmix_supervisor_v1','run':plan['run'],'status':'push_attempt_started','intent_sha256':args['intent_sha256'],
        'started_at':current.isoformat(),**budget,'runtime_stop_verified':False,'confirmation_evaluated':False,'submitted':False,
        'stop_or_cancel_requested':False,'backend':'kaggle','kernel_ref':plan['kernel_id']}
    if plan.get('reuse_parent'):
        receipt.update(expected_parent_version=1,expected_new_version=2,parent_archive_manifest_sha256=plan['reuse_parent']['archive_manifest_sha256'],
            no_atomic_version_precondition=True,explicit_existing_failed_container=True)
    receipt['quota_before_raw']=quota_before_raw
    receipt_path=folder/'kaggle_supervisor.json'
    def save():
        temporary=receipt_path.with_suffix('.writing'); temporary.write_text(json.dumps(receipt,indent=2)); temporary.replace(receipt_path)
    save()
    release_ready=False
    try:
        accepted=client.call('push',args)
        expected_version=2 if plan.get('reuse_parent') else 1
        require(accepted.get('ref')==plan['kernel_id'] and accepted.get('version')==expected_version and accepted.get('private_verified') is True and
            (not plan.get('reuse_parent') or (accepted.get('kernel_id')==plan['reuse_parent']['kernel_id'] and accepted.get('parent_v1_content_unchanged') is True)),
            "Actual accepted private kernel identity/version or preserved parent differs")
        receipt.update(status='running',accepted=accepted); save(); pinned=accepted['ref']+'/'+str(accepted['version'])
        terminal_at=None
        while time.time()<budget['absolute_deadline_unix']+180:
            observed=client.call('status',{'ref':pinned})
            require(observed.get('ref')==accepted['ref'] and observed.get('version')==accepted['version'],"Pinned kernel version changed")
            if observed.get('status') in ('COMPLETE','ERROR','CANCEL_ACKNOWLEDGED'):
                terminal_at=time.time(); receipt.update(terminal=observed,terminal_observed_unix=terminal_at); save(); break
            time.sleep(poll_seconds)
        require(terminal_at is not None,"Known version did not reach terminal state; retain lease")
        try:
            client.call('output',{'ref':pinned,'folder':str(folder/'download')})
            outputs=list((folder/'download').rglob('augmix_output_manifest.json')); archives=list((folder/'download').rglob('augmix_artifacts.private.zip'))
            require(len(outputs)==len(archives)==1,"Exactly one pinned output archive and manifest required")
            _,remote=extract_outputs(folder,archives[0],outputs[0],folder/'outputs')
            receipt.update(remote_status=remote['status'],output_archive_sha256=digest(archives[0]))
            if remote['status']=='complete': receipt['cpu_audit']=audit_outputs(folder,folder/'outputs',root/'data')
        except BaseException as error: receipt['output_error_type']=type(error).__name__
        # No cancellation is attempted: kernel_id is never a numeric session ID.
        # Keep the reservation until a separate delayed raw quota observation.
        while time.time()<terminal_at+180: time.sleep(min(20,terminal_at+180-time.time()))
        raw=quota(client); validate_terminal(accepted,receipt['terminal'],raw,terminal_at,time.time())
        receipt.update(status='complete' if receipt.get('cpu_audit',{}).get('status')=='cpu_audit_complete' else 'terminal_negative',
            runtime_stop_verified=True,cleanup_usage=raw,quota_after_raw=client.last_raw_quota,cleanup_verified_at=datetime.now(timezone.utc).isoformat())
        receipt['benchmark_passed']=receipt.get('cpu_audit',{}).get('benchmark_passed',False)
        release_ready=True
    except BaseException as error:
        receipt.update(status='cleanup_unverified' if 'terminal' in receipt else 'outcome_unknown',error_type=type(error).__name__)
        raise
    finally:
        receipt['finished_at']=datetime.now(timezone.utc).isoformat(); save()
        # Keep the full conservative reservation even for a terminal job.
        # Actual receipt evidence is appended, never a weekly-reset refund.
        accounted=read(ledger_path)
        for row in accounted['entries']:
            if row['run']==plan['run']:
                row.update(status='terminal' if receipt['runtime_stop_verified'] else 'push_unknown',
                    supervisor_path=str(receipt_path),supervisor_sha256=digest(receipt_path))
        temporary=ledger_path.with_suffix('.writing'); temporary.write_text(json.dumps(accounted,indent=2)); temporary.replace(ledger_path)
        if release_ready: release_matching(lease,intent)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare',action='store_true'); modes.add_argument('--validate',action='store_true'); modes.add_argument('--launch',action='store_true')
    modes.add_argument('--stage-baseline',action='store_true'); modes.add_argument('--initialize-ledger',action='store_true')
    parser.add_argument('--folder',type=Path); parser.add_argument('--run'); parser.add_argument('--source-commit',default='HEAD'); parser.add_argument('--max-seconds',type=int,default=MAX_SECONDS)
    parser.add_argument('--ledger',type=Path); parser.add_argument('--baseline-receipt',type=Path); parser.add_argument('--initial-audit',type=Path); parser.add_argument('--initial-audit-sha256'); parser.add_argument('--kaggle-python',type=Path)
    parser.add_argument('--reuse-owned-failed-kernel',type=Path,help='Explicit complete private archive of the one approved failed v1; prepare only')
    args=parser.parse_args()
    if args.reuse_owned_failed_kernel and not args.prepare: parser.error('--reuse-owned-failed-kernel is selected only during a new immutable --prepare')
    if not any((args.prepare,args.validate,args.launch,args.stage_baseline,args.initialize_ledger)):
        print(json.dumps({'status':'no_action','account_calls':0,'prerequisites':list(BLOCKERS)})); return
    if args.initialize_ledger:
        if not args.initial_audit or not args.initial_audit_sha256: parser.error('--initialize-ledger requires --initial-audit and --initial-audit-sha256')
        initialize_ledger(ROOT,args.initial_audit,args.initial_audit_sha256); print(json.dumps({'status':'audited_ledger_initialized','account_calls':0})); return
    if not args.folder: parser.error('An explicit private --folder is required')
    if args.prepare:
        if not args.run: parser.error('--prepare requires --run')
        result=prepare(ROOT,args.folder,args.run,args.source_commit,args.max_seconds,args.reuse_owned_failed_kernel)
        print(json.dumps({'status':'offline_prepared','run':result['run'],'account_calls':0,'prerequisites':result['launch_blockers']}))
    elif args.validate: print(json.dumps({'status':'offline_validated','run':validate_prepared(args.folder)['run'],'account_calls':0}))
    elif args.stage_baseline: print(json.dumps(stage_baseline(ROOT,args.folder,Client(args.kaggle_python))))
    else: print(json.dumps(launch(args.folder,client=Client(args.kaggle_python),ledger_path=args.ledger,baseline_receipt=args.baseline_receipt)))


if __name__=='__main__': main()
