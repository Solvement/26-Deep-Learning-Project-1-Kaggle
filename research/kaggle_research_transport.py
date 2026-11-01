"""Isolated official Kaggle transport. No Torch, retries of metadata, or cancel.

Only the owning supervisor invokes this script with a source-bound durable
intent. All SDK output is captured; credentials and signed URLs never return.
"""
from __future__ import annotations
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import time
import zipfile

REUSE_REF='kevin1220/saber-wrn22-repeat-seed2027-20261030'
REUSE_ID=141868079
REUSE_SOURCE_SHA='1cbccc63b39f2270e1aef4b868b25e6e64b371fabac723a2c58ce364c2569eb1'
REUSE_TITLE='Saber WRN22 repeat seed2027 20261030'
REUSE_OUTPUTS={REUSE_REF.split('/')[1]+'.log','repeat-seed2027/launch_manifest.json','repeat-seed2027/repeat_status.json',
    *('saber/'+n for n in ('train.py','model.py','data.py','runtime.py','prepare_data.py','predict.py','configs/wrn_final.json','configs/repeat_seed2027.json','__pycache__/runtime.cpython-313.pyc'))}


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def ref(value,version=False):
    pattern=r'kevin1220/[a-z0-9][a-z0-9-]{5,49}'+(r'/[1-9][0-9]*' if version else '')
    if not isinstance(value,str) or re.fullmatch(pattern,value) is None: raise ValueError('Unexpected owned Kaggle reference')
    return value.split('/')


def intent(payload,kind,require_lease=True):
    path=Path(payload['intent_path']); document=json.loads(path.read_text())
    if sha(path)!=payload['intent_sha256'] or document.get('kind')!=kind: raise ValueError('Durable intent differs')
    if require_lease:
        lease=json.loads(Path(document['lease_path']).read_text())
        if lease.get('nonce')!=document['nonce']: raise ValueError('Shared cloud owner differs')
    if any(sha(p)!=h for p,h in document['files_sha256'].items()): raise ValueError('Bound source/payload changed')
    if sha(__file__)!=document['transport_sha256']: raise ValueError('Transport source changed')
    return document


def dataset_view(api,value):
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
    owner,slug=ref(value); request=ApiGetDatasetRequest(); request.owner_slug=owner; request.dataset_slug=slug
    with api.build_kaggle_client() as client: result=client.datasets.dataset_api_client.get_dataset(request)
    return {'ref':result.ref,'is_private':result.is_private,'version':result.current_version_number}


def kernel_response(api,value,pinned=False):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    pieces=ref(value,pinned); request=ApiGetKernelRequest(); request.user_name=pieces[0]; request.kernel_slug=pieces[1]
    if pinned: request.version_label='v'+pieces[2]
    with api.build_kaggle_client() as client: result=client.kernels.kernels_api_client.get_kernel(request)
    return result


def kernel_view(api,value,pinned=False):
    result=kernel_response(api,value,pinned); metadata=result.metadata
    return {'ref':metadata.ref,'version':metadata.current_version_number,'is_private':metadata.is_private,
        'internet_enabled':metadata.enable_internet,'datasets':metadata.dataset_data_sources,'kernels':metadata.kernel_data_sources,
        'competitions':metadata.competition_data_sources,'source_sha256':hashlib.sha256(result.blob.source.encode('utf-8')).hexdigest()}


class KernelIdentityAmbiguous(ValueError):
    code='visible_identity_ambiguous'


def kernel_inventory(api,target,max_pages=5,slugifier=None):
    """Prove absence only in complete authenticated inventory with an anchor.

    SaveKernel is an upsert. An inaccessible exact GET (403 or 404) is never
    absence evidence here. Reference-free rows require successful positive
    identity resolution; their title or a random target cannot substitute.
    """
    owner,_=ref(target); anchor='kevin1220/saber-official-data-export-20261030'
    control=kernel_response(api,anchor+'/1',True).metadata
    if control.ref!=anchor or control.is_private is not True or control.current_version_number!=1:
        raise KernelIdentityAmbiguous('Known private owner/visibility control differs')
    rows=[]; pages=[]; seen=set(); token=None; complete=False
    for _ in range(max_pages):
        response=api.kernels_list_with_response(mine=True,user=None,page_size=100,page_token=token)
        if response is None or not isinstance(response.kernels,list) or any(row is None for row in response.kernels):
            raise KernelIdentityAmbiguous('Incomplete authenticated inventory response')
        rows.extend(response.kernels); nxt=response.next_page_token
        if not isinstance(nxt,str): raise KernelIdentityAmbiguous('Missing pagination token contract')
        pages.append({'rows':len(response.kernels),'has_next':bool(nxt)})
        if not nxt: complete=True; break
        if nxt in seen: raise KernelIdentityAmbiguous('Repeated authenticated inventory cursor')
        seen.add(nxt); token=nxt
    if not complete: raise KernelIdentityAmbiguous('Authenticated inventory pagination did not end')
    resolved=[]; positive=[]
    for row in rows:
        if row.ref:
            if not row.ref.startswith(owner+'/') or len(row.ref.split('/'))!=2: raise KernelIdentityAmbiguous('Authenticated inventory owner differs')
            resolved.append(row.ref); continue
        if set(row.to_dict())!={'title','lastRunTime'} or not row.title or row.last_run_time is None or row.id or row.slug or row.author:
            raise KernelIdentityAmbiguous('Unidentified inventory row cannot prove absence')
        if slugifier is None:
            from slugify import slugify
            slugifier=slugify
        candidate=slugifier(row.title)
        if not candidate or '/' in candidate: raise KernelIdentityAmbiguous('Reference-free title cannot identify an owned kernel')
        try: actual=kernel_response(api,owner+'/'+candidate).metadata
        except Exception: raise KernelIdentityAmbiguous('Reference-free kernel GET did not positively resolve identity') from None
        if actual.ref!=owner+'/'+candidate or actual.id<=0 or actual.title!=row.title or actual.last_run_time!=row.last_run_time:
            raise KernelIdentityAmbiguous('Reference-free kernel identity did not match its inventory row')
        resolved.append(actual.ref); positive.append({'ref':actual.ref,'id':actual.id,'title_sha256':hashlib.sha256(row.title.encode()).hexdigest()})
    if anchor not in resolved or len(set(resolved))!=len(resolved): raise KernelIdentityAmbiguous('Owner visibility anchor missing or inventory identities duplicated')
    return {'exists':target in resolved,'inventory_complete':True,'authenticated_owner':owner,'known_private_anchor':{'ref':anchor,'version':1,'is_private':True},
        'pages':pages,'addressable_count':len(resolved),'positive_resolutions':positive,'no_search_or_user_filter':True}


def object_exists(api,value,view):
    """Read-only diagnostic, never permission to infer absence from 403."""
    try: return {'exists':True,**view(api,value)}
    except Exception as error:
        status=getattr(error,'status',None) or getattr(getattr(error,'response',None),'status_code',None)
        if status==404: return {'exists':False}
        raise


class ReuseParentMismatch(ValueError):
    code='reuse_parent_drift'


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def verify_reuse_parent(api,contract,proof_folder,after_version=None):
    """Positive identity verification for ONE explicitly selected failed v1.

    This is not create-only/CAS. It cannot authorize updating any other object.
    All before bytes remain in the independent immutable parent archive.
    """
    if contract.get('ref')!=REUSE_REF or contract.get('kernel_id')!=REUSE_ID or contract.get('parent_version')!=1 or contract.get('source_sha256')!=REUSE_SOURCE_SHA:
        raise ReuseParentMismatch('Only the explicitly archived failed v1 may be reused')
    if after_version not in (None,2): raise ReuseParentMismatch('Only the explicit parent-v1 to candidate-v2 transition is permitted')
    if set(contract.get('v1_outputs_sha256',{}))!=REUSE_OUTPUTS or any(re.fullmatch(r'[0-9a-f]{64}',h) is None for h in contract['v1_outputs_sha256'].values()):
        raise ReuseParentMismatch('Complete fixed v1 output identities are required')
    folder=Path(proof_folder)
    if folder.exists(): raise FileExistsError('Reuse checks are immutable; use a fresh proof directory')
    folder.mkdir(parents=True)
    proof={'schema':'owned_failed_kernel_parent_check_v1','ref':REUSE_REF,'kernel_id':REUSE_ID,'expected_parent_version':1,
        'expected_latest_version':after_version or 1,'pinned_v1_requested_version':1,'pinned_metadata_version_semantics_asserted':False,
        'no_atomic_version_precondition':True,'checked_at_unix':time.time(),'status':'failed','metadata_comparisons':{}}
    def write(name,value): (folder/name).write_text(json.dumps(value,indent=2))
    try:
        for label,pinned in (('current',False),('pinned_v1',True)):
            response=kernel_response(api,REUSE_REF+'/1' if pinned else REUSE_REF,pinned)
            meta=response.metadata; document=meta.to_dict(ignore_defaults=False); source=response.blob.source.encode('utf-8')
            write(label+'.metadata.json',document); (folder/(label+'.source.py')).write_bytes(source)
            expected=contract['pinned_metadata_sha256'] if pinned else contract['current_metadata_sha256']
            original=json.loads((Path(contract['archive_path'])/(label+'.metadata.json')).read_text())
            if canonical_hash(original)!=expected: raise ReuseParentMismatch('Original archived metadata bytes drifted')
            current_hash=canonical_hash(document)
            proof['metadata_comparisons'][label]={'old_canonical_sha256':expected,'new_canonical_sha256':current_hash,
                'changed_fields':sorted(k for k in set(original)|set(document) if k not in original or k not in document or original[k]!=document[k]),
                'unchanged':current_hash==expected}
            versions=(1,2) if pinned and after_version==2 else ((1,) if pinned else (after_version or 1,))
            if pinned:
                proof['pinned_v1_metadata_reported_version']=meta.current_version_number
                proof['parent_metadata_unchanged']=current_hash==expected
            if meta.id!=REUSE_ID or meta.ref!=REUSE_REF or type(meta.current_version_number) is not int or meta.current_version_number not in versions or meta.is_private is not True or meta.enable_internet is not False or meta.title!=REUSE_TITLE:
                raise ReuseParentMismatch('Positive parent identity, privacy, title or version drifted')
            if pinned or after_version is None:
                if hashlib.sha256(source).hexdigest()!=REUSE_SOURCE_SHA: raise ReuseParentMismatch('Parent v1 source drifted')
            # Before an update, the complete original snapshots must match.
            # Afterward, metadata can expose container context; only old source,
            # full outputs and protected identity/privacy are invariants here.
            if after_version is None and current_hash!=expected: raise ReuseParentMismatch('Archived parent metadata drifted')
        status=api.kernels_status(REUSE_REF+'/1'); terminal=getattr(status.status,'name',str(status.status))
        write('pinned_v1.status.json',status.to_dict(ignore_defaults=False))
        if terminal!='ERROR': raise ReuseParentMismatch('The selected parent is no longer the failed terminal v1')
        raw=api.quota_view().gpu_quota
        quota={n:getattr(raw,n).total_seconds() for n in ('time_used','time_reserved','total_time_allowed','minimum_time_allowed')}
        quota['pay_to_scale_enabled']=raw.is_pay_to_scale_enabled; write('quota.json',quota)
        if quota['pay_to_scale_enabled'] is not False or (after_version is None and quota['time_reserved']!=0):
            raise ReuseParentMismatch('Parent reuse requires a clear non-paid GPU reservation')
        outputs=folder/'outputs'; outputs.mkdir()
        api.kernels_output(REUSE_REF+'/1',path=str(outputs),force=False,quiet=True)
        files={str(p.relative_to(outputs)).replace('\\','/'):sha(p) for p in outputs.rglob('*') if p.is_file()}
        if files!=contract['v1_outputs_sha256']: raise ReuseParentMismatch('Pinned v1 complete outputs drifted')
        proof.update(status='verified',latest_version=after_version or 1,parent_source_unchanged=True,
            parent_metadata_unchanged=proof['metadata_comparisons']['pinned_v1']['unchanged'],
            parent_complete_outputs_unchanged=True,v1_outputs_sha256=files,raw_gpu_quota=quota)
        return proof
    except BaseException as error:
        proof['error_type']=type(error).__name__
        raise
    finally: write('proof.json',proof)


def bounded_server_timeout(document,now=None):
    seconds=min(int(document['server_timeout_seconds']),int(document['absolute_deadline_unix']-(time.time() if now is None else now)))
    if not 300<=seconds<=10800: raise ValueError('Insufficient remaining absolute deadline before push')
    return seconds


def verify_dataset(api,document,version):
    value=document['dataset_ref']; info=dataset_view(api,value)
    if info['ref']!=value or info['is_private'] is not True or info['version']!=version: raise ValueError('Private pinned dataset identity differs')
    expected=document['payload_sha256']; owner,slug=ref(value)
    with tempfile.TemporaryDirectory(prefix='saber-private-baseline-') as temporary:
        api.dataset_download_files(f'{value}/{version}',path=temporary,force=False,quiet=True,unzip=False,licenses=[])
        with zipfile.ZipFile(Path(temporary)/(slug+'.zip')) as archive:
            names=archive.namelist()
            if len(names)!=len(set(names)) or set(names)!=set(expected) or sum(i.file_size for i in archive.infolist())>32*1024*1024 or any(
                PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n or ':' in n or
                (archive.getinfo(n).external_attr>>16)&0o170000==0o120000 for n in names): raise ValueError('Private dataset ZIP differs')
            actual={n:hashlib.sha256(archive.read(n)).hexdigest() for n in names}
    if actual!=expected: raise ValueError('Actual downloaded baseline bytes differ')
    return {**info,'files_sha256':actual,'actual_download_verified':True}


def execute(action,payload):
    # Fail before authentication on invalid mutation provenance. Read-only
    # verification is allowed after staging has released its original lease.
    if action in ('stage','verify-baseline'): intent(payload,'baseline_stage',require_lease=action=='stage')
    if action=='push': intent(payload,'kernel_push')
    if action in ('stage','push'):
        # A second interpreter must not replay a mutation merely because the
        # first interpreter's response was lost. This claim predates auth and
        # the very first blob/create/save request and is never auto-cleared.
        claim=Path(payload['intent_path']).parent/('baseline_transport_claim.json' if action=='stage' else 'kernel_transport_claim.json')
        with claim.open('x') as stream:
            json.dump({'action':action,'intent_sha256':payload['intent_sha256']},stream); stream.flush(); os.fsync(stream.fileno())
    if importlib.metadata.version('kaggle')!='2.2.5' or importlib.metadata.version('kagglesdk')!='0.1.38': raise RuntimeError('Verified isolated SDK versions required')
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi(); api.authenticate()
    if action=='quota':
        result=api.quota_view(); gpu=result.gpu_quota
        values={n:getattr(gpu,n).total_seconds() for n in ('time_used','time_reserved','total_time_allowed','minimum_time_allowed')}
        return {**values,'pay_to_scale_enabled':gpu.is_pay_to_scale_enabled,
            'remaining_seconds':values['total_time_allowed']-values['time_used'],'reserved_seconds':values['time_reserved']}
    if action=='dataset-exists':
        return object_exists(api,payload['ref'],dataset_view)
    if action=='kernel-exists':
        return object_exists(api,payload['ref'],kernel_view)
    if action=='kernel-inventory':
        return kernel_inventory(api,payload['ref'])
    if action=='verify-train':
        result=kernel_view(api,payload['ref'])
        if result['ref']!=payload['ref'] or result['is_private'] is not True or type(result['version']) is not int or result['version']<=0: raise ValueError('Private official training export identity differs')
        return result
    if action=='verify-reuse-parent':
        return verify_reuse_parent(api,payload['contract'],payload['proof_folder'])
    if action in ('stage','verify-baseline'):
        document=intent(payload,'baseline_stage',require_lease=action=='stage')
        if action=='verify-baseline': return verify_dataset(api,document,int(payload['version']))
        if document.get('reuse_parent'):
            verify_reuse_parent(api,document['reuse_parent'],Path(document['prepared_folder'])/'reuse_checks'/('stage_'+os.urandom(6).hex()))
        elif kernel_inventory(api,document['kernel_ref'])['exists'] is not False:
            raise ValueError('Baseline staging cannot precede an already existing target kernel')
        from kaggle.api.kaggle_api_extended import ResumableUploadContext
        from kagglesdk.blobs.types.blob_api_service import ApiBlobType
        from kagglesdk.datasets.types.dataset_api_service import ApiCreateDatasetRequest
        folder=Path(document['upload_folder']); expected=set(document['payload_sha256'])|{'dataset-metadata.json'}
        if {p.name for p in folder.iterdir()}!=expected or any(not p.is_file() or p.is_symlink() for p in folder.iterdir()): raise ValueError('Private upload folder exact members differ')
        metadata=json.loads((folder/'dataset-metadata.json').read_text()); owner,slug=ref(document['dataset_ref'])
        if document.get('creation_policy')!='create_new_only_no_update' or re.fullmatch(r'[0-9a-f]{12}',str(document.get('object_nonce',''))) is None or not slug.endswith('-'+document['object_nonce']): raise ValueError('Fresh create-only dataset identity differs')
        request=ApiCreateDatasetRequest(); request.id=None; request.owner_slug=owner; request.slug=slug; request.title=metadata['title']; request.license_name=metadata['licenses'][0]['name']
        request.subtitle=''; request.description='Private own scratch baseline; no images or predictions.'; request.files=[]; request.is_private=True; request.category_ids=[]
        # Blob-start/upload operations may resume their own byte transport.
        # The durable intent predates them. Metadata create itself is ONCE;
        # never call dataset_create_new(), whose with_retry repeats create.
        with ResumableUploadContext() as context:
            api.upload_files(request,[],str(folder),ApiBlobType.DATASET,context,True,'skip',['dataset-metadata.json'])
            intent(payload,'baseline_stage')
            with api.build_kaggle_client() as client: result=client.datasets.dataset_api_client.create_dataset(request)
        if result.error or result.ref!=document['dataset_ref'] or getattr(result,'invalid_tags',None): raise RuntimeError('One-shot private dataset create not accepted')
        progress={'ref':result.ref,'version':1,'metadata_create_returned':True}
        Path(document['progress_path']).write_text(json.dumps(progress))
        return progress
    if action=='push':
        document=intent(payload,'kernel_push'); folder=Path(document['kernel_folder'])
        reuse=document.get('reuse_parent')
        if reuse:
            verify_reuse_parent(api,reuse,Path(document['prepared_folder'])/'reuse_checks'/('prepush_'+os.urandom(6).hex()))
        else:
            absence=kernel_inventory(api,document['kernel_ref'])
            if absence['exists'] is not False: raise ValueError('Existing kernel cannot be updated by this campaign path')
        ledger=json.loads(Path(document['ledger_path']).read_text())
        charges=ledger.get('opening_charge_seconds',0)+sum(row['reserved_seconds'] for row in ledger.get('entries',[]))
        matching=[row for row in ledger.get('entries',[]) if row.get('run')==document['run'] and row.get('intent_sha256')==payload['intent_sha256']]
        if ledger.get('limit_seconds')!=90000 or charges>90000 or len(matching)!=1 or matching[0]['reserved_seconds']!=document['reserve_gpu_seconds']: raise ValueError('Durable cumulative GPU reservation differs before push')
        metadata=json.loads((folder/'kernel-metadata.json').read_text())
        if metadata['id']!=document['kernel_ref'] or metadata['is_private'] is not True or metadata['enable_internet'] is not False or metadata['competition_sources'] or metadata['kernel_sources']!=[document['train_export']] or metadata['dataset_sources']!=[document['baseline_pinned_ref']]: raise ValueError('Private kernel sources differ')
        if set(p.name for p in folder.iterdir())!={'kernel-metadata.json','augmix_entry.py'}: raise ValueError('Kernel upload must contain source only')
        if reuse and (metadata.get('id_no')!=REUSE_ID or metadata.get('title')!=REUSE_TITLE): raise ReuseParentMismatch('Explicit parent numeric locator differs')
        if not reuse and metadata.get('id_no') is not None: raise ValueError('New kernel path cannot target a numeric parent')
        raw=api.quota_view().gpu_quota
        if raw.time_reserved.total_seconds()!=0 or raw.is_pay_to_scale_enabled is not False or (raw.total_time_allowed-raw.time_used).total_seconds()<document['reserve_gpu_seconds']: raise ValueError('Raw GPU quota changed before single push')
        actual_timeout=bounded_server_timeout(document)
        result=api.kernels_push(str(folder),timeout=str(actual_timeout),acc='NvidiaTeslaT4')
        if result.error or any(getattr(result,n,None) for n in ('invalid_dataset_sources','invalid_kernel_sources','invalid_competition_sources','invalid_model_sources','invalid_tags')): raise RuntimeError('Kernel save response rejected sources')
        version=result.version_number
        expected_version=2 if reuse else 1
        if result.ref!=document['kernel_ref'] or type(version) is not int or version!=expected_version or (reuse and result.kernel_id!=REUSE_ID): raise ValueError('Actual accepted kernel id/ref/version differs')
        Path(document['progress_path']).write_text(json.dumps({'ref':result.ref,'version':version,'kernel_id':result.kernel_id,'save_returned_success':True}))
        actual=kernel_view(api,f'{result.ref}/{version}',True)
        source=hashlib.sha256((folder/'augmix_entry.py').read_text().encode('utf-8')).hexdigest()
        if actual['is_private'] is not True or actual['internet_enabled'] is not False or actual['ref']!=result.ref or actual['version']!=version or actual['datasets']!=[document['baseline_pinned_ref']] or actual['kernels']!=[document['train_export']] or actual['competitions'] or actual['source_sha256']!=source: raise ValueError('Actual private pinned kernel/source configuration differs')
        if reuse:
            check=verify_reuse_parent(api,reuse,Path(document['prepared_folder'])/'reuse_checks'/('postpush_'+os.urandom(6).hex()),after_version=2)
            if check['status']!='verified': raise ReuseParentMismatch('Parent v1 did not remain unchanged after new v2')
        return {'ref':result.ref,'version':version,'kernel_id':result.kernel_id,'private_requested':True,'private_verified':True,'source_sha256':source,
            'server_timeout_sent_seconds':actual_timeout,'parent_v1_content_unchanged':bool(check['parent_source_unchanged'] and check['parent_complete_outputs_unchanged']) if reuse else None,
            'parent_metadata_unchanged':check['parent_metadata_unchanged'] if reuse else None,
            'parent_metadata_comparisons':check['metadata_comparisons'] if reuse else None,'no_atomic_version_precondition':bool(reuse)}
    if action=='status':
        owner,slug,version=ref(payload['ref'],True); result=api.kernels_status('/'.join((owner,slug,version)))
        status=getattr(result.status,'name',str(result.status))
        return {'ref':owner+'/'+slug,'version':int(version),'status':status}
    if action=='output':
        ref(payload['ref'],True); target=Path(payload['folder'])
        if target.exists(): raise FileExistsError('Use a fresh output download directory')
        target.mkdir(parents=True)
        api.kernels_output(payload['ref'],path=str(target),file_pattern=r'(^|/)(augmix_artifacts\.private\.zip|augmix_output_manifest\.json)$',force=False,quiet=True)
        return {'files':[str(p.resolve()) for p in target.rglob('*') if p.is_file()]}
    raise ValueError('Unknown isolated operation')


def main():
    action=sys.argv[1]; payload=json.load(sys.stdin)
    try:
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()): result=execute(action,payload)
        print(json.dumps({'ok':True,'result':result}))
    except BaseException as error:
        print(json.dumps({'ok':False,'error_type':type(error).__name__,'error_code':getattr(error,'code',None)})); raise SystemExit(1) from None


if __name__=='__main__': main()
