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
import zipfile


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


def kernel_view(api,value,pinned=False):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    pieces=ref(value,pinned); request=ApiGetKernelRequest(); request.user_name=pieces[0]; request.kernel_slug=pieces[1]
    if pinned: request.version_label='v'+pieces[2]
    with api.build_kaggle_client() as client: result=client.kernels.kernels_api_client.get_kernel(request)
    metadata=result.metadata
    return {'ref':metadata.ref,'version':metadata.current_version_number,'is_private':metadata.is_private,
        'internet_enabled':metadata.enable_internet,'datasets':metadata.dataset_data_sources,'kernels':metadata.kernel_data_sources,
        'competitions':metadata.competition_data_sources,'source_sha256':hashlib.sha256(result.blob.source.encode('utf-8')).hexdigest()}


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
        try: return {'exists':True,**dataset_view(api,payload['ref'])}
        except Exception as error:
            status=getattr(error,'status',None) or getattr(getattr(error,'response',None),'status_code',None)
            if status==404: return {'exists':False}
            raise
    if action=='kernel-exists':
        try: return {'exists':True,**kernel_view(api,payload['ref'])}
        except Exception as error:
            status=getattr(error,'status',None) or getattr(getattr(error,'response',None),'status_code',None)
            if status==404: return {'exists':False}
            raise
    if action=='verify-train':
        result=kernel_view(api,payload['ref'])
        if result['ref']!=payload['ref'] or result['is_private'] is not True or type(result['version']) is not int or result['version']<=0: raise ValueError('Private official training export identity differs')
        return result
    if action in ('stage','verify-baseline'):
        document=intent(payload,'baseline_stage',require_lease=action=='stage')
        if action=='verify-baseline': return verify_dataset(api,document,int(payload['version']))
        from kaggle.api.kaggle_api_extended import ResumableUploadContext
        from kagglesdk.blobs.types.blob_api_service import ApiBlobType
        from kagglesdk.datasets.types.dataset_api_service import ApiCreateDatasetRequest
        folder=Path(document['upload_folder']); expected=set(document['payload_sha256'])|{'dataset-metadata.json'}
        if {p.name for p in folder.iterdir()}!=expected or any(not p.is_file() or p.is_symlink() for p in folder.iterdir()): raise ValueError('Private upload folder exact members differ')
        metadata=json.loads((folder/'dataset-metadata.json').read_text()); owner,slug=ref(document['dataset_ref'])
        request=ApiCreateDatasetRequest(); request.owner_slug=owner; request.slug=slug; request.title=metadata['title']; request.license_name=metadata['licenses'][0]['name']
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
        ledger=json.loads(Path(document['ledger_path']).read_text())
        charges=ledger.get('opening_charge_seconds',0)+sum(row['reserved_seconds'] for row in ledger.get('entries',[]))
        matching=[row for row in ledger.get('entries',[]) if row.get('run')==document['run'] and row.get('intent_sha256')==payload['intent_sha256']]
        if ledger.get('limit_seconds')!=90000 or charges>90000 or len(matching)!=1 or matching[0]['reserved_seconds']!=document['reserve_gpu_seconds']: raise ValueError('Durable cumulative GPU reservation differs before push')
        metadata=json.loads((folder/'kernel-metadata.json').read_text())
        if metadata['id']!=document['kernel_ref'] or metadata['is_private'] is not True or metadata['enable_internet'] is not False or metadata['competition_sources'] or metadata['kernel_sources']!=[document['train_export']] or metadata['dataset_sources']!=[document['baseline_pinned_ref']]: raise ValueError('Private kernel sources differ')
        if set(p.name for p in folder.iterdir())!={'kernel-metadata.json','augmix_entry.py'}: raise ValueError('Kernel upload must contain source only')
        raw=api.quota_view().gpu_quota
        if raw.time_reserved.total_seconds()!=0 or raw.is_pay_to_scale_enabled is not False or (raw.total_time_allowed-raw.time_used).total_seconds()<document['reserve_gpu_seconds']: raise ValueError('Raw GPU quota changed before single push')
        result=api.kernels_push(str(folder),timeout=str(document['server_timeout_seconds']),acc='NvidiaTeslaT4')
        if result.error or any(getattr(result,n,None) for n in ('invalid_dataset_sources','invalid_kernel_sources','invalid_competition_sources','invalid_model_sources','invalid_tags')): raise RuntimeError('Kernel save response rejected sources')
        version=result.version_number
        if result.ref!=document['kernel_ref'] or type(version) is not int or version!=1: raise ValueError('Actual accepted new kernel version differs')
        Path(document['progress_path']).write_text(json.dumps({'ref':result.ref,'version':version,'save_returned_success':True}))
        actual=kernel_view(api,f'{result.ref}/{version}',True)
        source=hashlib.sha256((folder/'augmix_entry.py').read_text().encode('utf-8')).hexdigest()
        if actual['is_private'] is not True or actual['internet_enabled'] is not False or actual['ref']!=result.ref or actual['version']!=version or actual['datasets']!=[document['baseline_pinned_ref']] or actual['kernels']!=[document['train_export']] or actual['competitions'] or actual['source_sha256']!=source: raise ValueError('Actual private pinned kernel/source configuration differs')
        return {'ref':result.ref,'version':version,'private_requested':True,'private_verified':True,'source_sha256':source}
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
        print(json.dumps({'ok':False,'error_type':type(error).__name__})); raise SystemExit(1) from None


if __name__=='__main__': main()
