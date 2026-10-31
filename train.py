"""Scratch CIFAR WideResNet training with official inputs and exact resume state."""
from __future__ import annotations

# Runtime must isolate the Windows cuDNN DLLs before torch is imported.
from runtime import ROOT, seed_all, environment, write_json, configure_runtime
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from torch.nn import functional as F
from model import build_model, count_parameters
from data import load_training_arrays

DEFAULT_CONFIG = {
    'depth': 22, 'widen_factor': 4, 'dropout': 0.0, 'num_classes': 10,
    'epochs': 160, 'batch_size': 128, 'lr': 0.1, 'min_lr': 0.0001,
    'warmup_epochs': 5, 'momentum': 0.9, 'weight_decay': 0.0005,
    'label_smoothing': 0.0, 'cutout': 16, 'mixup_alpha': 0.2,
    'mixup_probability': 0.5, 'ema_decay': 0.995, 'seed': 2026,
    'split_seed': 2026,
    'train_all': False, 'final_variant': 'ema', 'fixed_flip_tta': False,
}
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_provenance():
    code = {p: sha256(ROOT/p) for p in ('train.py','model.py','data.py','runtime.py')}
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {'git_commit': commit, 'source_sha256': code}


def training_source(data_dir):
    """Canonical source identity for either official PNGs or a private export."""
    manifest=json.loads((Path(data_dir)/'training_manifest.json').read_text())
    source=dict(manifest['source_provenance'])
    source['archive_sha256']=source.get('official_train_npz_sha256',source['ordered_training_png_sha256'])
    source['train_csv_sha256']=source['train_labels_csv_sha256']
    source['competition']=manifest['competition']
    source['source']=manifest['source_type']
    return source


def tensor_images(images, device):
    # Store data once as uint8 NHWC; no CPU worker processes or repeated transfers.
    return torch.as_tensor(images, device=device)


def normalize(images):
    images = images.float().permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)/255.0
    mean = images.new_tensor(MEAN).view(1,3,1,1)
    std = images.new_tensor(STD).view(1,3,1,1)
    return (images-mean)/std


def augment(images, cutout=16):
    n = len(images)
    device = images.device
    # Reflection pad and independent integer crop offsets, matching RandomCrop.
    padded = F.pad(images.permute(0,3,1,2), (4,4,4,4), mode='reflect').permute(0,2,3,1)
    row = torch.arange(32, device=device)[None,:]+torch.randint(0,9,(n,1),device=device)
    col = torch.arange(32, device=device)[None,:]+torch.randint(0,9,(n,1),device=device)
    col = torch.where(torch.rand(n,1,device=device)<0.5, col.flip(1), col)
    x = padded[torch.arange(n,device=device)[:,None,None],row[:,:,None],col[:,None,:]]
    x = normalize(x)
    if cutout:
        cy = torch.randint(0,32,(n,1,1),device=device)
        cx = torch.randint(0,32,(n,1,1),device=device)
        axis = torch.arange(32,device=device)
        keep = (axis[None,:,None] < cy-cutout//2) | (axis[None,:,None] >= cy+cutout//2)
        keep = keep | (axis[None,None,:] < cx-cutout//2) | (axis[None,None,:] >= cx+cutout//2)
        x = x * keep[:,None]
    return x.contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def evaluate(model, images, labels, device, batch_size=256, flip_tta=False):
    model.eval()
    correct = torch.zeros((),device=device,dtype=torch.int64)
    loss = torch.zeros((),device=device)
    predictions = []
    for start in range(0,len(labels),batch_size):
        x = normalize(images[start:start+batch_size])
        y = labels[start:start+batch_size]
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            logits = model(x)
            if flip_tta:
                logits = (logits + model(x.flip(-1)))*0.5
            loss += F.cross_entropy(logits.float(),y,reduction='sum')
        predicted = logits.argmax(1)
        correct += predicted.eq(y).sum()
        predictions.append(predicted.cpu().numpy())
    return {'loss': loss.item()/len(labels), 'accuracy': correct.item()/len(labels),
            'predictions': np.concatenate(predictions)}


def schedule(step, total_steps, steps_per_epoch, config):
    warmup = config['warmup_epochs']*steps_per_epoch
    if step < warmup:
        return config['lr']*(step+1)/max(1,warmup)
    progress = (step-warmup)/max(1,total_steps-warmup-1)
    return config['min_lr']+(config['lr']-config['min_lr'])*0.5*(1+math.cos(math.pi*progress))


def save_checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)


def benchmark(config, device, steps, output):
    model = build_model({k:config[k] for k in ('depth','widen_factor','dropout','num_classes')})
    model = model.to(device=device,memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9,nesterov=True)
    images = torch.randint(0,256,(config['batch_size'],32,32,3),device=device,dtype=torch.uint8)
    labels = torch.randint(0,10,(config['batch_size'],),device=device)
    durations=[]
    for i in range(steps+3):
        if device.type=='cuda': torch.cuda.synchronize()
        t=time.perf_counter()
        x=augment(images,config['cutout'])
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            loss=F.cross_entropy(model(x),labels)
        loss.backward(); opt.step()
        if device.type=='cuda': torch.cuda.synchronize()
        if i>=3: durations.append(time.perf_counter()-t)
    result={'engineering_benchmark_only':True,'synthetic_data':True,'batch_size':config['batch_size'],
            'parameters':count_parameters(model),'median_step_seconds':float(np.median(durations)),
            'estimated_epoch_seconds':float(np.median(durations))*math.ceil(45000/config['batch_size']),
            'peak_gpu_memory_GiB':torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else None,
            'environment':environment()}
    write_json(output/'benchmark.json',result)
    print(json.dumps(result,indent=2),flush=True)


def train(config, output, data_dir, device, stop_after_epoch=None, resume=False):
    images, labels, train_ids, val_ids = load_training_arrays(data_dir)
    source = training_source(data_dir)
    data_manifest = json.loads((data_dir/'training_manifest.json').read_text())
    if data_manifest['split']['seed'] != config['split_seed']:
        raise ValueError('Configured split seed does not match prepared official split')
    signature = {'archive_sha256':source['archive_sha256'],
                 'split_sha256':data_manifest['split']['indices_sha256']}
    provenance = code_provenance()
    if config['train_all']:
        train_ids=np.arange(len(labels),dtype=np.int64)
        val_ids=np.array([],dtype=np.int64)
    train_x=tensor_images(images[train_ids],device)
    train_y=torch.as_tensor(labels[train_ids],device=device)
    val_x=tensor_images(images[val_ids],device)
    val_y=torch.as_tensor(labels[val_ids],device=device)
    model_config={k:config[k] for k in ('depth','widen_factor','dropout','num_classes')}
    model=build_model(model_config).to(device=device,memory_format=torch.channels_last)
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    opt=torch.optim.SGD(model.parameters(),lr=config['lr'],momentum=config['momentum'],
                        weight_decay=config['weight_decay'],nesterov=True)
    model_params=list(model.parameters()); ema_params=list(ema.parameters())
    steps_per_epoch=math.ceil(len(train_ids)/config['batch_size'])
    total_steps=config['epochs']*steps_per_epoch
    history=[]; start_epoch=1; updates=0; best_accuracy=-1; best_loss=float('inf'); previous_seconds=0
    if resume:
        last=torch.load(output/'last.pt',map_location=device,weights_only=False)
        if last['config'] != config or last['data_signature'] != signature:
            raise ValueError('Resume configuration or official data/split mismatch')
        if last['provenance']['source_sha256'] != provenance['source_sha256']:
            raise ValueError('Training source changed: resume requires original source files')
        model.load_state_dict(last['model']); ema.load_state_dict(last['ema']); opt.load_state_dict(last['optimizer'])
        history=last['history']; start_epoch=last['epoch']+1; updates=last['updates']
        best_accuracy=last['best_accuracy']; best_loss=last['best_loss']; previous_seconds=last['elapsed_seconds']
        random.setstate(last['python_rng']); np.random.set_state(last['numpy_rng'])
        torch.set_rng_state(last['torch_rng'].cpu())
        if device.type=='cuda': torch.cuda.set_rng_state_all([x.cpu() for x in last['cuda_rng']])
    elif (output/'last.pt').exists():
        raise FileExistsError('Existing experiment: use --resume or a new output directory')
    write_json(output/'config.json',config)
    write_json(output/'environment.json',environment())
    write_json(output/'provenance.json',{'code':provenance,'official_data':source,'data_signature':signature,
                                      'train_examples':len(train_ids),'validation_examples':len(val_ids),
                                      'parameter_count':count_parameters(model),'normalization':{'mean':MEAN,'std':STD}})
    print(json.dumps({'event':'training_started','config':config,'train_examples':len(train_ids),
                      'validation_examples':len(val_ids),'parameters':count_parameters(model),'device':str(device)}),flush=True)
    started=time.perf_counter()
    end_epoch=min(config['epochs'],stop_after_epoch or config['epochs'])
    for epoch in range(start_epoch,end_epoch+1):
        epoch_start=time.perf_counter(); model.train()
        order=torch.randperm(len(train_y),device=device)
        loss_sum=torch.zeros((),device=device); correct=torch.zeros((),device=device); seen=0
        for batch,start in enumerate(range(0,len(order),config['batch_size'])):
            step=(epoch-1)*steps_per_epoch+batch
            lr=schedule(step,total_steps,steps_per_epoch,config)
            for group in opt.param_groups: group['lr']=lr
            ids=order[start:start+config['batch_size']]; y=train_y[ids]
            x=augment(train_x[ids],config['cutout'])
            mix=(config['mixup_alpha']>0 and random.random()<config['mixup_probability'])
            lam=float(np.random.beta(config['mixup_alpha'],config['mixup_alpha'])) if mix else 1.0
            perm=torch.randperm(len(y),device=device) if mix else None
            if mix: x=x*lam+x[perm]*(1-lam)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                logits=model(x)
                loss=F.cross_entropy(logits,y,label_smoothing=config['label_smoothing'])*lam
                if mix: loss=loss+F.cross_entropy(logits,y[perm],label_smoothing=config['label_smoothing'])*(1-lam)
            if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at epoch {epoch}, batch {batch}')
            loss.backward(); opt.step(); updates+=1
            with torch.no_grad():
                decay=min(config['ema_decay'],(1+updates)/(10+updates))
                torch._foreach_lerp_(ema_params,model_params,1-decay)
                predicted=logits.detach().argmax(1)
                correct+=predicted.eq(y).sum()*lam
                if mix: correct+=predicted.eq(y[perm]).sum()*(1-lam)
                loss_sum+=loss.detach()*len(y); seen+=len(y)
        # Copy running BN statistics from the online model; EMA is one inference model.
        with torch.no_grad():
            for a,b in zip(ema.buffers(),model.buffers()): a.copy_(b)
        if config['train_all']:
            regular={'loss':None,'accuracy':None}; average=regular.copy()
            choice=config['final_variant']; metric={'loss':None,'accuracy':None}
        else:
            regular=evaluate(model,val_x,val_y,device)
            average=evaluate(ema,val_x,val_y,device)
            choice='ema' if (average['accuracy'],-average['loss']) >= (regular['accuracy'],-regular['loss']) else 'model'
            metric=average if choice=='ema' else regular
        elapsed=previous_seconds+time.perf_counter()-started
        record={'epoch':epoch,'train_loss':loss_sum.item()/seen,'train_accuracy':correct.item()/seen,
                'validation_loss':regular['loss'],'validation_accuracy':regular['accuracy'],
                'ema_validation_loss':average['loss'],'ema_validation_accuracy':average['accuracy'],
                'selected_variant':choice,'lr':lr,'epoch_seconds':time.perf_counter()-epoch_start,
                'elapsed_seconds':elapsed}
        history.append(record)
        improved=(not config['train_all'] and (metric['accuracy'],-metric['loss']) > (best_accuracy,-best_loss))
        fixed_final=(config['train_all'] and epoch==config['epochs'])
        if improved or fixed_final:
            best_accuracy=metric['accuracy']; best_loss=metric['loss']
            selected=ema if choice=='ema' else model
            save_checkpoint(output/'best.pt',{'model':selected.state_dict(),'model_config':model_config,'config':config,
                'epoch':epoch,'variant':choice,'validation_accuracy':best_accuracy,'validation_loss':best_loss,
                'parameter_count':count_parameters(selected),'data_signature':signature,'provenance':provenance})
        write_json(output/'history.json',history)
        write_json(output/'metrics.json',{'status':'complete' if epoch==config['epochs'] else 'running',
            'epochs_completed':epoch,'best_validation_accuracy':None if config['train_all'] else best_accuracy,
            'best_validation_loss':None if config['train_all'] else best_loss,
            'parameter_count':count_parameters(model),'elapsed_seconds':elapsed,
            'training_examples':len(train_ids),'validation_examples':len(val_ids),'config':config})
        save_checkpoint(output/'last.pt',{'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':opt.state_dict(),
            'epoch':epoch,'updates':updates,'config':config,'data_signature':signature,'provenance':provenance,
            'history':history,'best_accuracy':best_accuracy,'best_loss':best_loss,'elapsed_seconds':elapsed,
            'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []})
        print(json.dumps(record),flush=True)
    print(json.dumps({'event':'training_complete' if end_epoch==config['epochs'] else 'training_paused',
                      'best_validation_accuracy':None if config['train_all'] else best_accuracy,'epochs_completed':end_epoch}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--stop-after-epoch',type=int)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--benchmark-steps',type=int,default=0)
    args=parser.parse_args()
    config=DEFAULT_CONFIG.copy()
    if args.config: config.update(json.loads(args.config.read_text()))
    if config['epochs']<1 or config['batch_size']<1 or config['warmup_epochs']>=config['epochs']:
        raise ValueError('Positive epochs/batch size and warmup < epochs are required')
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    seed_all(config['seed']); device=torch.device(args.device)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.benchmark_steps: benchmark(config,device,args.benchmark_steps,args.output_dir)
    else: train(config,args.output_dir,args.data_dir,device,args.stop_after_epoch,args.resume)


if __name__=='__main__': main()
