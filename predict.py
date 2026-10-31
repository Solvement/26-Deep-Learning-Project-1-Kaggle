"""Validate a scratch checkpoint and emit all official IDs in sample order."""
from runtime import ROOT, seed_all, write_json
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from model import build_model, count_parameters
from data import load_training_arrays, load_kaggle_test
from train import evaluate, normalize, tensor_images, training_source


def validate_submission(path, data_dir):
    with (data_dir/'sample_submission.csv').open(newline='') as f:
        expected=list(csv.DictReader(f))
    with Path(path).open(newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != ['id','label']: raise ValueError('Expected exactly id,label columns')
        rows=list(reader)
    if len(rows)!=2000 or len(rows)!=len(expected): raise ValueError('Expected exactly 2000 predictions')
    if [r['id'] for r in rows] != [r['id'] for r in expected]: raise ValueError('IDs/order mismatch official sample')
    if len({r['id'] for r in rows})!=len(rows): raise ValueError('Duplicate IDs')
    for row in rows:
        if row['label'] not in {str(i) for i in range(10)}: raise ValueError('Labels must be integer 0..9')
    return {'rows':len(rows),'columns':['id','label'],'ids_match_official_order':True,
            'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--validation-only',action='store_true')
    parser.add_argument('--flip-tta',choices=('auto','yes','no'),default='auto')
    args=parser.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    seed_all(); device=torch.device(args.device)
    ckpt=torch.load(args.checkpoint,map_location=device,weights_only=False)
    model=build_model(ckpt['model_config']).to(device=device,memory_format=torch.channels_last)
    model.load_state_dict(ckpt['model']); model.eval()
    if count_parameters(model)>5_000_000: raise ValueError('Model exceeds parameter limit')
    images,labels,train_ids,val_ids=load_training_arrays(args.data_dir)
    signature=json.loads((args.data_dir/'training_manifest.json').read_text())['split']['indices_sha256']
    source=training_source(args.data_dir)
    if (signature!=ckpt['data_signature']['split_sha256'] or
            source['archive_sha256']!=ckpt['data_signature']['archive_sha256']):
        raise ValueError('Checkpoint and official evaluation split/source mismatch')
    full=ckpt['config'].get('train_all',False)
    if full and args.validation_only: raise ValueError('Full-data retrain has no held-out validation set')
    if full:
        if args.flip_tta=='auto':
            use_flip=bool(ckpt['config']['fixed_flip_tta'])
        else: use_flip=args.flip_tta=='yes'
        plain={'accuracy':None}; flip={'accuracy':None}
        selected={'accuracy':None,'loss':None,'predictions':np.array([],dtype=np.int64)}
        val_ids=np.array([],dtype=np.int64)
    else:
        x=tensor_images(images[val_ids],device); y=torch.as_tensor(labels[val_ids],device=device)
        plain=evaluate(model,x,y,device)
        flip=evaluate(model,x,y,device,flip_tta=True)
        # TTA selection uses only development data, before confirmation/hidden prediction.
        use_flip=((flip['accuracy'],-flip['loss']) > (plain['accuracy'],-plain['loss'])) if args.flip_tta=='auto' else args.flip_tta=='yes'
        selected=flip if use_flip else plain
    confusion=np.zeros((10,10),dtype=np.int64)
    np.add.at(confusion,(labels[val_ids],selected['predictions']),1)
    result={'checkpoint':str(args.checkpoint),'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'epoch':ckpt['epoch'],'variant':ckpt['variant'],'model_config':ckpt['model_config'],
            'parameter_count':count_parameters(model),'trainable_parameter_count':count_parameters(model,True),
            'validation_examples':len(val_ids),'plain_validation_accuracy':plain['accuracy'],
            'flip_validation_accuracy':flip['accuracy'],'validation_accuracy':selected['accuracy'],
            'validation_loss':selected['loss'],'flip_tta':use_flip,'confusion_matrix':confusion.tolist(),
            'private_test_accuracy':None,'public_test_accuracy':None,'submitted_to_kaggle':False,
            'data_signature':ckpt['data_signature'],'provenance':ckpt['provenance']}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if not args.validation_only:
        ids,test_images=load_kaggle_test(args.data_dir)
        test_x=tensor_images(test_images,device); predictions=[]
        with torch.no_grad():
            for start in range(0,len(ids),256):
                batch=normalize(test_x[start:start+256])
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    logits=model(batch)
                    if use_flip: logits=(logits+model(batch.flip(-1)))*0.5
                predictions.extend(logits.argmax(1).cpu().tolist())
        path=args.output_dir/'submission.csv'
        with path.open('w',newline='') as f:
            writer=csv.writer(f); writer.writerow(['id','label']); writer.writerows(zip(ids,predictions))
        result['submission']=validate_submission(path,args.data_dir)
        result['submission']['path']=str(path)
    write_json(args.output_dir/'evaluation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('confusion_matrix','provenance')},indent=2),flush=True)


if __name__=='__main__': main()
