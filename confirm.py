"""One-time evaluation of the already selected recipe on untouched confirmation."""
from runtime import ROOT, seed_all, write_json
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from data import load_training_arrays, load_confirmation_indices
from model import build_model
from train import evaluate, tensor_images


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'results/confirmation.json')
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=p.parse_args(); seed_all(); device=torch.device(args.device)
    digest=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    selection=json.loads(args.selection.read_text())
    selection_digest=hashlib.sha256(args.selection.read_bytes()).hexdigest()
    if selection['checkpoint_sha256']!=digest: raise ValueError('Selection checkpoint fingerprint mismatch')
    if args.output.exists():
        previous=json.loads(args.output.read_text())
        if previous['checkpoint_sha256']!=digest: raise RuntimeError('Confirmation was already reserved for another model')
        if (previous['selection_file_sha256']!=selection_digest or previous['flip_tta']!=selection['flip_tta']
                or previous['run']!=selection['run']):
            raise RuntimeError('Confirmation was already reserved for a different frozen selection')
        if previous.get('status')=='complete':
            manifest=json.loads((args.data_dir/'training_manifest.json').read_text())
            if previous['split_sha256']!=manifest['split']['indices_sha256']:
                raise RuntimeError('Cached confirmation uses a different data split')
            print(json.dumps(previous,indent=2)); return
    images,labels,_,_=load_training_arrays(args.data_dir)
    ids=load_confirmation_indices(args.data_dir)
    ckpt=torch.load(args.checkpoint,map_location=device,weights_only=False)
    if ckpt['config'].get('train_all',False):
        raise ValueError('A full-data model cannot be evaluated on a held-out confirmation set it trained on')
    manifest=json.loads((args.data_dir/'training_manifest.json').read_text())
    if ckpt['data_signature']['split_sha256']!=manifest['split']['indices_sha256']:
        raise ValueError('Confirmation partition does not match training provenance')
    receipt={'status':'started','run':selection['run'],'checkpoint_sha256':digest,
             'flip_tta':selection['flip_tta'],'examples':len(ids),
             'selection_file_sha256':selection_digest,
             'confirmation_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(args.output,receipt)
    model=build_model(ckpt['model_config']).to(device=device,memory_format=torch.channels_last)
    model.load_state_dict(ckpt['model'])
    result=evaluate(model,tensor_images(images[ids],device),torch.as_tensor(labels[ids],device=device),
                    device,flip_tta=selection['flip_tta'])
    confusion=np.zeros((10,10),dtype=np.int64)
    np.add.at(confusion,(labels[ids],result['predictions']),1)
    receipt.update(status='complete',accuracy=result['accuracy'],loss=result['loss'],
                   confusion_matrix=confusion.tolist(),split_sha256=manifest['split']['indices_sha256'])
    write_json(args.output,receipt)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
