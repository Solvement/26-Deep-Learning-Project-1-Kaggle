"""Run fixed-budget architecture/recipe comparisons, confirm once, then retrain.

Kaggle upload is a separate explicit delivery step after reviewing artifacts.
"""
from runtime import ROOT, write_json
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
import torch
from train import DEFAULT_CONFIG, code_provenance, training_source
from predict import validate_submission
from data import load_kaggle_test


def verify_completed(folder, requested_config):
    """Reuse an experiment only when its actual code/config/checkpoint still match."""
    metrics_path=folder/'metrics.json'
    if not metrics_path.exists(): return False
    metrics=json.loads(metrics_path.read_text())
    if metrics.get('status')!='complete': return False
    expected=DEFAULT_CONFIG.copy(); expected.update(requested_config)
    actual=json.loads((folder/'config.json').read_text())
    if actual!=expected or metrics['config']!=expected or metrics['epochs_completed']!=expected['epochs']:
        raise RuntimeError(f'Stale completed experiment configuration: {folder}')
    provenance=json.loads((folder/'provenance.json').read_text())
    if provenance['code']['source_sha256']!=code_provenance()['source_sha256']:
        raise RuntimeError(f'Stale completed experiment source: {folder}')
    manifest=json.loads((ROOT/'data/training_manifest.json').read_text())
    source=training_source(ROOT/'data')
    checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    signature={'split_sha256':manifest['split']['indices_sha256'],'archive_sha256':source['archive_sha256']}
    if checkpoint['config']!=expected or checkpoint['data_signature']!=signature:
        raise RuntimeError(f'Stale completed checkpoint inputs: {folder}')
    return True


def verify_evaluation(folder):
    path=folder/'evaluation.json'
    if not path.exists(): return False
    evaluation=json.loads(path.read_text())
    digest=hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest()
    if evaluation['checkpoint_sha256']!=digest:
        raise RuntimeError(f'Stale evaluation checkpoint: {folder}')
    config=json.loads((folder/'config.json').read_text())
    if config.get('train_all'):
        if evaluation['validation_accuracy'] is not None or evaluation['flip_tta']!=config['fixed_flip_tta']:
            raise RuntimeError('Final full-data evaluation contains invalid validation/TTA metadata')
        load_kaggle_test(ROOT/'data')
        current=validate_submission(folder/'submission.csv',ROOT/'data')
        if current['sha256']!=evaluation['submission']['sha256']:
            raise RuntimeError('Final submission changed after evaluation')
    return True


def run(command, log_path, deadline):
    remaining=deadline-time.monotonic()
    if remaining<=0: raise TimeoutError('Campaign wall-time budget exhausted')
    log_path.parent.mkdir(parents=True,exist_ok=True)
    print(json.dumps({'event':'stage_started','command':command,'log':str(log_path),
        'time':datetime.now(ZoneInfo('America/New_York')).isoformat()}),flush=True)
    with log_path.open('a',encoding='utf-8') as log:
        subprocess.run([sys.executable,*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                       check=True,timeout=remaining)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-seconds',type=int,default=10800)
    args=parser.parse_args(); deadline=time.monotonic()+args.max_seconds
    results=ROOT/'results'; results.mkdir(exist_ok=True)
    experiments=[('baseline','configs/wrn_baseline.json'),('architecture','configs/wrn_architecture.json'),
                 ('regularized','configs/wrn_final.json')]
    for name,config in experiments:
        folder=results/name
        requested=json.loads((ROOT/config).read_text())
        if not verify_completed(folder,requested):
            command=['train.py','--config',config,'--output-dir',str(folder),'--device','cuda']
            if (folder/'last.pt').exists(): command+=['--resume']
            run(command,folder/'train.log',deadline)
        if not verify_evaluation(folder):
            run(['predict.py','--checkpoint',str(folder/'best.pt'),'--output-dir',str(folder),
                 '--validation-only','--device','cuda'],folder/'evaluation.log',deadline)
        print(json.dumps({'event':'experiment_complete','run':name,
              'metrics':json.loads((folder/'evaluation.json').read_text())['validation_accuracy']}),flush=True)
    selection_path=results/'selection.json'
    if selection_path.exists():
        selection=json.loads(selection_path.read_text())
        scores={name:json.loads((results/name/'evaluation.json').read_text()) for name,_ in experiments}
        chosen=max(scores,key=lambda n:(scores[n]['validation_accuracy'],-scores[n]['validation_loss']))
        expected=scores[chosen]
        if (selection['run']!=chosen or selection['checkpoint_sha256']!=expected['checkpoint_sha256']
                or selection['flip_tta']!=expected['flip_tta'] or selection['variant']!=expected['variant']
                or selection['all_development_accuracies']!={k:v['validation_accuracy'] for k,v in scores.items()}):
            raise RuntimeError('Frozen selection no longer matches verified development evaluations')
    else:
        scores={name:json.loads((results/name/'evaluation.json').read_text()) for name,_ in experiments}
        chosen=max(scores,key=lambda n:(scores[n]['validation_accuracy'],-scores[n]['validation_loss']))
        evaluation=scores[chosen]
        selection={'run':chosen,'criterion':'highest development accuracy; lower NLL as tie-breaker',
            'validation_accuracy':evaluation['validation_accuracy'],'flip_tta':evaluation['flip_tta'],
            'checkpoint_sha256':evaluation['checkpoint_sha256'],'variant':evaluation['variant'],
            'all_development_accuracies':{k:v['validation_accuracy'] for k,v in scores.items()},
            'frozen_at':datetime.now(ZoneInfo('America/New_York')).isoformat(),
            'confirmation_used_for_selection':False}
        write_json(selection_path,selection)
    chosen=selection['run']
    run(['confirm.py','--checkpoint',str(results/chosen/'best.pt'),'--selection',str(selection_path),
         '--output',str(results/'confirmation.json')],results/'confirmation.log',deadline)
    # Recipe and duration remain fixed after observing confirmation. No retuning.
    config=json.loads((results/chosen/'config.json').read_text())
    config.update(train_all=True,final_variant=selection['variant'],fixed_flip_tta=selection['flip_tta'])
    final_config=results/'final_full_config.json'
    if final_config.exists() and json.loads(final_config.read_text())!=config:
        raise RuntimeError('Frozen final training recipe changed')
    write_json(final_config,config)
    folder=results/'final_full'
    if not verify_completed(folder,config):
        command=['train.py','--config',str(final_config),'--output-dir',str(folder),'--device','cuda']
        if (folder/'last.pt').exists():command+=['--resume']
        run(command,folder/'train.log',deadline)
    if not verify_evaluation(folder):
        run(['predict.py','--checkpoint',str(folder/'best.pt'),'--output-dir',str(folder),'--device','cuda'],
            folder/'evaluation.log',deadline)
    write_json(results/'campaign_status.json',{'status':'complete','selected_run':chosen,
        'confirmation':json.loads((results/'confirmation.json').read_text()),
        'final_submission':json.loads((folder/'evaluation.json').read_text())['submission']})
    print('CAMPAIGN_COMPLETE: final_full/submission.csv validated, ready for authorized Kaggle delivery.',flush=True)


if __name__=='__main__':main()
