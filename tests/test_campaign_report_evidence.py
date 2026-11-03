"""Pure synthetic report contracts; no kernels, arrays, tensors or real scores."""
import ast
from contextlib import ExitStack,redirect_stdout
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from scripts import campaign_report_evidence as report
from tests import test_incumbent_compare as fixtures


class CampaignReportContracts(unittest.TestCase):
    def doc(self):
        doc=fixtures.IncumbentComparisonTests().fixture()
        baseline={'flip':copy.deepcopy(doc['policies']['flip']),'evaluation_context':doc['evaluation_context'],
            'data_signature':doc['data_signature'],'checkpoint_sha256':'f'*64,'benchmark_sha256':'e'*64}
        doc['baseline_sha256']=baseline['benchmark_sha256']
        pixels={c:doc['policies']['flip']['cases'][c]['input_pixel_sha256'] for c in report.CASES}
        doc['promotion']={p:report.legacy._gate(r,baseline['flip']) for p,r in doc['policies'].items()}
        return doc,baseline,pixels

    def origin(self):
        doc,baseline,pixels=self.doc()
        doc.update(baseline_sha256=None,checkpoint_sha256=report.BASELINE_CHECKPOINT,parameter_count=4298970,
                   checkpoint_epoch=151)
        doc['config'].update(depth=22,widen_factor=4,epochs=160)
        doc['model_config']={k:doc['config'][k] for k in ('depth','widen_factor','dropout','num_classes')}
        baseline.update(checkpoint_sha256=report.BASELINE_CHECKPOINT,benchmark_sha256=report.BASELINE_BENCHMARK)
        return doc,baseline,pixels

    def test_origin_keeps_real_null_reference_and_has_independent_full_audit(self):
        doc,baseline,pixels=self.origin();before=copy.deepcopy(doc)
        with patch.object(report.legacy,'_benchmark',side_effect=AssertionError('origin is not a candidate')):
            self.assertEqual(report._origin_baseline(doc,baseline,pixels,report.BASELINE_BENCHMARK),doc['policies'][report.POLICY])
        self.assertEqual(doc,before);self.assertIsNone(doc['baseline_sha256'])
        with self.assertRaises(ValueError):report._full_benchmark(doc,baseline,pixels)
        for kind in ('self_reference','digest','parameters','case','partial','pixel','context','gate','grid'):
            bad=copy.deepcopy(doc);digest=report.BASELINE_BENCHMARK
            if kind=='self_reference':bad['baseline_sha256']=report.BASELINE_BENCHMARK
            if kind=='digest':digest='a'*64
            if kind=='parameters':bad['parameter_count']=4166298
            if kind=='case':del bad['policies']['plain']['cases']['boxblur3']
            if kind=='partial':bad['policies']['plain']['cases']['noise_004']['examples']=4999
            if kind=='pixel':bad['policies']['plain']['cases']['noise_004']['input_pixel_sha256']='a'*64
            if kind=='context':bad['evaluation_context']['amp_dtype']='torch.float16'
            if kind=='gate':bad['promotion']['plain']['passed']=True
            if kind=='grid':
                row=bad['policies']['plain'];row['cases']['clean']['accuracy']=row['clean_accuracy']=.96001
                bad['promotion']['plain']=report.legacy._gate(row,baseline['flip'])
            with self.subTest(kind=kind),self.assertRaises(ValueError):report._origin_baseline(bad,baseline,pixels,digest)

    def test_SE_constructor_retains_and_checks_architecture_reduction(self):
        doc,baseline,pixels=self.doc();doc['config'].update(depth=22,widen_factor=4,epochs=160,
            architecture='se_wrn22_4_r16_v1',se_reduction=16,sam_rho=.05)
        doc.update(parameter_count=4332654,checkpoint_epoch=151,
            model_config={'architecture':'se_wrn22_4_r16_v1',**{k:doc['config'][k] for k in ('depth','widen_factor','dropout','num_classes')},'reduction':16},
            research_training_integrity={'training':{'schema':'research_se_sam_training_v1',
                'architecture':'se_wrn22_4_r16_v1','reduction':16,
                'source_sha256':{'research/train_se_sam.py':report._sha(report.ROOT/'research/train_se_sam.py')}}})
        self.assertEqual(report._full_benchmark(doc,baseline,pixels),doc['policies'][report.POLICY])
        for kind in ('architecture','reduction','constructor','marker','depth'):
            bad=copy.deepcopy(doc)
            if kind=='architecture':bad['config']['architecture']='ordinary_wrn'
            if kind=='reduction':bad['config']['se_reduction']=8
            if kind=='constructor':del bad['model_config']['architecture']
            if kind=='marker':bad['research_training_integrity']['training']['reduction']=8
            if kind=='depth':bad['config']['depth']=34;bad['model_config']['depth']=34
            with self.subTest(kind=kind),self.assertRaises(ValueError):report._full_benchmark(bad,baseline,pixels)

    def test_import_help_and_component_audit_are_stdlib_only(self):
        code="import sys;from scripts import campaign_report_evidence as p;p.main([]);assert not {'torch','numpy','PIL','nbclient','reportlab'}&set(sys.modules)"
        subprocess.check_output([sys.executable,'-c',code],cwd=report.ROOT,text=True)
        before=set(sys.modules);doc,baseline,pixels=self.doc();report._full_benchmark(doc,baseline,pixels)
        self.assertFalse({'torch','numpy','runtime','PIL'}&(set(sys.modules)-before))
        tree=ast.parse(Path(report.__file__).read_text())
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='load_followup' for n in ast.walk(tree)))

    def test_complete_raw_requires_all18_same_context_pixels_sources_and5000_integer_grid(self):
        doc,baseline,pixels=self.doc();report._full_benchmark(doc,baseline,pixels)
        changes=(('examples',4999),('training_examples',50000),('bn_recalibrated',True),('checkpoint_epoch',0),('checkpoint_variant','unbound'))
        for key,value in changes:
            with self.subTest(key=key),self.assertRaises(ValueError):report._full_benchmark({**doc,key:value},baseline,pixels)
        for kind in ('case','context','pixel','source','grid','seed'):
            bad=copy.deepcopy(doc)
            if kind=='case':bad['policies']['flip']['cases'].pop('resize24')
            if kind=='context':bad['evaluation_context']['amp_dtype']='torch.float16'
            if kind=='pixel':bad['policies']['plain']['cases']['clean']['input_pixel_sha256']='f'*64
            if kind=='source':bad['source']['source_sha256']['train.py']='f'*64
            if kind=='seed':bad['config']['seed']=2027
            if kind=='grid':
                row=bad['policies'][report.POLICY];row['cases']['clean']['accuracy']=row['clean_accuracy']=.96001
                bad['promotion'][report.POLICY]=report.legacy._gate(row,baseline['flip'])
            with self.subTest(kind=kind),self.assertRaises(ValueError):report._full_benchmark(bad,baseline,pixels)

    def test_record_hash_public_relative_refs_and_configuration_privacy(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'private.json';path.write_text('{"safe":1}')
            record={'path':str(path),'sha256':report._sha(path)}
            obj,public,_=report._read_record(record,path.parent)
            self.assertEqual(obj,{'safe':1});self.assertEqual(public,{'sha256':record['sha256']});self.assertNotIn(str(path),json.dumps(public))
            for ref in (str(path),'../private.json','https://private.example/token'):
                with self.assertRaises(ValueError):report._read_record({**record,'public_ref':ref},path.parent)
            path.write_text('{"changed":1}')
            with self.assertRaises(ValueError):report._read_record(record,path.parent)
        self.assertEqual(report._safe_config({'seed':2026,'unknown_secret':'DO_NOT_PUBLISH'}),{'seed':2026})
        with self.assertRaises(ValueError):report._safe_config({'lr':'DO_NOT_PUBLISH'})
        with self.assertRaises(ValueError):report._source_map({'results/private.pt':'a'*64})

    def test_real_public_legacy_projection_preserves_its_frozen_protocol(self):
        summary=report.legacy._json(report.ROOT/'docs/research-followup-results.json');baseline=summary['baseline']
        pixels={k:v['input_pixel_sha256'] for k,v in summary['wrn34_3_primary']['policies']['flip']['cases'].items()}
        rows=report._legacy_rows(summary,baseline,pixels,summary['WRN34_300_local_primary']['policies'][report.POLICY])
        self.assertEqual(len(rows),5);self.assertEqual({r['id'] for r in rows},{'wrn34_160','sam','resizemix','wrn34_300','swa40'})
        self.assertFalse(next(r for r in rows if r['id']=='swa40')['gate_passed'])
        self.assertTrue(next(r for r in rows if r['id']=='sam')['repeat_complete'])
        self.assertTrue(all(r['incumbent_comparison_scope'].startswith('retrospective') for r in rows))

    def pair_fixture(self):
        summary=report.legacy._json(report.ROOT/'docs/research-followup-results.json');baseline=summary['baseline']
        ta_public=report.legacy._json(report.ROOT/'docs/trivialaugment-completed-results.json')
        public=report.legacy._json(report.ROOT/'docs/trivialaugment-paired-results.json')
        ta_report={'clean_accuracy':ta_public['benchmark']['weighted_clean_accuracy'],'clean_loss':ta_public['benchmark']['weighted_clean_nll'],
           'stress_accuracy':ta_public['benchmark']['weighted_mean_stress_accuracy'],'cases':{c:{'accuracy':v['accuracy'],'loss':v['nll'],'examples':5000} for c,v in ta_public['benchmark']['weighted_cases'].items()}}
        incumbent=summary['WRN34_300_local_primary']['policies'][report.POLICY]
        raw_boots={}
        for kind,key in (('incumbent_bootstrap','paired_against_current_incumbent'),('legacy_bootstrap','paired_against_original_baseline')):
            b=public[key];raw_boots[kind]={k:b[k] for k in ('original_images','replicates','seed','sampling_unit','confidence_level')}
            for metric in ('clean','stress'):
                raw_boots[kind][metric]={k:v for k,v in b[metric].items() if k not in ('delta_percentage_points','ci95_percentage_points','ci_includes_zero')}
        raw={'schema':'research_trivialaugment_dual_image_bootstrap_v2','status':'complete','candidate_checkpoint_sha256':report.TA_CHECKPOINT,
           'candidate_benchmark_sha256':report.TA_BENCHMARK,'incumbent_checkpoint_sha256':report.INCUMBENT_CHECKPOINT,
           'incumbent_benchmark_sha256':report.INCUMBENT_BENCHMARK,'baseline_checkpoint_sha256':baseline['checkpoint_sha256'],
           'baseline_benchmark_sha256':baseline['benchmark_sha256'],'candidate_policy':report.POLICY,'incumbent_policy':report.POLICY,'baseline_policy':'flip',
           'data_signature':baseline['data_signature'],'evaluation_context':baseline['evaluation_context'],'shape':[5000,6],
           'original_ids_aligned':True,'raw_flags_persisted':False,**raw_boots}
        for k in ('original_id_order_sha256','development_index_order_sha256','candidate_correctness_sha256','incumbent_correctness_sha256','baseline_correctness_sha256'):raw[k]='a'*64
        frozen={'schema':'research_trivialaugment_primary_freeze_v2','status':'frozen','checkpoint_sha256':report.TA_CHECKPOINT,
           'benchmark_sha256':report.TA_BENCHMARK,'paired_sha256':report._digest(raw),'policy':report.POLICY,'planned_repeat_seed':2027,
           'planned_repeat_config_sha256':'9686ddea85d0f56ba1f58ff8668056c146a78b675f12f0624799f2fd0b2f4981',
           'frozen_at':public['primary_freeze']['frozen_at'],'planned_repeat_source_sha256':{'train.py':baseline['evaluation_context']['training_core_sha256']['train.py']}}
        hashes={'ta_pair':report._digest(raw),'ta_benchmark':report.TA_BENCHMARK,'ta_frozen':report._digest(frozen)}
        public['raw_evidence_sha256']['pair_result']=hashes['ta_pair'];public['raw_evidence_sha256']['primary_freeze']=hashes['ta_frozen']
        public['primary_freeze']['freeze_sha256']=hashes['ta_frozen']
        files={'ta_pair':raw,'ta_pair_summary':public,'ta_frozen':frozen}
        return files,hashes,{'policies':{report.POLICY:ta_report}},baseline,incumbent

    def test_dual_CI_references_are_not_interchanged_and_crossing_zero_is_preserved(self):
        files,hashes,ta,baseline,incumbent=self.pair_fixture();row={}
        report._ta_pair(files,hashes,ta,baseline,incumbent,row)
        self.assertEqual(row['clean_ci95'],[-.0006,.0088]);self.assertEqual(row['incumbent_clean_ci95'],[-.0028,.006])
        self.assertEqual(row['repeat_status'],'pending');self.assertEqual(row['delivery_status'],'awaiting_repeat')
        # The older completed snapshot says paired=false, but the later typed pair wins.
        self.assertFalse(files['ta_pair_summary']['decision']['independent_repeat_complete'])
        bad=copy.deepcopy(files);bad['ta_pair']['legacy_bootstrap'],bad['ta_pair']['incumbent_bootstrap']=bad['ta_pair']['incumbent_bootstrap'],bad['ta_pair']['legacy_bootstrap']
        with self.assertRaises(ValueError):report._ta_pair(bad,hashes,ta,baseline,incumbent,{})
        bad=copy.deepcopy(files);bad['ta_pair']['baseline_policy']=report.POLICY
        with self.assertRaises(ValueError):report._ta_pair(bad,hashes,ta,baseline,incumbent,{})

    def repeat_fixture(self,passed):
        doc,baseline,pixels=self.doc();config={**doc['config'],'epochs':160,'seed':2027}
        frozen={'frozen_at':'2026-11-03T00:07:37+00:00','planned_repeat_effective_config':config,'planned_repeat_config_sha256':'c'*64,
            'planned_repeat_source_sha256':{'train.py':doc['source']['source_sha256']['train.py']}}
        marker={'schema':'research_trivialaugment_training_v1','precision':'bf16','training_seed':2027,'ta_rng_seed':2027,
            'input_config_sha256':'c'*64,'effective_config_sha256':report._digest(config)}
        doc.update(config=config,research_training_integrity={'training':marker},point_and_case_gates_passed=passed)
        report_values=copy.deepcopy(doc['policies'][report.POLICY])
        if passed:
            report_values['clean_accuracy']=report_values['cases']['clean']['accuracy']=.97
            report_values['stress_accuracy']=.81
            for c in report.CASES[1:]:report_values['cases'][c]['accuracy']=.81
        else:
            report_values['clean_accuracy']=report_values['cases']['clean']['accuracy']=.94
        doc['policies'][report.POLICY]=report_values
        result={'schema':'research_trivialaugment_repeat_result_v1','status':'complete','role':'development_repeat','seed':2027,'epochs_completed':160,
            'config':config,'config_sha256':'c'*64,'checkpoint_sha256':doc['checkpoint_sha256'],'benchmark_sha256':'b'*64,'frozen_manifest_sha256':'f'*64,
            'source_sha256':frozen['planned_repeat_source_sha256'],'repeat_weights_hidden_use_authorized':False,'primary_replacement_authorized':False,
            'hidden_images_loaded':False,'confirmation_evaluated':False,'input_manifest_sha256':'d'*64,'successful_updates':50080,
            'skipped_updates':0,'source_examples_seen':6400000,'model_views_seen':6400000,'scalar_screen':{'scalar_screen_passed':passed}}
        receipt={'schema':'research_trivialaugment_repeat_supervisor_v1','status':'complete','mode':'formal','backend':'local','role':'development_repeat','seed':2027,
            'source_sha256':result['source_sha256'],'artifact':'repeat_result.json','artifact_sha256':'r'*64,'frozen_manifest_sha256':'f'*64,'input_manifest_sha256':'d'*64,
            'process_tree_cleanup_verified':True,'confirmation_evaluated':False,'started_at':'2026-11-03T00:08:53+00:00','finished_at':'2026-11-03T01:30:00+00:00',
            'process_tree_guard_results':[{'cleanup_verified':True,'completion_authenticated':True,'completion_exit_code':0,'timed_out':False}]}
        files={'ta_repeat_benchmark':doc,'ta_repeat_result':result,'ta_repeat_receipt':receipt,'ta_frozen':frozen,
               'ta_benchmark':{'config':{**config,'seed':2026}},'incumbent_benchmark':{'policies':{report.POLICY:copy.deepcopy(self.doc()[0]['policies'][report.POLICY])}}}
        return files,{'ta_repeat_benchmark':'b'*64,'ta_repeat_result':'r'*64,'ta_frozen':'f'*64},baseline,pixels,report_values

    def test_complete_failed_repeat_is_a_recorded_negative_not_pending_or_missing(self):
        for passed in (True,False):
            files,hashes,baseline,pixels,values=self.repeat_fixture(passed);row={'repeat_status':'pending'}
            with patch.object(report,'_full_benchmark',return_value=values):self.assertIs(report._ta_repeat(files,hashes,baseline,pixels,row),passed)
            self.assertTrue(row['repeat_complete']);self.assertIs(row['repeat_passed'],passed)
            self.assertEqual(row['repeat_status'],'complete_passed' if passed else 'complete_failed')
            if not passed:self.assertEqual(row['delivery_status'],'not_promoted')
        files,hashes,baseline,pixels,values=self.repeat_fixture(False);files['ta_repeat_receipt']['status']='training'
        with patch.object(report,'_full_benchmark',return_value=values),self.assertRaises(ValueError):report._ta_repeat(files,hashes,baseline,pixels,{})

    def receipt(self):
        return {'competition':report.legacy.COMPETITION,'team':'Saber','status':'complete','seed':2026,'training_examples':40000,
           'parameter_count':4166298,'policy':report.POLICY,'checkpoint_sha256':report.TA_CHECKPOINT,
           'delivery_schema':'research_trivialaugment_prediction_delivery_v1','run':'wrn34_trivialaugment_bf16_s2026',
           'repeat_receipt_schema':'research_trivialaugment_repeat_supervisor_v1','submission_ref':900001,'public_score':0.,'private_score':None,
           'submission_sha256':'a'*64,'manifest_sha256':'b'*64,'frozen_manifest_sha256':'c'*64,'repeat_receipt_sha256':'d'*64,
           'repeat_validator_source_sha256':'e'*64,'submitted_at':'2026-11-03T01:40:00+00:00','checked_at':'2026-11-03T01:41:00+00:00'}

    def test_real_zero_score_and_unknown_private_score_not_confused_with_missing(self):
        self.assertEqual(report._ta_receipt(self.receipt(),'f'*64)['public_score'],0.)
        self.assertIsNone(report._ta_receipt(self.receipt(),'f'*64)['private_score'])
        for key,value in (('public_score',None),('public_score',False),('status','scoring_pending'),('repeat_receipt_schema','old'),('run','stability_seed2027')):
            with self.assertRaises(ValueError):report._ta_receipt({**self.receipt(),key:value},'f'*64)

    def test_only_actual_canonical_directories_and_matched_repeat_validator_can_score(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(report,'ROOT',Path(temp).resolve()):
            root=Path(temp).resolve();canonical=root/'docs/research-submission-receipts';canonical.mkdir(parents=True)
            record=self.receipt();p=canonical/'900001.json';p.write_text(json.dumps(record))
            row={'id':'trivialaugment','label':'TA','checkpoint_sha256':report.TA_CHECKPOINT,'policy':report.POLICY,'parameters':4166298,'delivery_status':'prediction_complete_unscored'}
            files={k:{} for k in report.REPEAT|report.DELIVERY}
            files.update(ta_delivery={'submission_sha256':'a'*64,'repeat_validator_source_sha256':'e'*64},
                         ta_frozen={'planned_repeat_source_sha256':{'research/run_trivialaugment_repeat.py':'e'*64}})
            hashes={'ta_delivery':'b'*64,'ta_frozen':'c'*64,'ta_repeat_receipt':'d'*64}
            item={'path':str(p),'sha256':report._sha(p)}
            result,_=report._receipts([item,item],root,files,hashes,[row],[])
            self.assertEqual(len(result),1);self.assertEqual(row['public_score'],0.)
            other=root/'arbitrary/research-submission-receipts';other.mkdir(parents=True);q=other/p.name;q.write_bytes(p.read_bytes())
            with self.assertRaises(ValueError):report._receipts([{'path':str(q),'sha256':report._sha(q)}],root,files,hashes,[row],[])
            files['ta_delivery']['repeat_validator_source_sha256']='f'*64
            with self.assertRaises(ValueError):report._receipts([item],root,files,hashes,[row],[])

    def test_input_manifest_partial_groups_unknown_roles_and_changed_old_source_reject(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'input.json';files={k:{'path':'absent.json','sha256':'a'*64} for k in report.REQUIRED}
            manifest={'schema':report.INPUT_SCHEMA,'recorded_at':'2026-11-03T02:00:00+00:00','files':files,'scored_receipts':[]}
            for extra in ({'ta_repeat_result':files['legacy_summary']},{'ta_delivery':files['legacy_summary']},{'unknown':files['legacy_summary']}):
                p.write_text(json.dumps({**manifest,'files':{**files,**extra}}))
                with self.assertRaises(ValueError):report.load_campaign(p)
            p.write_text(json.dumps(manifest))
            with patch.object(report,'LEGACY_SOURCE_SHA','0'*64),self.assertRaises(ValueError):report.load_campaign(p)

    def test_no_arguments_do_not_read_or_create_report_artifacts(self):
        with patch.object(report,'load_campaign') as load,redirect_stdout(io.StringIO()):
            report.main([]);load.assert_not_called()


if __name__=='__main__':unittest.main()
