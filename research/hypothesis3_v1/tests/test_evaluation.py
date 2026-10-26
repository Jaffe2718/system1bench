"""Evaluation contracts, alias weights and corruption/failure isolation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import build
import evaluate
import test_bank


def fixture(root):
    bank=root/'bank';bank.mkdir()
    item=test_bank.fixture_items()[1]
    paired,_=build.single_records([item],'paired',{item['base_id']:'confirmation'})
    other=test_bank.fixture_items()[2]
    other['gold_indices']=[0,2]
    multi=build.multi_records([other],{other['base_id']:'development'})
    files={}
    for panel,rows in [('native',[]),('paired',paired),('multi_select',multi)]:
        path=bank/(panel+'.jsonl')
        path.write_bytes(b''.join(build.encode(r)+b'\n' for r in rows))
        files[path.name]=dict(sha256=build.file_sha(path))
    build.dump(bank/'manifest.json',dict(files=files))
    result=root/'results';result.mkdir()
    build.dump(result/'run_status.json',dict(profile='full',status='OK',models={'english':{'status':'OK'}}))
    return bank,result,paired+multi


def save_model(result,bank,rows,model='english',invalid=None):
    output=result/model;output.mkdir(exist_ok=True)
    predictions=[]
    for row in rows:
        answers={}
        for qid,q in row['questions'].items():
            if q['type']=='choice':
                # Deliberately vary success by design condition, not unique row count.
                label=row['gold_ids'][0] if row['conditions'][0]['order']==0 else next(k for k in q['criteria'] if k not in row['gold_ids'])
                answers[qid]=dict(choice=label,probabilities={k:float(k==label) for k in q['criteria']})
            else:
                answers[qid]=dict(noul=.9)  # all-selected control
        failed=row['id']==invalid
        p=test_bank.runner.format_prediction(row,dict(answers=answers),
            {q:dict(complete=not failed) for q in row['questions']},
            'FixtureFailure' if failed else None,.01,lambda *args:None)
        predictions.append(p)
    journal=output/'predictions.jsonl'
    journal.write_bytes(b''.join(build.encode(p)+b'\n' for p in predictions))
    build.dump(output/'metadata.json',dict(status='DONE',predictions_sha256=build.file_sha(journal),
        signature=dict(bank_manifest_sha256=build.file_sha(bank/'manifest.json'),profile='full')))
    return predictions


class EvaluationTests(unittest.TestCase):
    def test_alias_weighting_matches_hand_count_and_multiselect_baselines(self):
        with tempfile.TemporaryDirectory(dir=build.ROOT) as tmp:
            root=Path(tmp);bank,result,rows=fixture(root)
            predictions=save_model(result,bank,rows)
            summary=evaluate.evaluate(result,root/'evaluation',bank,repetitions=100)
            byid={p['id']:p for p in predictions}
            for row in summary['single_choice']:
                if row['source']!='ALL' or row['split']!='ALL':
                    continue
                cases=[r for r in rows if r['panel']=='paired' and r['k']==row['k']]
                good=sum(len(r['conditions'])*(byid[r['id']]['predicted_ids']==r['gold_ids']) for r in cases)
                total=sum(len(r['conditions']) for r in cases)
                self.assertEqual(row['design_conditions'],4)
                self.assertAlmostEqual(row['accuracy'],good/total)
                k=row['k']
                self.assertAlmostEqual(row['chance_adjusted_score'],100*(good/total-1/k)/(1-1/k))
            multi=next(r for r in summary['multi_select'] if r['split']=='ALL' and r['k']=='ALL')
            self.assertEqual(multi['exact_match'],0.)
            self.assertEqual(multi['all_selected_rate'],1.)
            self.assertAlmostEqual(multi['macro_set_f1'],4/7)
            self.assertAlmostEqual(multi['all_selected_baseline_f1'],4/7)
            self.assertAlmostEqual(multi['fair_coin_exact_match_baseline'],1/32)
            self.assertAlmostEqual(multi['fair_coin_adjusted_em_score'],-100/31)

    def test_recorded_failure_counts_wrong_without_reducing_denominator(self):
        with tempfile.TemporaryDirectory(dir=build.ROOT) as tmp:
            root=Path(tmp);bank,result,rows=fixture(root)
            save_model(result,bank,rows,invalid=rows[0]['id'])
            summary=evaluate.evaluate(result,root/'evaluation',bank,repetitions=100)
            coverage=next(r for r in summary['coverage'] if r['model']=='english')
            self.assertEqual(coverage['status'],'COMPLETE_WITH_INVALID')
            self.assertEqual(coverage['invalid_requests'],1)
            row=next(r for r in summary['single_choice'] if r['source']=='ALL' and r['split']=='ALL' and r['k']==2)
            self.assertEqual(row['design_conditions'],4)
            self.assertLess(row['valid_condition_fraction'],1.)

    def test_missing_models_unscored_and_additional_model_discovered(self):
        with tempfile.TemporaryDirectory(dir=build.ROOT) as tmp:
            root=Path(tmp);bank,result,rows=fixture(root)
            save_model(result,bank,rows,'additional_model')
            summary=evaluate.evaluate(result,root/'evaluation',bank,repetitions=100)
            coverage={r['model']:r['status'] for r in summary['coverage']}
            self.assertEqual(coverage['english'],'NOT_RUN')
            self.assertEqual(coverage['additional_model'],'COMPLETE')
            self.assertEqual(coverage['jev-1.13.0'],'SKIPPED_API')
            self.assertEqual(coverage['nanojev'],'NOT_RUN')
            self.assertEqual({r['model'] for r in summary['single_choice']},{'additional_model'})

    def test_corrupt_journal_does_not_block_valid_peer(self):
        with tempfile.TemporaryDirectory(dir=build.ROOT) as tmp:
            root=Path(tmp);bank,result,rows=fixture(root)
            save_model(result,bank,rows,'english')
            save_model(result,bank,rows,'multilingual')
            with (result/'english/predictions.jsonl').open('ab') as f:f.write(b'\n')
            summary=evaluate.evaluate(result,root/'evaluation',bank,repetitions=100)
            coverage={r['model']:r['status'] for r in summary['coverage']}
            self.assertEqual(coverage['english'],'UNEVALUABLE')
            self.assertEqual(coverage['multilingual'],'COMPLETE')
            self.assertFalse(any(r['model']=='english' for r in summary['single_choice']))

    def test_missing_request_with_updated_hash_still_rejected(self):
        with tempfile.TemporaryDirectory(dir=build.ROOT) as tmp:
            root=Path(tmp);bank,result,rows=fixture(root)
            save_model(result,bank,rows)
            journal=result/'english/predictions.jsonl'
            journal.write_bytes(b'\n'.join(journal.read_bytes().splitlines()[1:])+b'\n')
            meta=json.loads((result/'english/metadata.json').read_text())
            meta['predictions_sha256']=build.file_sha(journal)
            build.dump(result/'english/metadata.json',meta)
            summary=evaluate.evaluate(result,root/'evaluation',bank,repetitions=100)
            self.assertFalse(summary['single_choice'])
            self.assertEqual(next(r for r in summary['coverage'] if r['model']=='english')['status'],'UNEVALUABLE')

    def test_directory_zip_equivalence_and_determinism(self):
        with tempfile.TemporaryDirectory(dir=build.ROOT) as tmp:
            root=Path(tmp);bank,result,rows=fixture(root)
            save_model(result,bank,rows)
            archive=root/'result.zip'
            with zipfile.ZipFile(archive,'w') as z:
                for file in result.rglob('*'):
                    if file.is_file():z.write(file,file.relative_to(result).as_posix())
                for file in bank.iterdir():z.write(file,'question_bank/'+file.name)
            a=evaluate.evaluate(result,root/'first',bank,repetitions=100)
            b=evaluate.evaluate(archive,root/'second',repetitions=100)
            self.assertEqual(a,b)
            for name in ('summary.json','protocol.json','paired_scores.csv','REPORT.md'):
                self.assertEqual((root/'first'/name).read_bytes(),(root/'second'/name).read_bytes())

    def test_cluster_bootstrap_keeps_columns_paired(self):
        values=np.array([[0.,0.],[1.,1.],[.5,.5]])
        draws=evaluate.bootstrap(values,[('s','same'),('s','same'),('s','other')],100,7)
        self.assertTrue(np.array_equal(draws[:,0],draws[:,1]))
        self.assertTrue(np.all(draws==.5))  # both cluster means are .5


if __name__=='__main__':
    unittest.main()
