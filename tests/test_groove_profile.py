"""表现统计只定向改演奏，不能换主题/补鼓或破坏冻结复现。"""
import copy
from pathlib import Path
import sys
import unittest
import tempfile
import subprocess
import json

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'skills/code-band-bgm/scripts'))
from score_tools import load, ScoreError
from groove_profile import apply_profile


class GrooveProfileTests(unittest.TestCase):
    def setUp(self):
        self.score=load(ROOT/'tests/fixtures/平衡修正版采样谱.json')
        self.profile=load(ROOT/'skills/code-band-bgm/assets/groove-profiles/gmd_hiphop110_closed_hat.json')

    def test_targeted_repeatable_without_changing_pitch_gate_or_input(self):
        before=copy.deepcopy(self.score)
        result,report=apply_profile(self.score,self.profile,'hats')
        self.assertEqual(self.score,before)
        self.assertEqual((result,report),apply_profile(before,self.profile,'hats'))
        self.assertEqual(result['randomness'],before['randomness'])
        for a,b in zip(before['parts'],result['parts']):
            if a['part_id']!='hats':self.assertEqual(a,b)
            for x,y in zip(a['notes'],b['notes']):
                for field in ('note_id','pitch','duration_ticks'):self.assertEqual(x[field],y[field])
        self.assertEqual(report['eligible_note_count'],40)
        self.assertEqual(report['unmatched_note_count'],16)

    def test_unknown_phases_and_section_starts_preserved(self):
        result,report=apply_profile(self.score,self.profile,'hats')
        a=next(p for p in self.score['parts']if p['part_id']=='hats')['notes']
        b=next(p for p in result['parts']if p['part_id']=='hats')['notes']
        self.assertEqual(a[0]['onset_tick'],b[0]['onset_tick'])
        for x,y in zip(a,b):
            if (x['onset_tick']//240)%16 in (7,13):self.assertEqual(x,y)
        self.assertTrue(all(abs(x['velocity_delta'])<=8 and abs(x['compiled_onset_delta_frames'])<=256 for x in report['changes']))

    def test_bad_profile_and_timeline_fail(self):
        for mutate in [lambda p:p.update(hit_key='kick'),lambda p:p['phases'][0].update(offset_step_fraction=float('nan')),
                       lambda p:p['phases'].append(p['phases'][0]),lambda p:p.update(min_observations=0)]:
            profile=copy.deepcopy(self.profile);mutate(profile)
            with self.assertRaises(ScoreError):apply_profile(self.score,profile,'hats')
        score=copy.deepcopy(self.score);score['timeline']['tempo_map'].append({'start_tick':15360,'bpm':{'numerator':110,'denominator':1}})
        with self.assertRaises(ScoreError):apply_profile(score,self.profile,'hats')
        with self.assertRaises(ScoreError):apply_profile(self.score,self.profile,'missing')

    def test_cli_writes_evidence_and_refuses_overwrite_or_bad_part(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'新谱 ; 无需执行.json'
            args=[sys.executable,str(ROOT/'skills/code-band-bgm/scripts/band.py'),'express','--score',
                  str(ROOT/'tests/fixtures/平衡修正版采样谱.json'),'--part','hats','--out',str(out)]
            first=subprocess.run(args,capture_output=True,text=True)
            self.assertEqual(first.returncode,0,first.stdout+first.stderr)
            self.assertEqual(json.loads(first.stdout)['eligible_note_count'],40)
            evidence=load(str(out)+'.change.json')
            self.assertEqual(evidence['provenance']['license'],'CC-BY-4.0')
            before=out.read_bytes()
            again=subprocess.run(args,capture_output=True,text=True)
            self.assertEqual(again.returncode,2)
            self.assertEqual(before,out.read_bytes())
            args[args.index('--part')+1]='不存在'
            args[args.index('--out')+1]=str(Path(folder)/'不得生成.json')
            failure=subprocess.run(args,capture_output=True,text=True)
            self.assertEqual(failure.returncode,2)
            self.assertFalse(Path(args[-1]).exists())


if __name__=='__main__':unittest.main()
