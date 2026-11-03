"""真实原生包络控制、冻结门禁与旧事件兼容；听感单列。"""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'skills/code-band-bgm/scripts'))
from sound_shaping import add_shaping_capability, validate_shaping
from fluid_engine import render_events, EngineError
from score_tools import ScoreError, compile_score, load, validate_frozen
from soundfont_inventory import inventory


class ShapingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.font = ROOT/'skills/code-band-bgm/assets/soundfonts/MuseScore_General.sf3'
        cls.score = load(ROOT/'tests/fixtures/音频闭环乐谱.json')
        cls.cap = load(ROOT/'tests/fixtures/engine-capabilities-v1.json')
        cls.mapping = inventory(cls.font)
        cls.sha = cls.mapping['source']['sha256']
        cls.result = add_shaping_capability({'capabilities':copy.deepcopy(cls.cap)}, cls.font)
        cls.shaped_cap = cls.result['capabilities']

    def test_custom_font_without_representative_does_not_break_basic_doctor(self):
        with patch("soundfont_inventory.inventory", return_value={"presets":[]}), patch("fluid_engine.render_events") as renderer:
            result=add_shaping_capability({"capabilities":copy.deepcopy(self.cap)},self.font)
        renderer.assert_not_called()
        self.assertEqual(result["capabilities"]["performance"]["sound_shaping"]["state"],"unknown")

    def test_actual_probe(self):
        self.assertTrue(self.result['sound_shaping_probe']['changed_audio'])
        self.assertTrue(self.result['sound_shaping_probe']['float_repeat_equal'])
        self.assertEqual(self.result['sound_shaping_probe']['accepted_offsets']['amp_attack_offset_timecents'], -3600)

    def test_tail_reduction_at_actual_pad_key(self):
        doc={'sample_rate_hz':48000,'total_frames':144000,'parts':[{'part_id':'pad','bank':0,'program':89,'gain_db':0,'pan':0}],
             'events':[{'event_id':'n','part_id':'pad','onset_frame':0,'gate_frames':20480,'midi':60,'velocity':90}],
             'mix':{'master_gain_db':0,'effects':[]},'render':{'loop':{'enabled':False}}}
        original,_=render_events(doc,self.font)
        doc['parts'][0]['sound_shaping']={'amp_attack_offset_timecents':-3600,'amp_release_offset_timecents':-2400}
        shaped,_=render_events(doc,self.font)
        fraction=lambda w:np.sum(w[20480:]**2)/np.sum(w**2)
        self.assertLess(fraction(shaped), fraction(original)*.5)
        self.assertEqual(len(shaped),len(original))

    def test_unsupported_parameters_do_not_silently_pass(self):
        for value in [{'amp_attack_offset_timecents':float('nan')}, {'unknown':2}, {'amp_release_offset_timecents':1}, {'filter_cutoff_offset_cents':-1201}, {'amp_attack_offset_timecents':True}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_shaping(value)

    def test_compile_and_frozen_require_capability_and_preserve_offsets(self):
        score=copy.deepcopy(self.score)
        score['parts'][0]['sound_shaping']={'amp_attack_offset_timecents':-3600}
        with self.assertRaises(ScoreError):
            compile_score(score,self.cap,self.sha,self.mapping)
        frozen=compile_score(score,self.shaped_cap,self.sha,self.mapping)
        self.assertEqual(frozen['parts'][0]['sound_shaping'],score['parts'][0]['sound_shaping'])
        self.assertIs(validate_frozen(frozen,score,self.shaped_cap,self.sha),frozen)
        with self.assertRaises(ScoreError):
            validate_frozen(frozen,score,self.cap,self.sha)
        frozen['parts'][0]['sound_shaping']['amp_attack_offset_timecents']=-1200
        with self.assertRaises(ScoreError):
            validate_frozen(frozen,score,self.shaped_cap,self.sha)


if __name__ == '__main__':
    unittest.main()
