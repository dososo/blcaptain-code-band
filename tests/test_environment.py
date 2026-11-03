"""缺依赖和未登记音色必须给出真实失败，不能借用已有许可。"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/code-band-bgm/scripts"))
import band
from score_tools import ScoreError


class EnvironmentTests(unittest.TestCase):
    def test_missing_dependency_is_machine_readable(self):
        r = subprocess.run([sys.executable, "-S", str(ROOT / "skills/code-band-bgm/scripts/band.py"), "doctor"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertEqual(json.loads(r.stdout)["code"], "dependency_missing")
        self.assertNotIn("Traceback", r.stderr)

    def test_unregistered_soundfont_rejected_before_engine_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "unregistered.sf3"; p.write_bytes(b"not a registered bank")
            with patch.object(band, "probe", side_effect=AssertionError("禁止加载未登记音色")):
                with self.assertRaises(ScoreError) as exc: band.capabilities(p)
        self.assertEqual(exc.exception.details["field"], "soundfont")
