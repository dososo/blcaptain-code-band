"""分离入口的实际CLI失败路径；不加载可选模型。"""
from pathlib import Path
import json,subprocess,sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1]

class SeparationCLITests(unittest.TestCase):
 def test_help_exposes_explicit_model_and_isolation_without_claiming_identity(self):
  p=subprocess.run([sys.executable,str(ROOT/'skills/code-band-bgm/scripts/band.py'),'separate-reference','--help'],capture_output=True,text=True)
  self.assertEqual(p.returncode,0,p.stderr)
  for text in ['--model-config','--backend-python','--start','--seconds']:self.assertIn(text,p.stdout)
 def test_missing_assets_return_json_failure_and_no_fake_output(self):
  out=Path(tempfile.mkdtemp(prefix='band-split-cli-'))/'不应生成'
  p=subprocess.run([sys.executable,str(ROOT/'skills/code-band-bgm/scripts/band.py'),'separate-reference','--input','缺少音频','--model-config','缺少模型配置','--backend-python',sys.executable,'--out',str(out)],capture_output=True,text=True)
  self.assertEqual(p.returncode,2);self.assertEqual(json.loads(p.stdout)['status'],'失败');self.assertFalse(out.exists())
