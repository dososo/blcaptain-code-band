"""本地语义后端的资产门禁，不加载可选大模型。"""
from pathlib import Path
import hashlib,importlib.util,json,sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'skills/code-band-bgm/scripts'))

class AssetTests(unittest.TestCase):
 def api(self):
  self.assertIsNotNone(importlib.util.find_spec('moss_review_backend'),'缺少独立本地后端')
  import moss_review_backend
  return moss_review_backend
 def model(self):
  root=Path(tempfile.mkdtemp(prefix='band-model-lock-'))
  (root/'model.safetensors').write_bytes('仅用于资产门禁测试，不是模型'.encode())
  (root/'config.json').write_text('{}');(root/'tokenizer.json').write_text('{}')
  return root
 def test_all_non_weight_files_are_locked_and_changed_tokenizer_fails(self):
  m=self.api();root=self.model();locked=m.model_assets(root)
  self.assertEqual(set(locked),{'model.safetensors','config.json','tokenizer.json'})
  m.verify_assets(root,locked)
  (root/'tokenizer.json').write_text('{"改变":true}')
  with self.assertRaises(ValueError):m.verify_assets(root,locked)
 def test_added_python_and_symlink_directory_are_rejected(self):
  m=self.api()
  for kind in ['python','symlink']:
   root=self.model()
   if kind=='python':(root/'remote.py').write_text('raise RuntimeError()')
   else:(root/'逃逸').symlink_to(root.parent,target_is_directory=True)
   with self.subTest(kind=kind),self.assertRaises(ValueError):m.model_assets(root)
 def test_added_or_removed_model_asset_is_rejected(self):
  m=self.api();root=self.model();locked=m.model_assets(root)
  (root/'额外.txt').write_text('不同的资产集合')
  with self.assertRaises(ValueError):m.verify_assets(root,locked)
 def test_standard_slow_tokenizer_does_not_require_fast_tokenizer_json(self):
  m=self.api();root=self.model();(root/'tokenizer.json').unlink()
  (root/'vocab.json').write_text('{}');(root/'merges.txt').write_text('')
  self.assertIn('vocab.json',m.model_assets(root))
 def test_lock_paths_cannot_escape_model_directory(self):
  m=self.api();root=self.model();locked=m.model_assets(root);locked['../旁路']='a'*64
  with self.assertRaises(ValueError):m.verify_assets(root,locked)
 def test_existing_config_is_never_overwritten(self):
  m=self.api();root=self.model();out=root.parent/(root.name+'配置.json');out.write_text('保留')
  with self.assertRaises(ValueError):m.register(root,'a'*64,out)
  self.assertEqual(out.read_text(),'保留')
 def test_import_keeps_optional_model_runtime_unloaded(self):
  self.api();self.assertNotIn('mlx.core',sys.modules)
