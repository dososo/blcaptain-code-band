"""维护者用：按音色注册表生成 soundfonts.lock.json（资产名、仓库内路径、字节数、SHA-256、许可）。

用法（仓库根目录，音色文件已在本地）：
    python3 scripts/build_soundfont_lock.py --base-url https://github.com/<owner>/<repo>/releases/download/<tag> --out soundfonts.lock.json
目标文件已存在则拒绝覆盖。
"""
import argparse
import json
from pathlib import Path

from fetch_soundfonts import ROOT, SCHEMA_VERSION, sha256

SKILL_ASSETS = ROOT / "skills/code-band-bgm/assets"
REGISTRY = SKILL_ASSETS / "soundfont-packs.json"
DEFAULT_FONT = ("soundfonts/MuseScore_General.sf3", "musescore_general_default", "MIT")


def main():
    parser = argparse.ArgumentParser(description="生成音色下载清单")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖：{out}")
    entries = {DEFAULT_FONT[0]: {"pack_ids": [DEFAULT_FONT[1]], "license_ids": [DEFAULT_FONT[2]]}}
    for pack in json.loads(REGISTRY.read_text(encoding="utf-8"))["packs"]:
        entry = entries.setdefault(pack["path"], {"pack_ids": [], "license_ids": []})
        entry["pack_ids"].append(pack["pack_id"])
        if pack["license_id"] not in entry["license_ids"]:
            entry["license_ids"].append(pack["license_id"])
    files, names = [], set()
    for rel, entry in sorted(entries.items()):
        path = SKILL_ASSETS / rel
        if path.name in names:
            raise SystemExit(f"资产名重复：{path.name}")
        names.add(path.name)
        files.append({"asset": path.name, "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                      "sha256": sha256(path), "pack_ids": entry["pack_ids"], "license_ids": entry["license_ids"]})
    lock = {"schema_version": SCHEMA_VERSION, "base_url": args.base_url.rstrip("/"), "files": files}
    out.write_text(json.dumps(lock, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(files)} 个文件，{sum(f['bytes'] for f in files) / 1e9:.2f} GB -> {out}")


if __name__ == "__main__":
    main()
