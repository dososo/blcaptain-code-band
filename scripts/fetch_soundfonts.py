"""下载代码乐团的音色文件（SF2/SF3），逐个校验 SHA-256。

仓库只放代码和小型数据；音色文件附在 GitHub Release 里，清单见 soundfonts.lock.json。
只用 Python 标准库。用法（仓库根目录）：

    python3 scripts/fetch_soundfonts.py              # 下载全部（约 3 GB）
    python3 scripts/fetch_soundfonts.py --list       # 只列清单，不下载
    python3 scripts/fetch_soundfonts.py --only MuseScore_General.sf3 guqin_songest_v2.sf2
    python3 scripts/fetch_soundfonts.py --verify     # 只校验本地已有文件

规则：
- 只接受 https 地址；目标路径必须落在仓库内（解析软链接后检查）。
- 先写到同目录的 .part 临时文件，摘要和字节数都对上才原子改名；对不上就删除临时文件并报错。
- 本地已有且摘要正确的文件直接跳过；摘要不对的旧文件不覆盖，报错让使用者自己处理。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "soundfonts.lock.json"
SCHEMA_VERSION = "1.0.0"
CHUNK = 1 << 20


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def load_lock(path):
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit(f"清单版本不支持：{lock.get('schema_version')}")
    if not str(lock.get("base_url", "")).startswith("https://"):
        raise SystemExit("清单里的 base_url 必须是 https 地址")
    for item in lock["files"]:
        if set(item) - {"asset", "path", "bytes", "sha256", "pack_ids", "license_ids"}:
            raise SystemExit(f"清单条目有未知字段：{item.get('asset')}")
        if "/" in item["asset"] or item["asset"].startswith("."):
            raise SystemExit(f"资产名不合法：{item['asset']}")
    return lock


def target_path(item):
    target = (ROOT / item["path"]).resolve()
    if ROOT.resolve() not in target.parents:
        raise SystemExit(f"目标路径越出仓库：{item['path']}")
    return target


def download(url, part, expected_bytes):
    request = urllib.request.Request(url, headers={"User-Agent": "code-band-fetch/1"})
    with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as out:
        if not response.geturl().startswith("https://"):
            raise RuntimeError("下载被重定向到非 https 地址")
        done = 0
        for block in iter(lambda: response.read(CHUNK), b""):
            done += len(block)
            if done > expected_bytes:
                raise RuntimeError("下载内容比清单记录的大")
            out.write(block)


def main():
    parser = argparse.ArgumentParser(description="下载并校验代码乐团音色文件")
    parser.add_argument("--lock", default=str(LOCK), help="清单路径，默认仓库根目录的 soundfonts.lock.json")
    parser.add_argument("--only", nargs="+", metavar="ASSET", help="只处理这些资产名")
    parser.add_argument("--list", action="store_true", help="只列清单")
    parser.add_argument("--verify", action="store_true", help="只校验本地文件，不下载")
    args = parser.parse_args()
    lock = load_lock(Path(args.lock))
    items = lock["files"]
    if args.only:
        unknown = set(args.only) - {i["asset"] for i in items}
        if unknown:
            raise SystemExit(f"清单里没有：{', '.join(sorted(unknown))}")
        items = [i for i in items if i["asset"] in args.only]
    if args.list:
        for i in items:
            print(f"{i['bytes'] / 1e6:9.1f} MB  {i['asset']}  ->  {i['path']}  [{', '.join(i['license_ids'])}]")
        print(f"共 {len(items)} 个，{sum(i['bytes'] for i in items) / 1e9:.2f} GB")
        return 0
    failed = []
    for i in items:
        target = target_path(i)
        if target.exists():
            ok = target.stat().st_size == i["bytes"] and sha256(target) == i["sha256"]
            print(("已有，校验通过  " if ok else "已有，校验不符  ") + i["asset"])
            if not ok:
                failed.append(i["asset"])
            continue
        if args.verify:
            print("缺失            " + i["asset"])
            failed.append(i["asset"])
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        try:
            print(f"下载 {i['bytes'] / 1e6:.1f} MB  {i['asset']}", flush=True)
            download(lock["base_url"].rstrip("/") + "/" + i["asset"], part, i["bytes"])
            if part.stat().st_size != i["bytes"] or sha256(part) != i["sha256"]:
                raise RuntimeError("字节数或 SHA-256 与清单不符")
            os.replace(part, target)
        except Exception as exc:
            part.unlink(missing_ok=True)
            print(f"失败：{i['asset']}：{exc}", file=sys.stderr)
            failed.append(i["asset"])
    if failed:
        print(f"{len(failed)} 个文件未通过：{', '.join(failed)}", file=sys.stderr)
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
