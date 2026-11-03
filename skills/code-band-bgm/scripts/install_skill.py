#!/usr/bin/env python3
"""为两个宿主建立指向同一技能源码的项目级入口。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


NAME = "code-band-bgm"
LOCATIONS = {"codex": ".agents/skills", "claude": ".claude/skills"}


def inspect_entry(host, project, source):
    entry = project / LOCATIONS[host] / NAME
    same = entry.is_symlink() and entry.resolve() == source
    return {
        "host": host,
        "entry": str(entry),
        "state": "linked" if same else (
            "conflict" if os.path.lexists(entry) else "absent"
        ),
        "source_matches": same,
    }


def operate(action, host, project, source):
    project = Path(project).expanduser().resolve()
    source = Path(source).expanduser().resolve()
    if not project.is_dir():
        raise ValueError("项目目录不存在。")
    if action == "install":
        manifest = source / "SKILL.md"
        if not manifest.is_file():
            raise ValueError("技能源码缺少 SKILL.md，不能安装。")
        text = manifest.read_text(encoding="utf-8")
        frontmatter = text.split("---", 2)
        if (not text.startswith("---\n") or len(frontmatter) != 3
                or "name: code-band-bgm" not in frontmatter[1].splitlines()):
            raise ValueError("SKILL.md 缺少预期的技能名称。")
    hosts = list(LOCATIONS) if host == "both" else [host]
    states = [inspect_entry(item, project, source) for item in hosts]
    if action != "status":
        for state in states:
            entry = Path(state["entry"])
            if state["state"] == "conflict":
                raise ValueError("同名技能冲突，未覆盖：" + str(entry))
            if not entry.parent.resolve().is_relative_to(project):
                raise ValueError("宿主入口父目录通过链接指向项目外，未修改：" + str(entry.parent))
    changed = []
    if action == "install":
        try:
            for state in states:
                if state["state"] == "absent":
                    entry = Path(state["entry"])
                    entry.parent.mkdir(parents=True, exist_ok=True)
                    entry.symlink_to(os.path.relpath(source, entry.parent), target_is_directory=True)
                    changed.append(entry)
        except OSError:
            for entry in changed:
                entry.unlink()
            raise
    elif action == "remove":
        for state in states:
            if state["state"] == "linked":
                entry = Path(state["entry"])
                entry.unlink()
                changed.append(entry)
    manifest = source / "SKILL.md"
    return {
        "status": "ok",
        "action": action,
        "source": str(source),
        "skill_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None,
        "changed_entries": [str(path) for path in changed],
        "entries": [inspect_entry(item, project, source) for item in hosts],
        "note": "仅验证文件入口；宿主发现、触发和音乐交付需另行实测。",
    }


def main():
    parser = argparse.ArgumentParser(description="代码乐团技能的可逆项目级安装与检查。")
    parser.add_argument("action", choices=["install", "status", "remove"], help="安装、检查或移除入口")
    parser.add_argument("--host", choices=["codex", "claude", "both"], default="both", help="目标宿主")
    parser.add_argument("--project", default=".", help="目标项目目录")
    parser.add_argument("--source", default=str(Path(__file__).resolve().parents[1]), help="权威技能源码目录")
    args = parser.parse_args()
    try:
        result = operate(args.action, args.host, args.project, args.source)
        code = 0
    except (OSError, ValueError) as exc:
        result, code = {"status": "error", "message": str(exc)}, 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
