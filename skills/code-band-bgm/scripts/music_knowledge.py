"""按显式主题选取本地音乐方法；不推断原器、不评分音乐、不联网。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


LIBRARY = Path(__file__).resolve().parents[1] / "assets/knowledge/音乐方法库.json"


def load_library(path=LIBRARY):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("schema_version") != "1.0.0":
        raise ValueError("不支持的音乐方法库版本")
    for field in ("topics", "sources", "rules"):
        rows = data[field]
        if not rows or len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"方法库{field}为空或ID重复")
    topics = {row["id"] for row in data["topics"]}
    sources = {row["id"] for row in data["sources"]}
    for rule in data["rules"]:
        if not rule["topics"] or not set(rule["topics"]) <= topics:
            raise ValueError(f"规则{rule['id']}引用了未知主题")
        if not rule["source_ids"] or not set(rule["source_ids"]) <= sources:
            raise ValueError(f"规则{rule['id']}缺少有效来源")
        for field in ("title", "use_when", "action", "counterexample", "project_application", "verification", "limits"):
            if not rule.get(field):
                raise ValueError(f"规则{rule['id']}缺少{field}")
    return data, hashlib.sha256(raw).hexdigest()


def query_knowledge(topics=None, rule_ids=None, path=LIBRARY):
    """多个主题或规则取并集；未指定时只返回索引，不倾倒全部资料。"""
    data, sha = load_library(path)
    topics, rule_ids = set(topics or []), set(rule_ids or [])
    unknown_topics = topics - {r["id"] for r in data["topics"]}
    unknown_rules = rule_ids - {r["id"] for r in data["rules"]}
    if unknown_topics or unknown_rules:
        raise ValueError("未知主题或规则：" + "、".join(sorted(unknown_topics | unknown_rules)))
    result = {
        "status": "通过", "status_scope": "knowledge_selection",
        "library_version": data["version"], "library_sha256": sha,
        "updated_at": data["updated_at"],
        "note": "检索成功仅表示选出资料；不代表音源已安装、模型已运行、原器已识别或音乐已验收。",
    }
    if not topics and not rule_ids:
        result.update(
            topics=[{**row, "rule_count": sum(row["id"] in r["topics"] for r in data["rules"])}
                    for row in data["topics"]],
            rules=[{k: r[k] for k in ("id", "title", "topics")} for r in data["rules"]],
            source_count=len(data["sources"]), rule_count=len(data["rules"]),
        )
        return result
    selected = [r for r in data["rules"] if set(r["topics"]) & topics or r["id"] in rule_ids]
    required = {s for r in selected for s in r["source_ids"]}
    result.update(rules=selected, sources=[r for r in data["sources"] if r["id"] in required],
                  rule_count=len(selected), source_count=len(required))
    return result
