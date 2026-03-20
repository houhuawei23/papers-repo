#!/usr/bin/env python3
"""
papers-repo 仓库分析与报告脚本。

扫描 papers/ 目录，收集论文元数据、校验质量、检测异常与重复，
输出 Markdown、JSON、CSV 三种报告。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


@dataclass
class PaperEntry:
    """单篇论文条目。"""

    dir_path: str
    dir_name: str
    paper_yml_path: str | None
    paper_yml: dict[str, Any] | None
    title: str | None = None
    year: int | None = None
    venue: str | None = None
    arxiv_id: str | None = None
    tags: list[str] = field(default_factory=list)
    has_notes: bool = False
    has_trans: bool = False
    md_count: int = 0
    trans_count: int = 0
    parse_errors: list[str] = field(default_factory=list)
    naming_parsed: dict[str, str] | None = None
    naming_issues: list[str] = field(default_factory=list)
    is_hidden: bool = False


@dataclass
class RepoStats:
    """仓库统计。"""

    total_dirs: int = 0
    with_paper_yml: int = 0
    without_paper_yml: int = 0
    file_type_counts: dict[str, int] = field(default_factory=dict)
    venue_distribution: dict[str, int] = field(default_factory=dict)
    year_distribution: dict[int, int] = field(default_factory=dict)
    tag_distribution: dict[str, int] = field(default_factory=dict)
    with_notes: int = 0
    with_trans: int = 0
    hidden_dirs: int = 0
    naming_inconsistencies: list[dict] = field(default_factory=list)
    duplicates: list[list[str]] = field(default_factory=list)
    missing_fields: list[dict] = field(default_factory=list)
    parse_failures: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 解析逻辑
# ---------------------------------------------------------------------------

# 目录命名正则：YYYYMMDD-Source-Short-Title 或 YYYYMMDD-Source-Title
DIR_NAME_RE = re.compile(
    r"^(\d{8})-([^-]+)(?:-([^-]+))?-(.+)$",
    re.IGNORECASE,
)


def parse_dir_name(name: str) -> tuple[dict[str, str] | None, list[str]]:
    """
    解析目录名，返回 (parsed, issues)。
    parsed: {date, source, short, title} 或 None
    issues: 问题列表
    """
    issues = []
    if name.startswith("."):
        issues.append("隐藏目录（以 . 开头）")
        name = name[1:]
    if "http" in name.lower() or "github" in name.lower() or "droid-dataset" in name:
        issues.append("目录名含 URL 或异常片段")
    if "_parsed" in name:
        issues.append("疑似临时解析目录")
    m = DIR_NAME_RE.match(name)
    if not m:
        issues.append("不符合 YYYYMMDD-Source-Short-Title 格式")
        return None, issues
    date, source, short, title = m.groups()
    parsed = {
        "date": date,
        "source": source,
        "short": short or "",
        "title": title or "",
    }
    if not date or len(date) != 8:
        issues.append("日期段无效")
    return parsed, issues


def normalize_publication(pub: Any) -> list[dict]:
    """将 publication 统一为列表。"""
    if pub is None:
        return []
    if isinstance(pub, list):
        return [p for p in pub if isinstance(p, dict)]
    if isinstance(pub, dict):
        return [pub]
    return []


def extract_metadata(paper_yml: dict[str, Any] | None) -> dict[str, Any]:
    """从 paper.yml 提取标准化元数据。"""
    out = {
        "title": None,
        "year": None,
        "venue": None,
        "arxiv_id": None,
        "tags": [],
    }
    if not paper_yml:
        return out
    root = paper_yml.get("paper") or paper_yml
    out["title"] = root.get("title")
    ids = root.get("identifiers") or {}
    arxiv = ids.get("arxiv") if isinstance(ids, dict) else None
    if arxiv:
        out["arxiv_id"] = str(arxiv).split("v")[0] if "v" in str(arxiv) else str(arxiv)
    relations = root.get("relations") or {}
    out["tags"] = list(relations.get("tags") or [])
    pubs = normalize_publication(root.get("publication"))
    if pubs:
        # 取第一个有 year 的
        for p in pubs:
            y = p.get("year")
            if y is not None:
                out["year"] = int(y) if isinstance(y, (int, float)) else None
                break
        # venue 取第一个
        v = pubs[0].get("venue")
        if v:
            out["venue"] = str(v)
    return out


def load_paper_yml(path: Path) -> tuple[dict | None, list[str]]:
    """加载 paper.yml，返回 (data, errors)。"""
    errors = []
    if not path.exists():
        return None, ["文件不存在"]
    if not yaml:
        return None, ["未安装 PyYAML，请运行: pip install pyyaml"]
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            return None, ["根节点不是字典"]
        return data, []
    except Exception as e:
        return None, [str(e)]


def scan_papers(root: Path) -> list[PaperEntry]:
    """扫描 papers 目录，返回 PaperEntry 列表。"""
    entries: list[PaperEntry] = []
    if not root.is_dir():
        return entries
    for item in sorted(root.iterdir()):
        if not item.is_dir():
            continue
        name = item.name
        is_hidden = name.startswith(".")
        paper_yml_path = item / "paper.yml"
        data, load_errors = load_paper_yml(paper_yml_path)
        meta = extract_metadata(data) if data else {}
        md_files = list(item.glob("*.md"))
        trans_files = [f for f in md_files if "_trans" in f.stem]
        notes_path = item / "notes" / "reading-notes.md"
        has_notes = notes_path.exists()
        parsed, naming_issues = parse_dir_name(name)
        entry = PaperEntry(
            dir_path=str(item.resolve()),
            dir_name=name,
            paper_yml_path=str(paper_yml_path) if paper_yml_path.exists() else None,
            paper_yml=data,
            title=meta.get("title"),
            year=meta.get("year"),
            venue=meta.get("venue"),
            arxiv_id=meta.get("arxiv_id"),
            tags=meta.get("tags") or [],
            has_notes=has_notes,
            has_trans=len(trans_files) > 0,
            md_count=len(md_files),
            trans_count=len(trans_files),
            parse_errors=load_errors,
            naming_parsed=parsed,
            naming_issues=naming_issues,
            is_hidden=is_hidden,
        )
        entries.append(entry)
    return entries


def compute_stats(entries: list[PaperEntry]) -> RepoStats:
    """计算仓库统计。"""
    stats = RepoStats()
    stats.total_dirs = len(entries)
    stats.with_paper_yml = sum(1 for e in entries if e.paper_yml is not None)
    stats.without_paper_yml = stats.total_dirs - stats.with_paper_yml
    stats.with_notes = sum(1 for e in entries if e.has_notes)
    stats.with_trans = sum(1 for e in entries if e.has_trans)
    stats.hidden_dirs = sum(1 for e in entries if e.is_hidden)
    for e in entries:
        if e.year is not None:
            stats.year_distribution[e.year] = stats.year_distribution.get(e.year, 0) + 1
        if e.venue:
            stats.venue_distribution[e.venue] = stats.venue_distribution.get(e.venue, 0) + 1
        for t in e.tags:
            stats.tag_distribution[t] = stats.tag_distribution.get(t, 0) + 1
    stats.file_type_counts = {
        "paper_yml": stats.with_paper_yml,
        "with_notes": stats.with_notes,
        "with_trans": stats.with_trans,
    }
    for e in entries:
        if e.naming_issues:
            stats.naming_inconsistencies.append(
                {"dir": e.dir_name, "issues": e.naming_issues}
            )
        if e.parse_errors:
            stats.parse_failures.append(
                {"dir": e.dir_name, "errors": e.parse_errors}
            )
        if e.paper_yml is None and not e.is_hidden:
            stats.missing_fields.append({"dir": e.dir_name, "missing": ["paper.yml"]})
        elif e.paper_yml and not e.is_hidden:
            miss = []
            root = e.paper_yml.get("paper") or e.paper_yml
            if not root.get("title"):
                miss.append("title")
            if not root.get("authors"):
                miss.append("authors")
            pubs = normalize_publication(root.get("publication"))
            if not pubs:
                miss.append("publication")
            if miss:
                stats.missing_fields.append({"dir": e.dir_name, "missing": miss})
    # 重复检测：基于 arxiv_id 或 title 标准化
    def norm(s: str | None) -> str:
        if not s:
            return ""
        return re.sub(r"\s+", " ", s.lower().strip())

    by_arxiv: dict[str, list[str]] = {}
    by_title: dict[str, list[str]] = {}
    for e in entries:
        if e.is_hidden:
            continue
        if e.arxiv_id:
            key = e.arxiv_id
            by_arxiv.setdefault(key, []).append(e.dir_name)
        if e.title:
            key = norm(e.title)
            if key:
                by_title.setdefault(key, []).append(e.dir_name)
    seen_dup: set[frozenset] = set()
    for dirs in by_arxiv.values():
        if len(dirs) > 1:
            t = frozenset(dirs)
            if t not in seen_dup:
                seen_dup.add(t)
                stats.duplicates.append(sorted(dirs))
    for dirs in by_title.values():
        if len(dirs) > 1:
            t = frozenset(dirs)
            if t not in seen_dup:
                seen_dup.add(t)
                stats.duplicates.append(sorted(dirs))
    return stats


# ---------------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------------


def dict_to_json_serializable(obj: Any) -> Any:
    """将 dataclass 等转为可 JSON 序列化的结构。"""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: dict_to_json_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: dict_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [dict_to_json_serializable(v) for v in obj]
    return obj


def write_markdown_report(entries: list[PaperEntry], stats: RepoStats, out_path: Path) -> None:
    """写入 Markdown 报告。"""
    lines = [
        "# papers-repo 仓库分析报告",
        "",
        "## 1. 基础统计",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 论文目录总数 | {stats.total_dirs} |",
        f"| 含 paper.yml | {stats.with_paper_yml} |",
        f"| 无 paper.yml | {stats.without_paper_yml} |",
        f"| 含阅读笔记 | {stats.with_notes} |",
        f"| 含译文 | {stats.with_trans} |",
        f"| 隐藏目录 | {stats.hidden_dirs} |",
        "",
        "## 2. 年份分布",
        "",
    ]
    for year in sorted(stats.year_distribution.keys(), reverse=True):
        lines.append(f"- {year}: {stats.year_distribution[year]} 篇")
    lines.extend(["", "## 3. 来源/会议分布", ""])
    for venue, cnt in sorted(stats.venue_distribution.items(), key=lambda x: -x[1]):
        lines.append(f"- {venue}: {cnt} 篇")
    lines.extend(["", "## 4. 标签分布", ""])
    for tag, cnt in sorted(stats.tag_distribution.items(), key=lambda x: -x[1]):
        lines.append(f"- {tag}: {cnt} 篇")
    lines.extend(["", "## 5. 质量检查", ""])
    if stats.missing_fields:
        lines.append("### 5.1 缺失字段")
        lines.append("")
        for m in stats.missing_fields[:20]:
            lines.append(f"- `{m['dir']}`: 缺失 {m['missing']}")
        if len(stats.missing_fields) > 20:
            lines.append(f"- ... 共 {len(stats.missing_fields)} 项")
        lines.append("")
    if stats.parse_failures:
        lines.append("### 5.2 解析失败")
        lines.append("")
        for p in stats.parse_failures[:10]:
            lines.append(f"- `{p['dir']}`: {p['errors']}")
        if len(stats.parse_failures) > 10:
            lines.append(f"- ... 共 {len(stats.parse_failures)} 项")
        lines.append("")
    if stats.naming_inconsistencies:
        lines.append("### 5.3 目录命名问题")
        lines.append("")
        for n in stats.naming_inconsistencies[:15]:
            lines.append(f"- `{n['dir']}`: {', '.join(n['issues'])}")
        if len(stats.naming_inconsistencies) > 15:
            lines.append(f"- ... 共 {len(stats.naming_inconsistencies)} 项")
        lines.append("")
    if stats.duplicates:
        lines.append("### 5.4 疑似重复")
        lines.append("")
        for d in stats.duplicates:
            lines.append(f"- {d}")
        lines.append("")
    lines.extend(["", "## 6. 改进建议", ""])
    suggestions = []
    if stats.duplicates:
        suggestions.append("合并或删除重复目录")
    if stats.naming_inconsistencies:
        suggestions.append("按报告中的命名问题逐项修正目录名")
    if stats.without_paper_yml > 0:
        suggestions.append("为无 paper.yml 的目录补充元数据")
    if stats.hidden_dirs:
        suggestions.append("确认隐藏目录（. 开头）是否需要保留或移出")
    if not suggestions:
        suggestions.append("当前无高优先级问题")
    for s in suggestions:
        lines.append(f"- {s}")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_json_report(entries: list[PaperEntry], stats: RepoStats, out_path: Path) -> None:
    """写入 JSON 报告。"""
    payload = {
        "stats": dict_to_json_serializable(stats),
        "entries": [
            {
                "dir_name": e.dir_name,
                "dir_path": e.dir_path,
                "title": e.title,
                "year": e.year,
                "venue": e.venue,
                "arxiv_id": e.arxiv_id,
                "tags": e.tags,
                "has_notes": e.has_notes,
                "has_trans": e.has_trans,
                "naming_issues": e.naming_issues,
                "parse_errors": e.parse_errors,
            }
            for e in entries
        ],
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv_index(entries: list[PaperEntry], out_path: Path) -> None:
    """写入 CSV 索引。"""
    rows = []
    for e in entries:
        rows.append({
            "dir_name": e.dir_name,
            "title": e.title or "",
            "year": e.year or "",
            "venue": e.venue or "",
            "arxiv_id": e.arxiv_id or "",
            "tags": ";".join(e.tags),
            "has_notes": e.has_notes,
            "has_trans": e.has_trans,
            "md_count": e.md_count,
            "trans_count": e.trans_count,
            "naming_issues": ";".join(e.naming_issues),
        })
    if not rows:
        out_path.write_text("dir_name,title,year,venue,arxiv_id,tags,has_notes,has_trans,md_count,trans_count,naming_issues\n", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="papers-repo 仓库分析与报告")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("papers"),
        help="papers 目录路径（默认: papers）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/latest"),
        help="报告输出目录（默认: reports/latest）",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = scan_papers(root)
    stats = compute_stats(entries)
    write_markdown_report(entries, stats, out_dir / "report.md")
    write_json_report(entries, stats, out_dir / "report.json")
    write_csv_index(entries, out_dir / "papers_index.csv")
    print(f"报告已生成: {out_dir}")
    print(f"  - report.md")
    print(f"  - report.json")
    print(f"  - papers_index.csv")
    print(f"论文目录: {stats.total_dirs}, 含 paper.yml: {stats.with_paper_yml}")


if __name__ == "__main__":
    main()
