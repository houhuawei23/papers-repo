#!/usr/bin/env python3
"""
扫描 paper 目录并生成可索引跳转的 List.md。

特性：
1. 扫描给定目录下的一层 paper 子目录（如 YYYYMMDD-Source-Title）。
2. 解析每个子目录下的 paper.yml，提取标题、年份、venue、arXiv、作者、标签等元信息。
3. 自动发现本地主文档（*.md）和译文（*_trans.md / *_tran.md）。
4. 输出易读且可点击跳转的 Markdown 索引文件。

用法示例：
  python scripts/generate_paper_list.py \
    --root papers/Diffusion-Model \
    --output papers/Diffusion-Model/List.md

  python scripts/generate_paper_list.py \
    --root papers/VLA \
    --output papers/VLA/List.md \
    --sort-by date --descending
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


LOGGER = logging.getLogger("generate_paper_list")
DATE_PREFIX_RE = re.compile(r"^(\d{8})-")


@dataclass
class PaperMeta:
    """单个论文目录的标准化信息。"""

    dir_name: str
    dir_path: Path
    date_prefix: str | None = None
    has_paper_yml: bool = False
    title: str | None = None
    year: int | None = None
    venue: str | None = None
    arxiv: str | None = None
    authors: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    main_md: Path | None = None
    trans_md: Path | None = None
    extra_docs: list[tuple[str, Path]] = field(default_factory=list)
    extra_trans: list[tuple[str, Path]] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="扫描 paper 目录并生成可跳转的 List.md"
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="paper 集合目录路径（例如: papers/Diffusion-Model）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 Markdown 文件路径（默认: <root>/List.md）",
    )
    parser.add_argument(
        "--sort-by",
        choices=["date", "year", "title", "dir"],
        default="date",
        help="排序字段（默认: date）",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="是否降序（默认升序；当 sort-by=date 时建议配合使用）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, (os.cpu_count() or 4) * 2),
        help="并发 worker 数（默认: min(16, CPU*2)）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印统计，不写入文件",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出更详细日志",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def normalize_publication(publication: Any) -> list[dict[str, Any]]:
    if publication is None:
        return []
    if isinstance(publication, dict):
        return [publication]
    if isinstance(publication, list):
        return [item for item in publication if isinstance(item, dict)]
    return []


def extract_date_prefix(name: str) -> str | None:
    match = DATE_PREFIX_RE.match(name)
    return match.group(1) if match else None


def find_markdown_files(
    paper_dir: Path,
) -> tuple[Path | None, Path | None, list[tuple[str, Path]], list[tuple[str, Path]]]:
    """返回 (主文档, 译文文档, 额外文档列表, 额外译文列表)。额外文档为 (label, path) 元组。"""
    md_files = sorted(paper_dir.glob("*.md"))
    if not md_files:
        return None, None, [], []

    preferred_main_name = f"{paper_dir.name}.md".lower()
    preferred_trans_names = {
        f"{paper_dir.name}_trans.md".lower(),
        f"{paper_dir.name}_tran.md".lower(),
    }

    main_md: Path | None = None
    trans_md: Path | None = None

    for fp in md_files:
        name_lower = fp.name.lower()
        if name_lower == preferred_main_name:
            main_md = fp
        elif name_lower in preferred_trans_names:
            trans_md = fp

    if main_md is not None:
        base_prefix = main_md.stem
    else:
        # 启发式：找最可能是主文档的文件（无 _trans/_tran 后缀且 stem 最短）
        non_trans: list[Path] = []
        for fp in md_files:
            stem_lower = fp.stem.lower()
            if stem_lower.endswith("_trans") or stem_lower.endswith("_tran"):
                continue
            non_trans.append(fp)
        if non_trans:
            main_md = min(non_trans, key=lambda f: len(f.stem))
        else:
            main_md = md_files[0]
        base_prefix = main_md.stem

    extra_docs: list[tuple[str, Path]] = []
    extra_trans: list[tuple[str, Path]] = []

    base_lower = base_prefix.lower()
    for fp in md_files:
        if fp == main_md or fp == trans_md:
            continue

        stem_lower = fp.stem.lower()
        # 主译文
        if stem_lower == f"{base_lower}_trans" or stem_lower == f"{base_lower}_tran":
            trans_md = fp
            continue

        # 额外文档或额外译文
        if stem_lower.startswith(base_lower + "-"):
            rest = fp.stem[len(base_prefix) + 1 :]
            rest_lower = rest.lower()
            if rest_lower.endswith("_trans") or rest_lower.endswith("_tran"):
                extra_trans.append((rest, fp))
            else:
                extra_docs.append((rest, fp))

    return main_md, trans_md, extra_docs, extra_trans


def load_yaml(yml_path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if not yml_path.exists():
        return None, ["paper.yml 不存在"]
    if yaml is None:
        return None, ["缺少 PyYAML（请安装 pyyaml）"]

    try:
        content = yml_path.read_text(encoding="utf-8", errors="replace")
        data = yaml.safe_load(content)
    except Exception as exc:
        return None, [f"YAML 解析失败: {exc}"]

    if not isinstance(data, dict):
        return None, ["YAML 根节点不是字典"]
    return data, []


def extract_meta_from_yml(data: dict[str, Any]) -> dict[str, Any]:
    """
    兼容两种格式：
    1) paper: {...}
    2) 直接平铺字段
    """
    root = data.get("paper", data)
    if not isinstance(root, dict):
        return {}

    identifiers = root.get("identifiers", {})
    relations = root.get("relations", {})
    publication_list = normalize_publication(root.get("publication"))

    title = root.get("title")
    arxiv = None
    if isinstance(identifiers, dict):
        arxiv = identifiers.get("arxiv")
    if not arxiv:
        paper_id = root.get("id")
        if isinstance(paper_id, str) and "arxiv:" in paper_id:
            arxiv = paper_id.split("arxiv:")[-1].strip()

    venue: str | None = None
    year: int | None = None
    if publication_list:
        venue_raw = publication_list[0].get("venue")
        venue = str(venue_raw) if venue_raw is not None else None
        for pub in publication_list:
            pub_year = pub.get("year")
            if isinstance(pub_year, (int, float)):
                year = int(pub_year)
                break
            if isinstance(pub_year, str) and pub_year.isdigit():
                year = int(pub_year)
                break

    authors_raw = root.get("authors", [])
    authors: list[str] = []
    if isinstance(authors_raw, list):
        for author in authors_raw:
            if isinstance(author, dict) and author.get("name"):
                authors.append(str(author["name"]))
            elif isinstance(author, str):
                authors.append(author)

    tags: list[str] = []
    if isinstance(relations, dict):
        raw_tags = relations.get("tags", [])
        if isinstance(raw_tags, list):
            tags = [str(item) for item in raw_tags if item]

    return {
        "title": str(title) if title else None,
        "year": year,
        "venue": venue,
        "arxiv": str(arxiv) if arxiv else None,
        "authors": authors,
        "tags": tags,
    }


def scan_one_paper_dir(paper_dir: Path) -> PaperMeta:
    meta = PaperMeta(
        dir_name=paper_dir.name,
        dir_path=paper_dir,
        date_prefix=extract_date_prefix(paper_dir.name),
    )
    meta.main_md, meta.trans_md, meta.extra_docs, meta.extra_trans = find_markdown_files(paper_dir)

    yml_path = paper_dir / "paper.yml"
    meta.has_paper_yml = yml_path.exists()

    yml_data, errors = load_yaml(yml_path)
    meta.parse_errors.extend(errors)
    if yml_data is None:
        return meta

    yml_meta = extract_meta_from_yml(yml_data)
    meta.title = yml_meta.get("title")
    meta.year = yml_meta.get("year")
    meta.venue = yml_meta.get("venue")
    meta.arxiv = yml_meta.get("arxiv")
    meta.authors = yml_meta.get("authors", [])
    meta.tags = yml_meta.get("tags", [])
    return meta


def scan_paper_dirs(root: Path, workers: int) -> list[PaperMeta]:
    paper_dirs = sorted(
        item for item in root.iterdir() if item.is_dir() and not item.name.startswith(".")
    )
    LOGGER.info("发现 %d 个候选目录，开始扫描...", len(paper_dirs))

    results: list[PaperMeta] = []
    if not paper_dirs:
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(scan_one_paper_dir, paper_dir) for paper_dir in paper_dirs]
        total = len(futures)
        for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            meta = future.result()
            results.append(meta)
            if idx % 10 == 0 or idx == total:
                LOGGER.info("扫描进度: %d/%d", idx, total)
    return results


def format_authors(authors: Iterable[str]) -> str:
    names = [name.strip() for name in authors if name and name.strip()]
    if not names:
        return "-"
    if len(names) <= 3:
        return ", ".join(names)
    return f"{', '.join(names[:3])} 等 {len(names)} 位"


def to_relative_link(target: Path, base_dir: Path) -> str:
    relative = target.relative_to(base_dir).as_posix()
    return relative


def sort_entries(entries: list[PaperMeta], sort_by: str, descending: bool) -> list[PaperMeta]:
    def key_func(item: PaperMeta) -> Any:
        if sort_by == "date":
            return item.date_prefix or ""
        if sort_by == "year":
            return item.year or -1
        if sort_by == "title":
            return (item.title or "").lower()
        return item.dir_name.lower()

    return sorted(entries, key=key_func, reverse=descending)


def build_markdown(entries: list[PaperMeta], root: Path, output_file: Path) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(entries)
    with_yml = sum(1 for item in entries if item.has_paper_yml)
    with_main = sum(1 for item in entries if item.main_md is not None)
    with_trans = sum(1 for item in entries if item.trans_md is not None)
    with_error = sum(1 for item in entries if item.parse_errors)

    lines: list[str] = [
        "# Paper List",
        "",
        f"- 根目录: `{root.as_posix()}`",
        f"- 生成时间: `{generated_at}`",
        f"- 统计: 共 **{total}** 篇，含 paper.yml **{with_yml}**，主文档 **{with_main}**，译文 **{with_trans}**，解析异常 **{with_error}**",
        "",
        "## 索引",
        "",
        "| # | 日期 | 论文 | Venue/Year | ArXiv | 文档 | 译文 |",
        "|---:|:---:|---|---|---|---|---|",
    ]

    output_dir = output_file.parent
    for idx, item in enumerate(entries, start=1):
        date_text = item.date_prefix or "-"
        title_text = item.title or item.dir_name
        venue_text = item.venue or "-"
        year_text = str(item.year) if item.year is not None else "-"
        venue_year_text = f"{venue_text} / {year_text}"
        arxiv_text = item.arxiv or "-"

        if item.main_md:
            md_link = to_relative_link(item.main_md, output_dir)
            paper_text = f"[{title_text}]({md_link})"
            doc_parts = [f"[main]({md_link})"]
        else:
            dir_link = to_relative_link(item.dir_path, output_dir)
            paper_text = f"[{title_text}]({dir_link})"
            doc_parts: list[str] = []

        for label, fp in item.extra_docs:
            doc_parts.append(f"[{label}]({to_relative_link(fp, output_dir)})")
        doc_text = " / ".join(doc_parts) if doc_parts else "-"

        trans_parts: list[str] = []
        if item.trans_md:
            trans_parts.append(f"[trans]({to_relative_link(item.trans_md, output_dir)})")
        for label, fp in item.extra_trans:
            trans_parts.append(f"[{label}]({to_relative_link(fp, output_dir)})")
        trans_text = " / ".join(trans_parts) if trans_parts else "-"

        lines.append(
            f"| {idx} | {date_text} | {paper_text} | {venue_year_text} | {arxiv_text} | {doc_text} | {trans_text} |"
        )

    lines.extend(["", "## 元信息补充", ""])

    for idx, item in enumerate(entries, start=1):
        title_text = item.title or item.dir_name
        lines.append(f"### {idx}. {title_text}")
        lines.append(f"- 目录: `{item.dir_name}`")
        lines.append(f"- 作者: {format_authors(item.authors)}")
        lines.append(f"- 标签: {', '.join(item.tags) if item.tags else '-'}")

        doc_links: list[str] = []
        if item.main_md:
            doc_links.append(f"[main]({to_relative_link(item.main_md, output_dir)})")
        for label, fp in item.extra_docs:
            doc_links.append(f"[{label}]({to_relative_link(fp, output_dir)})")
        if doc_links:
            lines.append(f"- 文档: {' / '.join(doc_links)}")

        trans_links: list[str] = []
        if item.trans_md:
            trans_links.append(f"[trans]({to_relative_link(item.trans_md, output_dir)})")
        for label, fp in item.extra_trans:
            trans_links.append(f"[{label}]({to_relative_link(fp, output_dir)})")
        if trans_links:
            lines.append(f"- 译文: {' / '.join(trans_links)}")

        if item.parse_errors:
            lines.append(f"- 解析提示: {'; '.join(item.parse_errors)}")
        lines.append("")

    if with_yml == 0:
        lines.append("> 提示：当前未解析到有效 `paper.yml`，请检查文件格式。")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    root = args.root.resolve()
    if not root.exists() or not root.is_dir():
        LOGGER.error("输入目录不存在或不是目录: %s", root)
        return 1

    output = args.output.resolve() if args.output else (root / "List.md")
    output.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("输入目录: %s", root)
    LOGGER.info("输出文件: %s", output)

    entries = scan_paper_dirs(root=root, workers=args.workers)
    entries = sort_entries(entries, sort_by=args.sort_by, descending=args.descending)
    markdown = build_markdown(entries=entries, root=root, output_file=output)

    if args.dry_run:
        LOGGER.info("dry-run 模式：不写入文件，仅输出前 40 行预览。")
        preview_lines = markdown.splitlines()[:40]
        print("\n".join(preview_lines))
        return 0

    output.write_text(markdown, encoding="utf-8")
    LOGGER.info("已生成 List 文件: %s", output)
    LOGGER.info("完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
