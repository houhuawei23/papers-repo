#!/usr/bin/env python3
"""
批量更新 papers 目录下的 paper.yml 为统一嵌套格式。

用法:
  python update_paper_yml.py [papers_root]
  python update_paper_yml.py /path/to/papers

从每个 paper.yml 解析 arxiv id，调用 arXiv API 获取完整元数据，写入统一格式。
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

# 添加 arxiv2md-beta src 到路径
_SCRIPT_DIR = Path(__file__).resolve().parent
_ARXIV2MD_SRC = _SCRIPT_DIR / "arxiv2md-beta" / "src"
if _ARXIV2MD_SRC.exists():
    sys.path.insert(0, str(_ARXIV2MD_SRC))

try:
    from arxiv2md_beta.arxiv_api import fetch_arxiv_metadata
    from arxiv2md_beta.paper_metadata import save_paper_metadata
except ImportError as e:
    print(f"Error: 需要 arxiv2md_beta 模块。请确保 arxiv2md-beta 已安装或位于 my_scripts/arxiv2md-beta/src")
    print(f"  {e}")
    sys.exit(1)


def extract_arxiv_id_from_yml(yml_path: Path) -> str | None:
    """从 paper.yml 解析 arxiv id。支持多种格式。"""
    try:
        import yaml
    except ImportError:
        return None

    with open(yml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data or not isinstance(data, dict):
        return None

    # 1. 顶层 arxiv
    arxiv_val = data.get("arxiv") or data.get("arXiv")
    if arxiv_val and isinstance(arxiv_val, str):
        match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", arxiv_val)
        if match:
            return match.group(1)

    # 2. paper.identifiers.arxiv
    paper = data.get("paper")
    if isinstance(paper, dict):
        identifiers = paper.get("identifiers") or {}
        arxiv_val = identifiers.get("arxiv")
        if arxiv_val and isinstance(arxiv_val, str):
            return arxiv_val.strip()

        # paper.id
        paper_id = paper.get("id")
        if paper_id and isinstance(paper_id, str) and "arxiv" in paper_id.lower():
            match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", paper_id)
            if match:
                return match.group(1)

    return None


def find_paper_yml_files(papers_root: Path) -> list[Path]:
    """递归查找所有 paper.yml 文件。"""
    return list(papers_root.rglob("paper.yml"))


async def update_one(yml_path: Path, dry_run: bool = False) -> bool:
    """更新单个 paper.yml。"""
    arxiv_id = extract_arxiv_id_from_yml(yml_path)
    if not arxiv_id:
        print(f"  [跳过] 无 arxiv id: {yml_path}")
        return False

    metadata = await fetch_arxiv_metadata(arxiv_id)
    if not metadata.get("arxiv_id"):
        print(f"  [失败] 无法获取元数据: {arxiv_id} ({yml_path})")
        return False

    paper_dir = yml_path.parent
    if dry_run:
        print(f"  [dry-run] 将更新: {paper_dir} (arxiv:{arxiv_id})")
        return True

    save_paper_metadata(metadata, paper_dir)
    print(f"  [完成] {paper_dir.name} (arxiv:{arxiv_id})")
    return True


async def main(papers_root: Path, dry_run: bool = False) -> None:
    yml_files = find_paper_yml_files(papers_root)
    print(f"找到 {len(yml_files)} 个 paper.yml 文件")

    updated = 0
    skipped = 0
    failed = 0

    for yml_path in sorted(yml_files):
        try:
            ok = await update_one(yml_path, dry_run)
            if ok:
                updated += 1
            else:
                skipped += 1
        except Exception as e:
            failed += 1
            print(f"  [错误] {yml_path}: {e}")

    print(f"\n总计: 更新 {updated}, 跳过 {skipped}, 失败 {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量更新 paper.yml 为统一格式")
    parser.add_argument(
        "papers_root",
        nargs="?",
        default=Path(__file__).resolve().parent.parent / "00_Research" / "papers",
        type=Path,
        help="papers 目录路径",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要更新的文件，不实际写入",
    )
    args = parser.parse_args()

    papers_root = args.papers_root
    if not papers_root.exists():
        print(f"目录不存在: {papers_root}")
        sys.exit(1)

    asyncio.run(main(papers_root, dry_run=args.dry_run))
