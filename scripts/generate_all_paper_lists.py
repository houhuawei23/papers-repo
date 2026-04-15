#!/usr/bin/env python3
"""
批量为 papers/* 分组生成 List.md 的入口脚本。

本脚本会遍历给定 papers 根目录下的每个一级分组目录（如 VLA、Diffusion-Model），
并调用单目录生成逻辑，自动生成对应的 List.md。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# 复用单目录脚本中的核心逻辑，避免重复实现。
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_paper_list import (  # pylint: disable=wrong-import-position
    build_markdown,
    scan_paper_dirs,
    sort_entries,
)


LOGGER = logging.getLogger("generate_all_paper_lists")


@dataclass
class GroupResult:
    group_name: str
    output_path: Path
    paper_count: int
    success: bool
    error_message: str | None = None
    elapsed_ms: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量扫描 papers/* 并为每个分组生成 List.md"
    )
    parser.add_argument(
        "--papers-root",
        type=Path,
        default=Path("papers"),
        help="papers 根目录（默认: papers）",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="List.md",
        help="每个分组的输出文件名（默认: List.md）",
    )
    parser.add_argument(
        "--sort-by",
        choices=["date", "year", "title", "dir"],
        default="date",
        help="分组内列表排序字段（默认: date）",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="分组内列表按降序输出",
    )
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=min(16, (os.cpu_count() or 4) * 2),
        help="单个分组内部扫描并发数（默认: min(16, CPU*2)）",
    )
    parser.add_argument(
        "--group-workers",
        type=int,
        default=min(8, max(2, os.cpu_count() or 4)),
        help="分组级并发数（默认: min(8, max(2, CPU))）",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="*",
        default=None,
        help="仅处理这些分组名（例如: --include VLA Diffusion-Model）",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=None,
        help="排除这些分组名",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将处理的分组，不写入文件",
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


def discover_groups(
    papers_root: Path,
    include: set[str] | None,
    exclude: set[str] | None,
) -> list[Path]:
    candidates = [
        item
        for item in sorted(papers_root.iterdir())
        if item.is_dir() and not item.name.startswith(".")
    ]
    groups: list[Path] = []
    for item in candidates:
        name = item.name
        if include and name not in include:
            continue
        if exclude and name in exclude:
            continue
        groups.append(item)
    return groups


def generate_for_group(
    group_dir: Path,
    output_name: str,
    sort_by: str,
    descending: bool,
    scan_workers: int,
    dry_run: bool,
) -> GroupResult:
    start = time.perf_counter()
    output_path = group_dir / output_name
    try:
        entries = scan_paper_dirs(root=group_dir, workers=scan_workers)
        entries = sort_entries(entries=entries, sort_by=sort_by, descending=descending)
        markdown = build_markdown(entries=entries, root=group_dir, output_file=output_path)

        if not dry_run:
            output_path.write_text(markdown, encoding="utf-8")

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return GroupResult(
            group_name=group_dir.name,
            output_path=output_path,
            paper_count=len(entries),
            success=True,
            elapsed_ms=elapsed_ms,
        )
    except Exception as exc:  # pragma: no cover
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return GroupResult(
            group_name=group_dir.name,
            output_path=output_path,
            paper_count=0,
            success=False,
            error_message=str(exc),
            elapsed_ms=elapsed_ms,
        )


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    papers_root = args.papers_root.resolve()
    if not papers_root.exists() or not papers_root.is_dir():
        LOGGER.error("papers 根目录不存在或不是目录: %s", papers_root)
        return 1

    include = set(args.include) if args.include else None
    exclude = set(args.exclude) if args.exclude else None
    groups = discover_groups(papers_root=papers_root, include=include, exclude=exclude)
    if not groups:
        LOGGER.warning("未找到可处理分组。")
        return 0

    LOGGER.info("将处理 %d 个分组。", len(groups))
    if args.dry_run:
        LOGGER.info("dry-run 模式，不写入文件。")

    results: list[GroupResult] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.group_workers)
    ) as executor:
        futures = [
            executor.submit(
                generate_for_group,
                group_dir=group_dir,
                output_name=args.output_name,
                sort_by=args.sort_by,
                descending=args.descending,
                scan_workers=args.scan_workers,
                dry_run=args.dry_run,
            )
            for group_dir in groups
        ]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            if result.success:
                LOGGER.info(
                    "[%s] 完成: %s (%d 篇, %dms)",
                    result.group_name,
                    result.output_path,
                    result.paper_count,
                    result.elapsed_ms,
                )
            else:
                LOGGER.error(
                    "[%s] 失败: %s",
                    result.group_name,
                    result.error_message,
                )

    total = len(results)
    success = sum(1 for item in results if item.success)
    failed = total - success
    papers_total = sum(item.paper_count for item in results if item.success)

    LOGGER.info(
        "批量完成: 分组 %d, 成功 %d, 失败 %d, 累计论文 %d",
        total,
        success,
        failed,
        papers_total,
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
