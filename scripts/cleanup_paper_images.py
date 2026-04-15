#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理 papers 目录下各 paper 目录中未被 Markdown 文件引用的图片。

Usage:
    python cleanup_paper_images.py
    python cleanup_paper_images.py --dry-run
    python cleanup_paper_images.py --yes
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.table import Table
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import Button, Checkbox, Label, Static

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PAPERS_DIR: Path = Path("papers")
IMAGES_SUBDIR: str = "images"
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".tiff"}
)

app = typer.Typer(
    add_completion=False,
    help="清理 papers 目录下未被 Markdown 引用的图片。",
)
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_paper_directory(path: Path) -> bool:
    """判断一个路径是否为 paper 目录（包含 images 子目录）。"""
    return path.is_dir() and (path / IMAGES_SUBDIR).is_dir()


def collect_images(images_dir: Path) -> list[Path]:
    """收集 images 目录下的所有图片文件，按文件名排序。"""
    images: list[Path] = [
        f
        for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images.sort(key=lambda p: p.name.lower())
    return images


def collect_markdowns(paper_dir: Path) -> list[Path]:
    """收集 paper 目录下所有 .md 文件（递归）。"""
    return sorted(paper_dir.rglob("*.md"), key=lambda p: str(p).lower())


def build_markdown_content_cache(markdowns: list[Path]) -> str:
    """将所有 Markdown 文件内容合并为一个字符串，用于快速搜索。"""
    parts: list[str] = []
    for md in markdowns:
        try:
            parts.append(md.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("无法读取 Markdown 文件 {}: {}", md, exc)
    return "\n".join(parts)


def find_unreferenced_images(
    images: list[Path],
    markdown_content: str,
) -> list[Path]:
    """返回未被任何 Markdown 文件引用的图片列表。"""
    unreferenced: list[Path] = []
    for img in images:
        name = img.name
        stem = img.stem
        if name not in markdown_content and stem not in markdown_content:
            unreferenced.append(img)
    return unreferenced


def scan_papers(
    papers_dir: Path,
    progress: Progress,
) -> dict[Path, list[Path]]:
    """扫描所有 paper 目录，返回 {paper_dir: [unreferenced_image, ...]}。"""
    if not papers_dir.exists():
        console.print(f"[red]错误：目录不存在 {papers_dir.resolve()}[/red]")
        raise typer.Exit(code=1)

    if is_paper_directory(papers_dir):
        paper_dirs = [papers_dir]
    else:
        paper_dirs = sorted(
            {p.parent for p in papers_dir.rglob("*/images") if p.is_dir()},
            key=lambda p: str(p).lower(),
        )

    result: dict[Path, list[Path]] = {}
    task_id = progress.add_task("扫描 paper 目录...", total=len(paper_dirs))

    for paper_dir in paper_dirs:
        images_dir = paper_dir / IMAGES_SUBDIR
        images = collect_images(images_dir)
        if not images:
            progress.advance(task_id)
            continue

        markdowns = collect_markdowns(paper_dir)
        if not markdowns:
            result[paper_dir] = images
        else:
            content = build_markdown_content_cache(markdowns)
            unreferenced = find_unreferenced_images(images, content)
            if unreferenced:
                result[paper_dir] = unreferenced

        progress.advance(task_id)

    return result


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def _link(path: Path, label: str | None = None) -> str:
    """生成带 file:// 超链接的 rich markup 文本。"""
    label = label or path.name
    return f"[link=file://{path.resolve()}]{label}[/link]"


def format_size(size: int) -> str:
    """将字节数格式化为人类可读字符串。"""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def build_preview_table(images: list[Path]) -> Table:
    """构建双栏图片预览表。"""
    table = Table(show_header=False, show_edge=False, box=None, padding=(0, 1))
    table.add_column("左栏", ratio=1, no_wrap=True)
    table.add_column("右栏", ratio=1, no_wrap=True)
    half = (len(images) + 1) // 2
    left = images[:half]
    right = images[half:]
    for row in range(half):
        l_img = left[row]
        l_str = (
            f"{row + 1:>3}. {_link(l_img, l_img.name)} "
            f"[dim]({format_size(l_img.stat().st_size)})[/dim]"
        )
        r_str = ""
        if row < len(right):
            r_img = right[row]
            r_idx = row + 1 + half
            r_str = (
                f"{r_idx:>3}. {_link(r_img, r_img.name)} "
                f"[dim]({format_size(r_img.stat().st_size)})[/dim]"
            )
        table.add_row(l_str, r_str)
    return table


# ---------------------------------------------------------------------------
# TUI App
# ---------------------------------------------------------------------------
class CleanupApp(App[list[Path]]):
    """用于选择待清理目录的 Textual TUI。"""

    CSS = """
    Screen { align: center middle; }
    #main { width: 100%; height: 1fr; }
    #left-panel { width: 50%; height: 100%; border: solid green; }
    #right-panel { width: 50%; height: 100%; border: solid blue; }
    #button-bar { height: auto; dock: bottom; margin: 1; }
    """

    def __init__(
        self,
        candidates: dict[Path, list[Path]],
        *,
        dry_run: bool = False,
    ) -> None:
        self.candidates = candidates
        self.sorted_dirs = sorted(candidates.keys(), key=lambda p: p.name.lower())
        self.dry_run = dry_run
        self.checkboxes: dict[int, Checkbox] = {}
        self.idx_to_dir: dict[int, Path] = {}
        super().__init__()

    def compose(self) -> ComposeResult:
        with Horizontal(id="main"):
            with Vertical(id="left-panel"):
                yield Label("📁 选择要清理的目录（鼠标点击勾选 / 空格键切换）：")
                with ScrollableContainer():
                    for idx, paper_dir in enumerate(self.sorted_dirs, 1):
                        images = self.candidates[paper_dir]
                        total_size = sum(img.stat().st_size for img in images)
                        cb = Checkbox(
                            f"[{idx}] {paper_dir.name}  "
                            f"({len(images)} 张, {format_size(total_size)})",
                            id=f"cb-{idx}",
                        )
                        self.checkboxes[idx] = cb
                        self.idx_to_dir[idx] = paper_dir
                        yield cb
            with Vertical(id="right-panel"):
                yield Label("🖼️ 图片预览（勾选左侧目录后自动显示）：")
                with ScrollableContainer():
                    self.preview = Static("请勾选左侧目录查看详情", id="preview")
                    yield self.preview

        with Horizontal(id="button-bar"):
            if self.dry_run:
                yield Button(
                    "🔍 退出预览", variant="primary", id="btn-exit"
                )
            else:
                yield Button(
                    "✅ 删除选中", variant="error", id="btn-delete"
                )
            yield Button("☑️ 全选", id="btn-select-all")
            yield Button("⬜ 全不选", id="btn-select-none")
            yield Button("❌ 取消", variant="primary", id="btn-cancel")

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """当 Checkbox 状态变化时，若被勾选则更新右侧预览。"""
        for idx, cb in self.checkboxes.items():
            if cb == event.checkbox and event.value:
                self.preview.update(build_preview_table(self.candidates[self.idx_to_dir[idx]]))
                break

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """处理底部按钮事件。"""
        btn_id = event.button.id
        if btn_id == "btn-cancel":
            self.exit([])
        elif btn_id == "btn-exit":
            self.exit(
                [
                    self.idx_to_dir[idx]
                    for idx, cb in self.checkboxes.items()
                    if cb.value
                ]
            )
        elif btn_id == "btn-select-all":
            for cb in self.checkboxes.values():
                cb.value = True
        elif btn_id == "btn-select-none":
            for cb in self.checkboxes.values():
                cb.value = False
        elif btn_id == "btn-delete":
            selected = [
                self.idx_to_dir[idx]
                for idx, cb in self.checkboxes.items()
                if cb.value
            ]
            self.exit(selected)


# ---------------------------------------------------------------------------
# Main interaction
# ---------------------------------------------------------------------------
def confirm_and_delete(
    candidates: dict[Path, list[Path]],
    *,
    dry_run: bool,
    auto_yes: bool,
) -> None:
    """展示 TUI 让用户选择目录，随后逐个确认删除。"""
    if not candidates:
        console.print(Panel("[green]恭喜！未发现未被引用的图片。[/green]", title="结果"))
        return

    total_files = sum(len(imgs) for imgs in candidates.values())
    total_bytes = sum(
        img.stat().st_size for imgs in candidates.values() for img in imgs
    )
    console.print(
        Panel(
            f"发现 [yellow]{len(candidates)}[/yellow] 个 paper 目录中存在 "
            f"[yellow]{total_files}[/yellow] 张未引用的图片，"
            f"共计 [yellow]{format_size(total_bytes)}[/yellow]。",
            title="扫描完成",
        )
    )

    sorted_dirs = sorted(candidates.keys(), key=lambda p: p.name.lower())

    if auto_yes:
        selected_dirs = sorted_dirs
    else:
        tui = CleanupApp(candidates, dry_run=dry_run)
        selected_dirs = tui.run()
        if selected_dirs is None:
            selected_dirs = []

    if not selected_dirs:
        console.print("[dim]未选择任何目录，退出。[/dim]")
        return

    deleted_count = 0
    skipped_count = 0

    for current_idx, paper_dir in enumerate(selected_dirs, start=1):
        images = candidates[paper_dir]

        if dry_run:
            console.print(
                f"[{current_idx}/{len(selected_dirs)}] "
                f"[dim]{paper_dir.name} ({len(images)} 张) --dry-run 跳过[/dim]"
            )
            continue

        should_delete = Confirm.ask(
            f"[{current_idx}/{len(selected_dirs)}] "
            f"是否删除 [bold]{paper_dir.name}[/bold] 中的 {len(images)} 张图片？",
            default=False,
        )
        if should_delete:
            for img in images:
                try:
                    img.unlink()
                    logger.info("已删除: {}", img)
                    deleted_count += 1
                except Exception as exc:  # noqa: BLE001
                    logger.error("删除失败 {}: {}", img, exc)
                    console.print(f"  [red]删除失败: {img.name} ({exc})[/red]")
            console.print(f"  [green]✓ 已删除 {len(images)} 张图片[/green]")
        else:
            skipped_count += len(images)
            console.print("  [dim]已跳过该目录。[/dim]")

    # 未选中的算 skipped
    unselected_dirs = [d for d in sorted_dirs if d not in selected_dirs]
    skipped_count += sum(len(candidates[d]) for d in unselected_dirs)

    console.print("\n")
    summary_table = Table(title="执行摘要", show_header=False)
    summary_table.add_column("项目", style="cyan")
    summary_table.add_column("数量", justify="right", style="magenta")
    summary_table.add_row("待清理目录总数", str(len(candidates)))
    summary_table.add_row("待清理文件总数", str(total_files))
    summary_table.add_row("已处理目录", str(len(selected_dirs)))
    summary_table.add_row("已删除文件", str(deleted_count))
    summary_table.add_row("已跳过文件", str(skipped_count))
    console.print(summary_table)


# ---------------------------------------------------------------------------
# CLI Entry
# ---------------------------------------------------------------------------
@app.command()
def main(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="只显示结果，不实际删除文件。",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="跳过确认，直接删除。",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="输出详细日志。",
    ),
    papers_dir: Path = typer.Option(
        PAPERS_DIR,
        "--papers-dir",
        "-d",
        help="papers 根目录路径。",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
) -> None:
    """扫描 papers 目录，找出未被 Markdown 引用的图片并清理。"""
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="WARNING")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        candidates = scan_papers(papers_dir, progress)

    confirm_and_delete(candidates, dry_run=dry_run, auto_yes=yes)


if __name__ == "__main__":
    app()
