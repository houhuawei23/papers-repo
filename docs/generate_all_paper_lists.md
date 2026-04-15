# 批量生成 `papers/*/List.md` 使用说明

## 目标

`scripts/generate_all_paper_lists.py` 用于批量扫描 `papers` 下各个分组目录，并自动生成每个分组的可跳转索引文件 `List.md`。

它会复用单目录脚本 `scripts/generate_paper_list.py` 的解析能力（包括 `paper.yml` 元信息提取）。

## 快速开始

在仓库根目录执行：

```bash
python scripts/generate_all_paper_lists.py
```

默认行为：

- 输入目录：`papers`
- 每个分组输出：`<group>/List.md`
- 分组内排序：按日期前缀 `date` 升序

## 常用命令

### 1) 按时间降序生成（推荐）

```bash
python scripts/generate_all_paper_lists.py --descending
```

### 2) 仅生成指定分组

```bash
python scripts/generate_all_paper_lists.py --include VLA Diffusion-Model --descending
```

### 3) 排除某些分组

```bash
python scripts/generate_all_paper_lists.py --exclude Legacy Archive
```

### 4) 仅预览（不写入）

```bash
python scripts/generate_all_paper_lists.py --dry-run --verbose
```

### 5) 调整并发

```bash
python scripts/generate_all_paper_lists.py --group-workers 6 --scan-workers 12
```

## 参数说明

- `--papers-root`：papers 根目录，默认 `papers`
- `--output-name`：每个分组输出文件名，默认 `List.md`
- `--sort-by`：分组内排序字段，可选 `date | year | title | dir`
- `--descending`：分组内降序
- `--scan-workers`：单分组内部扫描并发数
- `--group-workers`：分组级并发数（多个分组并行处理）
- `--include`：只处理指定分组
- `--exclude`：排除指定分组
- `--dry-run`：仅演练，不写文件
- `--verbose`：打印详细日志

## 输出与返回码

- 成功时会在每个分组目录生成（或覆盖）`List.md`
- 终端会打印每个分组处理结果与总汇总
- 退出码：
  - `0`：全部成功
  - `2`：部分或全部分组失败

## 建议流程

1. 先运行 `--dry-run --verbose` 检查范围
2. 再正式执行批量生成
3. 最后 `git diff` 检查索引变更是否符合预期
