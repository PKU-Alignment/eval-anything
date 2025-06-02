# Eval-Anything

全模态大模型评测框架

## Installation

```bash
conda create -n eval-anything python==3.11
pip install -e .
```

## Quick Start

1. 设置评测[参数文件](eval_anything/configs/evaluate.yaml)

2. 启动评测脚本

    ```bash
    bash scripts/run.sh
    ```

## Development

开发指南请见[development.md](development.md)

## Pre-commit

本项目使用 pre-commit 来确保代码质量和一致性。Pre-commit 会在每次提交前自动运行配置的检查工具。

### 安装和设置

1. 安装 pre-commit
```bash
pip install pre-commit
```

2. 在项目根目录初始化 pre-commit
```bash
pre-commit install
```

### 使用 pre-commit

每次提交代码前，pre-commit 会自动运行。也可以手动运行检查：

```bash
# 检查所有文件
pre-commit run --all-files

# 检查暂存的文件
pre-commit run
```

如果有问题需要修复，pre-commit 会阻止提交并显示错误信息。修复问题后，重新添加文件并提交：

```bash
git add .
git commit -m "MESSAGE"
git push
```

如果想跳过检查，直接commit：
```bash
git commit -m "MESSAGE" --no-verify
```

### 配置的检查工具

本项目配置了以下检查工具：

- **pre-commit-hooks**: 基本文件检查（行尾空格、文件结尾、YAML/TOML格式、大文件检查等）
- **isort**: Python 导入语句排序
- **black**: Python 代码格式化
- **pyupgrade**: 升级 Python 语法到更新版本
- **autoflake**: 移除未使用的导入和变量
- **codespell**: 拼写检查

详细配置请查看 `.pre-commit-config.yaml` 文件。
