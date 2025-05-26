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

开发指南请见[development.md](eval-anything/development.md)