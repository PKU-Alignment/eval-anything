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

## To-do list

[ ] 清点所有数据集并写相应的支持文档

[ ] pre-commit 等格式清洗

[ ] 包装首页 Readme
