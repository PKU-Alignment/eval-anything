
# 开发文档

## 项目简介

### 评测任务流

- `eval_anything/pipeline/base_task.py`: 实现了从读取评测参数到保存评测结果的全流程，一般不做改动

### benchmark相关

- `eval_anything/benchmark/text-to-text/gsm8k/eval.py`: 继承自`t2t_benchmark.py`中的`T2TBenchmark`基类，实现每个benchmark评测流程的适配，包括读取benchmark设置、读取数据集、模型推理、结果评测、结果保存。一般而言只需根据需要重写部分函数。

- `eval_anything/benchmark/text-to-text/gsm8k/configs.yaml`: 包含了benchmark相关的各种设置

### dataloader相关

- `eval_anything/dataloader/t2t_dataloader.py`: 继承自`base_dataloader.py`中的`BaseDataLoader`基类，实现评测数据的读取、拼接等操作

- `eval_anything/dataloader/mm_dataloader.py`: 继承自`base_dataloder.py`中的`BaseDataLoader`基类，实现评测数据的读取、拼接等操作

- `eval_anything/dataloader/format_mm_dataset.py`: 鉴于多模态数据集的数据复杂度较高，在此实现原数据集的预处理，以便后续数据加载

### inference相关

- `eval_anything/models`: 文件夹下实现了不同backend的infer

### evaluate相关

- `eval_anything/evaluate_tools/metrics.py`: 实现了常用的metric计算，若有相关需求直接使用现成的，否则请在metric中进行添加（尽量不要在benchmark中单独实现metric）

- `eval_anything/evaluate_tools/t2t_tools.py`: 实现了常用评测工具的集成，例如正则匹配、判断相同等，若有相关需求直接使用现成的，否则考虑该需求的泛用性，尽量以面向对象的形式集成进去

### 其他（`eval_anything/utils/`）

- `utils.py`: 项目开发常用工具包

- `data_type.py`: 常用数据类型，用来实现不同模块之间的数据传递

- `cache_manager.py`: cache机制设计

- `regiser.py`: 实现了常用的注册表（template, metric, benchmark...）

- `logger.py`: 实现了log机制的开发

- `mm_data_manager`: 实现了各类模态数据的格式转换与conversation编制

## 通用开发指南

1. 在`benchmarks`文件夹下进行相关文件夹创建
    - 配置参数文件 `configs.yaml`，[参考示例](eval_anything/benchmarks/text_to_text/gsm8k/configs.yaml)
    - 评测代码 `eval.py`: 继承自benchmark基类进行开发
2. 如有新的metric以及evaluate_tools请在新增的benchmark子文件夹下新增`metrics.py`和`tools.py`并分别进行register，[参考示例](eval-anything/eval_anything/benchmarks/text_to_text/TruthfulQA)
3. 在`eval_anything/configs/evaluate.yaml`文件夹下进行评测任务配置
4. 运行`eval_anything/scripts/run.sh`进行测试

## any2t数据集配置
在进行多模态数据集开发时，可能会遇到多模态数据加载方式难以统一的问题，请依照以下步骤进行数据集预处理：

1. 在`dataloader/format_mm_dataset.py`中注册数据集
2. 在`dataloader/format_mm_dataset.py`中处理benchmark-specific的键值对和Prompt构造，以便完成数据加载。尽量使用已有的PromptBuilder，如有需要，请在`_to_InferenceInput`方法中定制
3. 运用`utils/mm_data_manager.py`中的工具类实现图片、音频、视频向`base64`的转换，以便嵌入到conversation中
4. 运用`utils/mm_data_manager.py`中的工具类为此模态任务定制conversation的形式，例如，`ImageManager.prompt_to_conversation`实现了常见图文输入的conversation构造
5. 数据加载完成后，推理与评测部分与此前开发过程相同

## VLA相关

1. 配置`objaverse`资源
```
python -m objathor.dataset.download_annotations --version 2023_07_28 --path /path/to/objaverse_assets
python -m objathor.dataset.download_assets --version 2023_07_28 --path /path/to/objaverse_assets
```
2. 配置`house`资源
```
python scripts/download_objaverse_houses.py --save_dir /path/to/objaverse_houses --subset val
```
or
```
python scripts/download_objaverse_houses.py --save_dir /path/to/objaverse_houses --subset train
```

3. 数据集
```
python scripts/download_dataset.py --save_dir /path/to/dataset
```

4. 环境
```
pip install -e .[vla]
pip install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+966bd7758586e05d18f6181f459c0e90ba318bec
pip install -e "git+https://github.com/allenai/allenact.git@d055fc9d4533f086e0340fe0a838ed42c28d932e#egg=allenact&subdirectory=allenact" --no-deps
pip install -e "git+https://github.com/allenai/allenact.git@d055fc9d4533f086e0340fe0a838ed42c28d932e#egg=allenact_plugins[all]&subdirectory=allenact_plugins" --no-deps
```

## Tips

- 尽量遵循奥卡姆剃刀原则进行开发，减少对现有函数的重写
- 如有需要新增的依赖库，请在`pyproject.toml`中进行添加
- 对现有基类有任何建议欢迎提出!