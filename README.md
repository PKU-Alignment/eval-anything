# eval-anything

## 代码结构
```
eval-anything/
├── eval-anything/
│   ├── benchmarks/                     # 各benchmark的评测代码
│   │   ├── text_image_to_text/         # 以模态作为一级分类
│   │   │   └── mmmu/                   # benchmark子文件夹
│   │   │       ├── mmmu.yaml           # 存储评测集的基本信息，包括数据集路径、size、评测方式、key值...
│   │   │       └── eval.py             # 评测代码，继承自eval-anything/pipeline下的基类
│   │   ├── text_to_text/
│   │   │   └── gsm8k/
│   │   │       ├── gsm8k.yaml
│   │   │       └── eval.py
│   │   └── text_vision_to_action/
│   │       └── chores/
│   │           ├── configs.yaml
│   │           └── eval.py
│   ├── configs/                        # 评测参数文件夹
│   │   └── evaluate.yaml               # 包含了推理参数（top-k, temperature...）、评测参数(cot, few-shot...)
│   ├── dataloader/                     # 数据集加载与预处理
│   │   ├── base_dataloader.py          # dataloader基类，继承使用
│   │   ├── mm_dataloader.py            # 多模态dataloader，支持多模态数据的本地和在线读取，支持数据预处理
│   │   ├── t2t_dataloader.py           # 纯文本dataloader，支持数据加载和预处理
│   │   └── tv2act_dataloader.py        # VLAdataloader，加载测试数据集和测试环境
│   ├── models/                         # 各模型推理的代码，包含在线api、hf、vllm以及多模态生成模型各自的生成代码
│   │   ├── base_model.py
│   │   ├── api.py                      # 在线api访问
│   │   ├── hf_lm.py                    # 基于transformers的t2t推理
│   │   ├── hf_mm.py                    # 基于transformers的多模态推理
│   │   ├── vllm_lm.py                  # 基于vllm的t2t推理
│   │   └── vllm_mm.py                  # 基于vllm的多模态推理推理
│   ├── pipeline/                       # 完整评测pipeline代码
│   │   ├── base_task.py                # pipeline基类，一般直接使用
│   │   ├── base_benchmark.py           # benchmark基类，继承使用
│   │   ├── mm_gen_benchmark.py         # 多模态生成任务benchmark
│   │   ├── mm_und_benchmark.py         # 多模态理解任务benchmark
│   │   └── t2t_benchmark.py            # t2t任务benchmark
│   │   └── tv2act_benchmark.py         # tv2act任务benchmark
│   ├── evaluate_tools/                 # 评测过程中常见的工具包，例如计算各种指标、pattern匹配等
│   │   ├── base_tools.py               # 评测指标和评测工具的基类
│   │   ├── metrics.py                  # 各种评测指标的计算 
│   │   └── t2t_tools.py                # 评测工具
│   └── utils/                          # 项目中使用到的工具包
│       ├── cache_manager.py            # 推理过程的cache机制
│       ├── data_type.py                # 定义数据基类，用于存储模型输入数据和评测结果数据
│       ├── register.py                 # 注册器类
│       ├── logger.py                   # 评测logger，用于输出评测结果以及写入本地log
│       ├── utils.py                    # 框架开发时使用的通用工具
│       └── template.py                 # chat_template
├── scripts/                        # 评测启动脚本文件夹
│   └── run.sh                      # 评测启动脚本
├── third_party/                    # 第三方VLA模型代码
|   └── SPOC
├── setup.py
└── README.md
```

## Installation

```bash
conda create -n eval-anything python==3.11
pip install -e .
```

## 开发指南
- [ ] 从旧框架中剥离出dataloader, inferencer进行适当复用，但是需要根据多模态特性进行更完善的基类开发
    - [x] T2T
    - [ ] 多模态
    - [ ] **$New$** 在dataloader中支持多轮对话数据的拼接
- [x] 复用旧框架中的data_type, logger, template
- [ ] **$New$** 设计更加完善的cache机制，保存中间推理和评测结果
    - [x] T2T
    - [ ] 多模态
- [x] **$New$** 参考lm_eval重新开发pipeline基类，后续的benchmark适配（eval.py）完全以面向对象的形式进行开发
- [x] **$New$** 开发evaluator.py（面向对象），实现常用评测指标的计算。例如win rate, pass rate, F1-Score, ASR, SD, word error rate (WER) (↓), BLEU(↑),  automatic speech recognition (ASR), perplexity
- [x] 对传参体系进行重新设计
    - [x] **$New$** 参考lm_eval，构建数据集的yaml文件
    - [x] 重新设计传递评测config的yaml文件
- [ ] 整理不在hf上的数据集，统一键值
- [ ] 在eval-anything/models底下进行各种多模态生成模型的推理单独适配（非紧急，但必要）

## 期望的Features

- 多节点自动化评测，给机器、任务、模型的list，自动分配机器进行评测
- 优化传参体系
    - 以yaml为一次评测的传参接口，在启动评测时既能支持bash传参，也能通过yaml传参
    - 支持同时指定多个yaml，实现每个benchmark细粒度的调参
    - 保存evaluation结果时，还同时能保存评测参数（yaml+命令行参数），可以参考align-anything
- 若有多个benchmark串行评测，不重复load模型；同一个benchmark不同task评测时，不重复infer
- 对于多模态评测，有一个直观展示评测结果（过程）的方式，类似gradio
- docker评测