# OSNIP-final：独立训练与评估

这个仓库独立复现 OSNIP 的训练和评估。语言模型的权重保持不变；客户端根据密钥改变传给模型的词向量，再比较改写前后的模型表现和信息泄露。编码器是本项目的工程实现。只想检查安装时，先跑下面的合成测试，不需要下载模型。

## 安装和快速验证

建议在独立 Python 环境中安装。GPU 训练需准备与本机驱动匹配的 PyTorch。版本约束见 [requirements.txt](requirements.txt)。

```bash
python -m venv .venv
# 激活虚拟环境后执行
python -m pip install -r requirements.txt
python scripts/run_standalone_smoke.py --steps 2
python -m pytest -q
```

smoke 用合成向量测试梯度、密钥敏感性和投影误差，写入 `outputs/standalone-smoke/report.json`，无需下载底座。依赖原始数据或历史快照的测试需要另行准备对应工件。

## 准备模型和数据

1. 准备匹配配置的 Hugging Face 格式底座（配置、tokenizer、权重），保留准确的模型 revision。
2. 使用 `python scripts/fetch_osnip_sources.py --download-alpaca --output-dir data/osnip_sources` 下载公开数据源。以生成的 manifest 为准，记录 Alpaca、ARC、SciQ 和 WikiText 各自的 revision。
3. 用 `prepare_osnip_datasets.py` 生成训练、验证、测试及 KNN 预留集；传入各输入文件和对应的 `--revision SOURCE=REVISION`。用 `prepare_paper_attack_sets.py` 生成独立评估输入，用 `prepare_task_test_set.py` 生成任务测试集。各脚本的必填参数见 `--help`。
4. 问题级无重叠的 1B 路线另用 `prepare_confirmatory_holdout.py` 和 `prepare_question_disjoint_training.py`。该数据版本需要从新编码器开始训练，不应复用旧优化器、旧检查点或数据游标。

## 训练、续训和评估

3B 入口默认只检查输入并生成执行计划；确认模型、数据、配置和显存后，可加 `--execute`：

```bash
python scripts/run_reproduction_3b.py --model-path model/Llama-3.2-3B-Instruct --data-root . --output-dir outputs/llama32-3b
```

数据路径由选定配置决定，`--data-root` 指向包含 `data/` 的目录。1B 主线可依次使用 `validate_reproduction_inputs.py`、`run_gate0.py`、`run_gate1.py` 和 `train_streaming.py`；续训使用 `run_osnip_continuation.py`。历史配置中的模型位置是原实验记录，新环境通过命令行显式传入本地模型路径。

训练尽量保持模型输出，同时拉开原始向量与扰动向量的方向，并让不同密钥产生不同表示。投影步骤保持每个向量的长度。正式评估用 `run_evaluation_suite.py` 将 PPL、任务效用和隐私指标绑定到同一检查点，再用 `aggregate_reproduction_report.py` 汇总。记录模型、输入、检查点的散列和评估范围；短程诊断和缩小词表的结果应单列。训练完成或单项效用达标，都不能证明隐私与效用已联合复现。

本仓库保存源码、配置和测试。数据、底座、训练权重和报告由操作者本地管理。A 组件见 [AloePri-final](https://github.com/Gowin-liuyin/AloePri-final)，组合运行时见 A+B 仓库。

## 代码在哪里

| 路径 | 用途 |
| --- | --- |
| `src/osnip_repro/` | 编码器、等范数投影、训练目标、课程调度、数据与检查点 |
| `src/attack_ready/` | 四组观测的合成诊断；其中的 A+B 噪声代理不等同于真实部署 |
| `configs/` | 1B、3B、消融及评估配方 |
| `scripts/` | 数据准备、输入校验、训练、续训、评估和报告汇总 |
| `tests/` | 数学性质、梯度、输入契约和实验流水线的回归测试 |

## 更多文档

- [文档导航](docs/README.md)
- [复现操作手册](docs/REPRODUCTION.md)
- [维护与贡献](CONTRIBUTING.md)
- [变更记录](CHANGELOG.md) · [引用与来源](CITATION.md) · [安全问题报告](SECURITY.md)
