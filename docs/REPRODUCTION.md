# OSNIP 复现操作手册

按模型检查、短程训练、正式训练和评估的顺序执行。下面的命令都从仓库根目录运行。安装及不加载模型的 smoke 见 [README](../README.md)。

## 1. 固定输入

| 输入 | 如何准备 | 要保留的记录 |
| --- | --- | --- |
| 底座 | 本地 Hugging Face 目录，含 config、tokenizer 和权重 | 模型 ID、revision、分片和 tokenizer 指纹 |
| 训练、验证与测试 | [公开源下载](../scripts/fetch_osnip_sources.py)、[拆分](../scripts/prepare_osnip_datasets.py)、[问题级无重叠重建](../scripts/prepare_question_disjoint_training.py) | 原始源版本、seed、拆分清单和每份 JSONL 的散列 |
| 独立评估 | [任务测试集](../scripts/prepare_task_test_set.py)、[KNN/VMA 集](../scripts/prepare_paper_attack_sets.py)、[WikiText-2 快照](../scripts/build_wikitext2_snapshot.py) | 样本来源、拆分、数量与散列 |
| 摘要评估 | [摘要样本准备](../scripts/prepare_summarization_dataset.py) 与[评分器检查](../scripts/verify_summary_scorer.py) | 源 test split、快照散列、评分器及其 tokenizer 版本 |

训练 JSONL 可以使用每行一个 `{"text":"..."}`。开发样本能验证训练链路；正式配方需要对应的原始输入和清单。问题级无重叠版本需要新训练，不承接旧训练状态。

## 2. 检查模型与训练链路

```bash
python scripts/validate_model_environment.py --model-path model/Llama-3.2-1B --output outputs/preflight/model.json
python scripts/run_gate1.py --model-path model/Llama-3.2-1B --train-jsonl data/train.jsonl --valid-jsonl data/valid.jsonl --output-dir outputs/gate1
```

模型检查和默认两步训练通过后，再锁定正式配方。已具备固定数据清单时，用 [run_gate0.py](../scripts/run_gate0.py) 绑定模型和输入。正式 3B 使用 [run_reproduction_3b.py](../scripts/run_reproduction_3b.py) 先生成计划，确认后再显式执行；手工研究训练入口是 [train_streaming.py](../scripts/train_streaming.py)。

## 3. 留存和比较结果

- 每个运行使用新目录。保留 `config.json`、`model_manifest.json`、`environment.json`、`source_hashes.json` 和 `report.json`。续训用包含优化器与数据游标的检查点；`encryptor.pt` 只是供推理用的编码器权重。
- 评估用 [run_evaluation_suite.py](../scripts/run_evaluation_suite.py) 将效用和隐私报告绑定到同一检查点。汇总用 [aggregate_reproduction_report.py](../scripts/aggregate_reproduction_report.py)，明确列出要比较的运行目录。
- 缺数据、候选词表缩减或评估中断时，保留真实状态和覆盖率。效用单项通过不等于隐私与效用联合通过。
- 遇到模型不匹配、数据散列不符或续训配方变化，先核对清单；不要修改散列来绕过检查。干净克隆中依赖固定外部数据快照的回归测试会跳过。
