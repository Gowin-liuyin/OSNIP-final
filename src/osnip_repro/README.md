# OSNIP 核心模块

此目录提供 [流式训练](../../scripts/train_streaming.py)、[合成检查](../../scripts/run_standalone_smoke.py) 和评估脚本共用的库。

| 路径 | 职责 |
| --- | --- |
| [encoder.py](encoder.py)、[projection.py](projection.py) | 密钥条件编码器与等范数投影 |
| [losses.py](losses.py)、[curriculum.py](curriculum.py) | 效用、隐私、密钥差异损失及训练调度 |
| [data.py](data.py)、[modeling.py](modeling.py) | JSONL、分词、冻结模型及前向适配 |
| [checkpointing.py](checkpointing.py)、[validation.py](validation.py) | 保存、续训配方校验和验证指标 |
| [attacks.py](attacks.py)、[vma.py](vma.py) | 受控隐私评估中的检索和观测计算 |

数据经分词进入冻结模型的嵌入表；客户端编码并投影，效用损失比较干净与扰动路径，隐私和多样性约束在同一次训练计算。真实复现入口与报告要求见 [操作手册](../../docs/REPRODUCTION.md)。
