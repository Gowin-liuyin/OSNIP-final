# src/osnip_repro/ — OSNIP 复现核心（训练与攻击的底层库）

被 `scripts/train_streaming.py`、`scripts/security_attacks_aloepri.py`（经 shared_attack 包）等调用。**不是命令行入口**。

## 模块

| 文件 | 作用 |
|---|---|
| `encoder.py` | `KeyConditionedEncryptor`：Key-conditioned 残差加密器（本项目变体 `mlp_residual_v1`：`z = iso_norm(h + 小残差(h,k))`；含多种架构备选） |
| `projection.py` | `iso_norm_project`：等范数投影（服务端收到的就是投影后的 z） |
| `attacks.py` | 论文形状的 KNN / VMA 协议实现（全词表、分块精确检索） |
| `config.py` / `curriculum.py` | 训练配置校验、约束 ramp curriculum（utility/privacy 权重调度） |
| `data.py` | 训练数据流（jsonl → token windows） |
| `checkpointing.py` | last/best 检查点与恢复 |

## 使用方法

```bash
# 通过训练脚本使用
python scripts/train_streaming.py --model-path ... --key-dim 32 --encryptor-architecture mlp_residual_v1 ...
# 单元测试
python -m pytest tests/test_decoder_recovery.py tests/test_effective_batch.py -q
```

修改 encoder/projection 会全局影响训练与攻击的口径，改完必须跑 `tests/` 并在 `docs/reproduction_issue_register_20261101.md` 留痕。
