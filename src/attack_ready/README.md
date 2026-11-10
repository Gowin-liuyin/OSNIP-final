# src/attack_ready/ — 运行时观测钩子

`observe.py`：在推理/评估时挂载观测，采集攻击所需的两类可观测数据：

1. **wire 嵌入（z）**：客户端加密后真正发给服务器的向量（KNN/ISA/IMA 的攻击输入）；
2. **隐态（hidden states）**：指定层的中间表示（ISA 隐态攻击输入）。

## 使用方法

```bash
# 一般由评估脚本内部调用；单独使用时：
python - <<'PY'
from src.attack_ready.observe import ...  # 见文件内 API
PY
python -m pytest tests/test_attack_ready.py -q
```

注意：观测口径必须与威胁模型一致——`observe` 只暴露服务器可见的张量，密钥/明文回读一律不进观测（见 `docs/experiment_plan_20261106.md` §8）。
