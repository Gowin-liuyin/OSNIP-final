# 四组向量观测辅助模块

[observe.py](observe.py) 用相同的输入 token 位置构造四种向量观测：明文、加噪的 AloePri 代理、OSNIP 投影，以及先投影再加噪的组合代理。OSNIP 路线需要提供编码器；这里不加载服务器模型，也不采集中间层隐态。

组合代理用于测试观测口径，不对应 A+B 仓库实际部署协议。验证入口：

```bash
python -m pytest tests/test_attack_ready.py -q
```
