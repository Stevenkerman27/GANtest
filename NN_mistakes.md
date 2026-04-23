1. GAN中真假样本数量不一致时，计算梯度惩罚 (Gradient Penalty) 必须确保条件 (Condition) 与样本索引严格对齐。在 Phase 2 物理约束训练中，如果只对不合理样本 (F_eps) 计算 GP，不能简单地对条件张量进行切片 (如 `conds[:min_size]`)，否则会导致样本特征与物理条件错位。

**错误做法：**
```python
# 导致索引错位：f_foils 是从原始 batch 中挑出来的，但 conds[:n] 取的是 batch 前 n 个
gradient_penalty = compute_gradient_penalty(D, real, f_foils, conds[:f_foils.size(0)])
```

**正确做法：**
使用对应的索引 (f_idx) 显式选取真样本和条件，确保每一对插值都在相同的物理条件下进行。
```python
gradient_penalty = compute_gradient_penalty(D, foils[f_idx], f_foils, conds[f_idx])
```
