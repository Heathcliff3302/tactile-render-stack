# Layer 3 Mechanics

该目录存放低阶材料响应模型、材料独立状态，以及 K1 实验装配。模型接收 `InteractionState`，输出独立的 `MaterialResponse`。Isaac Sim 不作为该层的运行时依赖。

| 文件 | 作用 |
| --- | --- |
| `material.py` | 刚体基线材料响应（K1） |
| `k1_rigid.py` | K1 实验装配：两遍运行、条件构造、五层闭环 |
| `k1_report.py` | 运行目录写入、产物哈希、清单与运行索引 |
| `run_k1.py` | 命令行入口 |

K1 的层 3 是透传基线。刚体冲量模式下内核约束是该接触对该步唯一的权威求解器，层 3 再算一次力就会把同一次交互计入两遍，因此响应标为 `passthrough`，刚度与阻尼为空值，弹性字段不可用。

Kelvin–Voigt、Standard Linear Solid、Hertz 和弹性地基模型在 K4 引入，并自带接触面积与压力场。它们是第一批可以把压力标为由物理面积导出的模型。

运行方式：

```text
cd "Phase 2"
.venv/Scripts/python.exe -m mechanics.run_k1
```

产物写入 `experiments/runs/<run_id>/`，入库索引写入 `experiments/index/`。
