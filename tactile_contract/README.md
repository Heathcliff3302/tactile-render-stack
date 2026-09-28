# Tactile Contracts

该目录保存 Phase 2 的跨后端公共契约。契约定义数据含义、单位、坐标约定、版本规则和验收边界。Isaac Sim、自研接触内核、材料模型和执行器都通过这些契约交换数据。

## 契约清单

| 契约 | 文件 | 作用 |
| --- | --- | --- |
| 逐帧公共契约 | [`InteractionState-v2.md`](InteractionState-v2.md) | 描述一个同步交互帧 |
| 实验级公共契约 | [`ExperimentManifest-v1.md`](ExperimentManifest-v1.md) | 描述一轮实验的输入、后端、观测量、产物和验收条件 |
| 实验规格契约 | [`ExperimentSpec-v1.md`](ExperimentSpec-v1.md) | 描述运行前可比较的参数、来源证据和未解析设置 |
| 迁移说明 | [`Migration-v1-to-v2.md`](Migration-v1-to-v2.md) | 说明 Phase 1 到 Phase 2 的字段变化 |
| 五层边界记录 | `pipeline.py` | 层 1 到层 5 的记录与固定顺序驱动器 |

Python 实现位于当前目录。JSON Schema 位于 `schemas/`。严格 JSON、JSONL 和非 pickle NPZ 回放接口位于 `io.py`。

`pipeline.py` 的 `MaterialResponse`、`TactileTarget`、`ActuatorCommand` 和 `ActuatorMeasurement` 是 K1 的进程内接口：它们带单位、模型标识和状态标记，但还不是带版本的 JSON 契约。它们在 K4 和 K5 升级为 Schema —— 那时才有真实材料模型和真实执行器来定义什么必须保持稳定。`FiveLayerLoop` 只按固定顺序驱动五层并记录每层状态，不计算任何物理量。

## 验收关系

实验清单绑定以下内容：

- `scenario.id` 和版本
- 物体、探头、材料和轨迹
- 时间区间、物理步长和输出步长
- 每个观测量的单位、定义、计算方法和支持区域
- 运行后端及版本
- 交互帧文件和原始后端证据的 SHA-256

只有清单与逐帧数据全部通过校验，数据才进入跨后端趋势比较。比较结果仍然表示趋势相似性，不表示求解器数值等价。
