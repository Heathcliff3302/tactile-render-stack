# InteractionState v2

`InteractionState` 是一个同步交互帧的后端无关表示。它承接 Isaac Sim、自研接触内核和后续实测数据，向材料响应层提供统一输入。

## 版本边界

Phase 1 的 `interaction-state/v1` 保持可读取。Phase 2 使用 `interaction-state/v2`，增加运行 ID、实验条件摘要、逐字段质量、严格缺失值语义和 JSON 安全序列化。v2 不改变力的单位、法向方向和面积代理的含义。

## 必须字段

| 组 | 内容 |
| --- | --- |
| 帧身份 | `run_id`、`scenario_id`、`conditions_sha256`、`source_backend`、`time_s`、`dt_s`、`sequence_id` |
| 交互对象 | `body0`、`body1`、`contact_present`、`contact_mode` |
| 几何 | 世界质心、局部质心、参考表面法向、切向基、接触点数量、Patch 数量 |
| 力 | 法向力、切向力、总力、法向冲量 |
| 运动 | 法向相对速度、切向速度、压入量、穿透量 |
| 形变 | 最大值、均值、RMS 和可选二维形变场 |
| 压力 | 均值、峰值和可选二维压力场 |
| 质量 | 每个观测量的状态、方法、单位和来源 |

## 语义规则

- `normal_world` 是参考表面的单位外法向。
- `normal_force_n` 是非负压缩力幅值。
- 向量力表示作用在 `body0` 上的力。
- `contact_area_proxy_m2` 是网格或核函数支持区域。
- `physical_contact_area_m2` 只有在后端提供明确几何依据时才填充。
- 无法获得的观测量使用 `null`，质量状态使用 `unavailable`。
- 估计量使用 `estimated`，并填写具体计算方法。
- 压力和形变二维场必须携带单位、原点、间距、切平面基和活动掩码。
- 非接触帧的接触力、冲量、压力和接触面积必须为 `null` 或零。

## 趋势比较

比较器使用 `normal_force_n`、`indentation_m`、`deformation_*` 和 `pressure_*` 等标准字段。比较前必须通过 `ExperimentManifest` 确认实验条件摘要、观测量定义、支持区域和计算方法一致。质量状态为 `unavailable` 的字段不能进入比较。

