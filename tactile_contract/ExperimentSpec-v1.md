# ExperimentSpec v1

`ExperimentSpec` 是运行前的实验规格。它回答“本次运行准备采用哪些参数、哪些值来自源代码、哪些值仍需要从后端读取”，不代替运行后的 `ExperimentManifest` 和逐帧 `InteractionState`。

## K0 的两种状态

K0 生成五份 Isaac Sim Step 6 来源清单：轨迹模式 32、64、128 网格，力模式 64 网格，多探头模式 64 网格。来源清单保留 Step 6 的命令行默认值和回归脚本覆盖值，并把未在场景构造中显式设置的引擎运行时参数标为 `unresolved`。来源清单可以追溯参数，不能单独作为跨后端数值等价的依据。

K0 同时生成 `k1_single_probe_cpu.json`。它把重力、摩擦、恢复系数、阻尼、接触偏移、求解器、迭代次数、睡眠和连续碰撞策略写成明确的设计值。该文件是单探头刚体模式的 CPU 参考输入，数值用于复现实验语义，不声称等于 Isaac Sim 的默认值。

## 字段规则

- `spec_sha256` 对除自身外的完整规格计算 SHA-256。修改任何物理、控制、采样或验收字段都必须重新生成摘要。
- `runtime.<name>.status` 为 `source_explicit`、`unresolved` 或 `design_choice`。当状态为 `unresolved` 时，`value` 必须为 `null`。
- 坐标使用米、秒、牛顿，右手系，世界 Z 轴向上，四元数顺序为 `wxyz`。
- K0 的输出步长等于控制步长。时间戳表示区间结束，序号从零开始。
- 网格是立方体顶面的正方形网格。`point_occupancy` 表示 Step 6 的点落入规则，不应解释为真实压力面积。

## 比较门槛

`spec_comparison_gate` 只做运行前资格检查。默认要求控制器、物理设置、采样定义、网格和时间步长完全一致。`grid_refinement` 允许只改变网格，`time_refinement` 允许只改变时间设置。任何未解析运行时参数都会使比较失去资格。

