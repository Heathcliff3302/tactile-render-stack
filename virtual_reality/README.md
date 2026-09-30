# Layer 1 Virtual Reality

该目录存放世界状态、轨迹输入和固定仿真时钟。Isaac Sim 作为参考后端保留，层 1 的输出通过层 2 转换为 `InteractionState`。

层 1 同时是接触内核实例的持有者：一个接触对每一步只能有一个权威动力学所有者，层 2 读取并格式化内核结果，不自行推进物理。

| 文件 | 作用 |
| --- | --- |
| `controllers.py` | Step 6 六段运动状态机，含滑动切向速度伺服 |
| `world.py` | `ExperimentSpec` 到 `KernelConfig` 的适配器，以及层 1 stage |
| `runtime_policy.py` | 运行时字段三分类裁决，不支持的取值在加载期拒绝 |
| `drive_limits.py` | 驱动语义下的力可达性推导与加载期校验 |

## 字段裁决

规格可以声明一个求解器从不读取的字段。记录它对溯源有用，但绝不能看起来像是生效了。每个 `runtime` 字段带一个裁决：`enforced`（内核读取且行为随之变）、`recorded_inert`（内核不读，且已核验取值确实是空操作）、`single_implementation`（只实现了一个取值，其余拒绝）。裁决与规格自带的 `status` 正交：后者说值从哪来，前者说内核是否据此行动。分类表写入 `effective_runtime.json` 的 `field_policy`。

控制器只发目标速度。重力补偿由驱动层按子步前馈完成，数值取自声明的 `gravity_compensation_mps2`。这是相对 Step 6 的唯一一处搬迁，理由与后果见 [`docs/k1_single_probe_rigid.md`](../docs/k1_single_probe_rigid.md)。
