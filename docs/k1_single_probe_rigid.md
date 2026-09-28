# K1 单探头刚体接触内核

K1 已实现：一个不依赖 Isaac Sim 的 CPU 参考内核，复现 Phase 1 Step 6 的单探头轨迹、接触、冲量、力和区域输出，并把同一条数据流接入五层闭环。Unity 可以作为场景显示与交互输入前端，但物理状态只能由一个内核负责。Unity 的 3D 物理系统本身集成 NVIDIA PhysX，因此前端接入必须保留后端标识和参数快照，避免把显示前端误当成第二个求解器。[Unity physics overview](https://docs.unity3d.com/6000.0/Documentation/Manual/PhysicsOverview.html)

## 代码位置

| 职责 | 位置 | 依赖 |
| --- | --- | --- |
| 接触内核（解耦） | [`contact_kernel/`](../contact_kernel) | 仅标准库 |
| 五层边界记录与驱动 | [`tactile_contract/pipeline.py`](../tactile_contract/pipeline.py) | 无 |
| 层 1 世界与轨迹状态机 | [`virtual_reality/`](../virtual_reality) | `contact_kernel` |
| 层 2 `InteractionState` 适配 | [`interaction/kernel_backend.py`](../interaction/kernel_backend.py) | `tactile_contract` |
| 层 3 刚体材料基线 | [`mechanics/material.py`](../mechanics/material.py) | `tactile_contract` |
| 层 4 触觉目标 | [`std_tactile/target.py`](../std_tactile/target.py) | `tactile_contract` |
| 层 5 单轴执行器 | [`actuators/single_axis.py`](../actuators/single_axis.py) | `tactile_contract` |
| K1 实验装配与产物 | [`mechanics/k1_rigid.py`](../mechanics/k1_rigid.py)、[`mechanics/k1_report.py`](../mechanics/k1_report.py) | 全部 |
| 验收门槛 | [`validation/k1_acceptance.py`](../validation/k1_acceptance.py) | `tactile_contract` |
| 运行时字段裁决 | [`virtual_reality/runtime_policy.py`](../virtual_reality/runtime_policy.py) | `contact_kernel` |

`contact_kernel` 不导入仿真器、不导入契约包、不导入任何层模块。规格到内核配置的转换在层 1 的适配器里完成，内核帧到契约记录的转换在层 2 完成。

运行方式：

```text
cd "Phase 2"
.venv/Scripts/python.exe -m mechanics.run_k1
```

## 冻结输入

K1 读取 [`k1_single_probe_cpu.json`](../scenarios/k0/k1_single_probe_cpu.json)，不修改它。立方体中心 `[0, 0, 0.25] m`，尺寸 `[0.5, 0.5, 0.5] m`，顶面 `z=0.5 m`；球半径 `0.12 m`，质量 `0.05 kg`，初始位置 `[0, 0, 1.2] m`，接触高度 `z=0.62 m`。控制步长与输出步长均为 `1/60 s`，探头角速度锁定，世界 Z 轴为接触法向。

CPU 参考值显式设定为重力 `[0, 0, -9.81] m/s²`、静摩擦和动摩擦为零、恢复系数为零、线性和角阻尼为零、接触偏移和恢复偏移为零、最大去穿透速度为零、关闭睡眠。这些是 `design_choice`，不是对未解析 PhysX 默认值的断言。

## 内核每步流程

1. 层 1 读取上一帧反馈，生成本控制步的目标速度，并记录控制步序号、区间起止时间和命令来源。
2. 驱动层按一次控制步覆盖线速度，随后在每个子步做重力前馈补偿。
3. 每个子步求解法向间隙 `g = z_probe - (z_surface + r)`，按位置级非穿透约束给出目标法向速度 `v_target = -g/h`，并用声明的去穿透速度上限截断。
4. `J_n = m * max(0, v_target - v_free_n)`。零摩擦下切向速度保持命令值，法向与切向冲量分开记录。
5. 数值穿透用几何投影消除，投影量单独记入 `position_correction_m` 和 `penetration_m`，不折算成额外接触力。
6. 输出区间内的冲量之和除以输出步长得到帧平均力。逐帧记录接触点、法向、相对速度、间隙、冲量、力和子步诊断。
7. 接触点投影到顶面局部坐标，执行边界策略和网格映射，构建 Point → Cell → Patch → Episode。

接触判定完全来自几何，不由运动阶段门控：`pre_contact` 阶段之所以没有接触，是因为探头确实在 `z=1.2 m`。

PhysX 的持久接触缓存会复用和升级上一帧接触，接触点数量不表示压力密度，因此 K1 不把接触点个数当作接触面积或压力。[PhysX advanced collision detection](https://nvidia-omniverse.github.io/PhysX/physx/5.1.3/docs/AdvancedCollisionDetection.html)

## 与 Step 6 的两处显式差异

**重力补偿位置。** Step 6 把 `gravity * physics_dt` 折进速度命令。K1 让控制器只发目标速度，由驱动层在每个子步做补偿（`gravity_compensation_mode: per_substep_feedforward`），数值仍取自声明的 `gravity_compensation_mps2 = 9.81`。原因是子步细化：命令只在控制步开头写一次，如果补偿量按整个控制区间计算，子步数一变，区间内的重力残差就会改变轨迹和接触时序。按子步补偿后，位移和冲量对子步数不变。

这一选择决定了力的量级，必须明说：保持阶段的法向力等于 `m * contact_maintain_speed / output_dt = 0.05 * 0.02 * 60 = 0.06 N`，来源是"每控制步覆盖一次速度"注入的动量，不是探头自重。改变控制频率会改变这个力。该结论与设计文档中"`set_linear_velocity` 语义会影响力与时间步长的关系"一致，记录在 `effective_runtime.json` 的 `drive.force_scale_note`。

**`released` 仍推进声明的释放时间。** 早期方案写的是"释放后禁止继续推进物理时间"。实现改为沿用 Step 6：`released` 段按 `release_time_s = 0.5 s` 继续出帧。没有这 30 帧，就没有"释放后接触力清零"的证据可看。这些帧的 `normal_force_n`、`normal_impulse_ns`、`contact_area_proxy_m2` 全为 `null`，`total_force_world_n` 为零向量。

## 区域解算

三层区域结果在同一帧同时输出（`metadata.area_layers`）：

1. `point_occupancy`：接触点所在网格单元获得一个占用标记，完全复现 Step 6 的离散基线。`contact_area_proxy_m2` 取这一层。
2. `swept_path`：按相邻帧接触点连线做超覆盖栅格化，补齐高速滑动时的漏格；它是轨迹覆盖，不是瞬时面积。
3. `geometric_footprint`：未变形几何的截交半径 `sqrt(2Rδ-δ²)`。刚体非穿透接触的 `δ = 0`，因此这一层恒为 `0`，这正是"点占格面积不能替代弹性接触面积"的直接证据。
4. `model_footprint_m2` 恒为 `null`，直到 K4 的 Hertz 与弹性地基模型给出模型接触面积。

`physical_contact_area_m2` 在 K1 全程为 `unavailable`：刚体点接触没有物理接触面积，写零会被下游当成真实测量。

压力字段按占格代理输出，状态只能是 `estimated`，方法名写作 `grid_occupancy_proxy_pressure`，并在 `metadata.pressure_semantics` 说明它随输出分辨率变化、不是物理接触压力。`InteractionState v2` 的 `status` 枚举没有 `proxy` 取值，因此代理语义由方法名和 `pressure_semantics` 承担。

网格覆盖计算 `disc_coverage` 已实现（子单元积分，带收敛测试），K1 运行中因 `δ = 0` 不产生非零覆盖，它服务于 K4。

## 采样阈值的可见性

`force_threshold_n = 0.05 N` 是声明的采样阈值。首次几何接触那一帧的冲量可能低于阈值（接近过程中探头刚好落到接触高度，冲量为零）。这种帧不报告接触，但会记录 `solver_contact = true`、`sampling_filter_reason = normal_force_below_sampling_threshold`，以及未过滤的 `solver_normal_impulse_ns`。求解器做了什么和采样报告了什么，两件事分开可见。

探头在区间中途滑出顶面的情况记为 `support_point_left_top_face_during_interval`：冲量确实施加过，但区间末的接触点已不在面上，因此不报告接触。边缘接触与侧面接触不在 K1 范围内。

## 两遍运行与条件哈希

`ExperimentManifest.conditions` 必须先于逐帧数据存在（每帧都带 `conditions_sha256`），但轨迹模式的命令时序取决于接触何时发生，力控命令更是反馈产物。K1 因此跑两遍：

- 第一遍**实现**：记录控制器真正发出的命令和探头位姿。
- 构造 `conditions`，其中 `trajectory.kind = realised_command_velocity_and_probe_pose`，明确标注这是实现结果而不是预先已知的计划。
- 第二遍**发布**：用 `digest(conditions)` 重跑并盖章。两遍的命令序列必须逐项相同，这同时就是确定性回放的证据。

## 验收结果

`experiments/index/k1_single_probe_cpu.json` 是提交入库的运行索引，完整产物写入被忽略的 `experiments/runs/<run_id>/`。

| 门槛 | 结果 |
| --- | --- |
| 运行完成所声明的程序 | 通过，`loop_state = completed`，无中止原因，段序列完整 |
| 单探头接触出现 | 通过，121 个接触帧，峰值 `0.6 N`，`patch_id = 1` |
| 保持阶段接触稳定 | 通过，保持段 60 帧覆盖率 `1.0`，切向速度 `0.0 m/s` |
| 受控段接触覆盖率达标 | 通过，hold+slide 120 帧覆盖率 `1.0`（门槛 `0.95`） |
| 滑动速度达标 | 通过，合格样本占比 `1.0`（门槛 `0.90`，容差 `±0.01 m/s`） |
| 释放后接触力清零 | 通过，`released` 段 30 帧接触帧数 0，最大总力 `0.0 N` |
| 总力与冲量一致 | 通过，冲量换算、法向加切向、Point→Cell→Patch 误差均为 `0.0 N`（门槛 `1e-5 N`） |
| 时间步变化不改变数据语义 | 通过，子步 1/2/4 的段序列、接触序列、Cell 与 Patch 序列完全相同；力差 `1.1e-16 N`，高度差 `2.7e-14 m` |
| `InteractionState` 完整回放 | 通过，14 个产物哈希校验 + `validate_run` 472 帧 |

### 阈值台账

前四项里有三项是冻结前补上的。第一版的 `all_passed` 只聚合"已发出的门槛"，于是两种真实失败被报成通过：在释放段中途耗尽步数预算的运行（`loop_state = aborted`，但接触、保持、清零三项都满足），以及滑动全程未达目标速度的运行（`slide_speed_pass_fraction = 0.0`）。指标算了、对着阈值打印了、但没有人比较。

补门槛只修好今天。结构性修复是 `ACCEPTANCE_KEY_GATES` 与 `DEFERRED_ACCEPTANCE_KEYS` 两张台账：规格 `acceptance` 块的每个键，要么被某个门槛消费，要么是被显式声明的延后项；有键落在两者之外时 `acceptance_key_ledger` 直接报错。`patch_coverage_min` 与 `resolution_force_spread_max` 登记为 K3 延后，`slide_settle_time_s` 现在被真正消费（滑动统计跳过起始沉降窗口）。K2–K5 每新增一个阈值都会被这张台账挡住。

`all_passed` 还要求 `RUN_GATE_ORDER` 中的每一项都已发出：缺失的门槛不能被读成通过。

段长度与声明参数一致：`pre_contact` 30、`approach` 175、`hold` 60、`lateral_slide` 60、`retract` 117、`released` 30，共 472 帧、7.867 s 仿真时间。

第 5 项的"语义"指段序列、接触与释放序列、帧数、Cell 与 Patch 身份、守恒等式。力的**量级**在一般情况下不声明与步长无关（它依赖驱动语义）；本次驱动语义下恰好也不变，因此报告实测离散而不是假定为零。

第 6 项在清单写完之后验证并单独写入 `replay_check.json`，不作为清单产物：哈希绑定的文件不能包含关于自身哈希的已验证结论。

## 运行时字段裁决

规格可以声明一个求解器从不读取的字段。记录它对溯源和后续参数比较有用，但绝不能看起来像是生效了。`virtual_reality/runtime_policy.py` 给 `runtime` 块的每个字段一个裁决，值不符合时在加载期拒绝——报告里的一行备注拦不住一次错误运行。

| 裁决 | 含义 | 字段 |
| --- | --- | --- |
| `enforced` | 内核读取，不同取值改变行为 | `gravity_mps2`、`linear_damping_per_s`、`rest_offset_m`、`max_depenetration_velocity_mps`、`initial_linear_velocity_mps`、`sampling.kinematics_source` |
| `recorded_inert` | 内核不读取，且对本求解器结构确实是空操作；只接受那个惰性取值，并在加载期核验 | `position_iterations`(1)、`velocity_iterations`(1)、`sleep_enabled`(false)、`angular_damping_per_s`(0，转动已锁定)、`initial_orientation_wxyz`(单位四元数)、两个 combine_mode（因摩擦与恢复系数为零而无效） |
| `single_implementation` | 字段指向真实的物理或数值选择，但只实现了一个取值，其余拒绝 | `static_friction`(0)、`dynamic_friction`(0)、`restitution`(0)、`contact_offset_m`(0)、`solver_type`、`ccd_mode` |

这套裁决与规格自带的 `status`（`source_explicit` / `unresolved` / `design_choice`）正交：后者说值从哪来，前者说内核是否据此行动。合并会丢信息。分类表写入 `effective_runtime.json` 的 `field_policy`，读者不必读求解器就能知道哪些字段真的在起作用。表中缺少任何一个已声明字段本身就是错误——未分类的设置等于没人检查过。

`contact_offset_m` 之所以只接受 0：内核用的是逐子步的预测式间隙检测，不是 PhysX 那种投机接触余量；非零偏移会改变接触起始时刻，而这没有实现。

### 两种运动学来源都已实现

`kinematics_source` 原先接受 `finite_difference` 却硬编码 `rigid_body_twist`，规格说一套、数据是另一套。现已按声明取值选择权威来源：

- `rigid_body` → 方法名 `rigid_body_twist`，探头刚体 twist（转动锁定时即其线速度）。
- `finite_difference` → 方法名 `contact_point_finite_difference`，接触点跨帧差分。

未被选中的那一个仍逐点保留为交叉校验（`rigid_body_tangent_speed_mps` 与 `finite_difference_*`），参数比较可以直接看到两者差异。差分来源在没有接触点的帧上无值，因此那些帧的 `tangent_velocity_world_mps` 和 `normal_relative_velocity_mps` 标为 `unavailable` 并退出 `required_observables`，而不是填零。接触第一帧没有前驱，同样标为不可用。

提前做这项的原因：Phase 1 `step6_3.py` 的 argparse 默认值就是 `finite_difference`，只有回归脚本覆盖成 `rigid_body`。接入 Isaac Sim 做参数一致性比较时两个取值都会用到。

## 五层状态机

驱动器 `FiveLayerLoop` 的生命周期为 `initialized → running → completed | aborted(reason)`，只有层 1 能结束运行且必须说明原因。每个控制步按固定顺序穿过五层，任何一层都必须产出记录：缺失会抛异常，不会静默跳过。层 1 的运动状态机沿用 Step 6 的六段顺序，`released` 为终止段。

各层在 K1 的诚实边界：

- 层 3 是**透传**基线。刚体冲量模式下内核约束是该接触对该步唯一的权威动力学求解器，层 3 再算一次力就会把同一次交互计入两遍。因此 `status = passthrough`，`stiffness_n_per_m` 与 `damping_ns_per_m` 为 `null`，弹性字段全部不可用。
- 层 4 只声明 `normal_force` 一个通道。振动与温度字段为 `null` 而不是零，因为 K1 的材料基线既不建模瞬态也不建模热流。
- 层 5 没有接硬件。`measured_force_n` 由声明的传递模型产生，`model_id = single_axis_ideal_first_order_v0`、`status = modelled_not_measured`。命令路径、逆模型、饱和、延迟和看门狗都是真实代码和真实记录，K5 只需替换传递模型。K1 唯一可见的误差是声明的一步延迟。

层 3、层 4、层 5 的记录是 K1 的进程内接口，带单位、模型标识和状态标记，但还不是冻结契约。它们在 K4 和 K5 成为带版本的 Schema —— 那时才有真实材料模型和真实执行器来定义什么必须保持稳定。

## K1 未做的事

- 摩擦、恢复系数、滚动、任意网格、堆叠、自碰撞。
- 平底长方体探头、多探头、多物体对独立流、32/64/128 网格回归（K3）。
- 力控模式与 1 N 调节（K2）。
- Hertz、弹性地基、压力场、模型接触面积、材料单元独立状态（K4）。
- Unity 前端与执行器标定（K5）。
- 与 Isaac Sim 的数值比较。未解析的 PhysX 设置采集完成之前，只有行为与守恒门槛，没有数值等价结论。
- `contact_offset_m`、摩擦、恢复系数、多次求解迭代、睡眠和其他 CCD 模式：声明为不支持并拒绝，而不是近似。

## 后续扩展顺序

球面摩擦 → 盒形探头 → 移动平面 → 多接触点 → 材料法向响应。非线性法向力可增加 Hertz 或弹性基础模型，参数必须从材料层输入，并与刚体冲量模式分开标识。Hertz 接触模型适合表达球面与平面接触的非线性法向响应。[Itasca Hertz contact model](https://docs.itascacg.com/itasca900/common/contactmodel/hertz/doc/manual/cmhertz.html)

分布式压力输出可以参考 hydroelastic 的接口思想。它提供压力场近似，不等同于有形变自由度的 FEM，因此只作为 `patch` 层的可选后端。[Drake hydroelastic contact guide](https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html)

若接入 Unity 的手动模拟，必须使用固定步长调用 `Physics.Simulate`，否则同一输入序列不能形成可比较的确定性回放。[Unity Physics.Simulate](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/Physics.Simulate.html)
