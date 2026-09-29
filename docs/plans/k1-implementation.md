# K1 单探头刚体实现计划

## 已完成

- [x] 把接触内核解耦到 [`contact_kernel/`](../../contact_kernel)，只依赖标准库；规格转换放在层 1，契约转换放在层 2。
- [x] 从 K0 CPU 参考规格读取固定表面、单球探头、时间步长和控制参数，不修改 K0 输入。
- [x] 实现半隐式积分、子步、球面与有限平面顶面的位置级解析法向冲量，以及与冲量分离的穿透修正记录。
- [x] 实现 `pre_contact`、`approach`、`hold`、`lateral_slide`、`retract` 五段运动状态机，以及 `released` 终止段。
- [x] 输出符合 `InteractionState v2` 的逐帧记录，每个观测量在整轮运行中保持单一计算方法。
- [x] 输出 Point、Cell、Patch、Episode，含稳定 ID、分裂合并事件和力守恒指標。
- [x] 同帧输出 `point_occupancy`、`swept_path`、`geometric_footprint` 三层面积，模型面积保留为空值，等待 K4。
- [x] 实现 `FiveLayerLoop` 五层状态机与层 3、层 4、层 5 的 K1 基线实现。
- [x] 两遍运行（实现 + 发布），条件哈希绑定实现轨迹，同时作为确定性证据。
- [x] 写出规格声明的全部运行产物，并由清单哈希绑定；`load_run` 可完整回放。
- [x] 九项验收门槛全部通过，阈值全部取自规格的 `acceptance` 块。
- [x] 阈值台账：规格 `acceptance` 的每个键要么被门槛消费，要么登记为显式延后；`all_passed` 同时要求所有单轮门槛均已发出。
- [x] 运行时字段三分类裁决（`enforced` / `recorded_inert` / `single_implementation`），不支持的取值在加载期拒绝，分类写入 `effective_runtime.json`。
- [x] `rigid_body` 与 `finite_difference` 两种运动学来源都真正实现，未选中的一个保留为交叉校验。
- [x] 内核、层 1 裁决与命令上限、层 2 来源、实验装配与验收六组测试，共 134 个测试。
- [x] 滑动命令在唯一出口限幅，越界初始速度加载期拒绝。
- [x] `max_duration_s` 作为硬性终止时间，超时中止并报告；预算无隐藏余量。
- [x] 区分 `run_checks_passed`（七项单轮）与 `k1_acceptance_passed`（九项完整），时间细化要求两个不同物理步长，所有结论由唯一的 `finalize_report` 收口。

## 已知边界

- 力的量级依赖驱动语义，不是探头自重。原因与后果记录在 [`k1_single_probe_rigid.md`](../k1_single_probe_rigid.md) 与 `effective_runtime.json`。
- 层 3、层 4、层 5 的记录是进程内接口，不是冻结契约。
- 与 Isaac Sim 只有行为与守恒门槛，没有数值等价结论。

## 后续任务

- [ ] K2：力控模式、1 N 目标力、控制器状态机、稀疏与密集输出。
- [ ] K3：平底长方体探头、两探头、多物体对独立流、多 Patch、32/64/128 网格回归。
- [ ] K4：Hertz 与弹性地基、压力场、模型接触面积、守恒栅格、材料单元独立状态。
- [ ] K5：Unity 场景输入、执行器标定、误差与延迟报告；层 3 到层 5 记录升级为带版本 Schema。
- [ ] 在保持 K1 接口不变的前提下加入切向摩擦模型。
- [ ] 采集 Isaac Sim 有效运行时参数，之后才做数值比较。
- [ ] K1 轨迹与力序列的曲线化验收报告。
