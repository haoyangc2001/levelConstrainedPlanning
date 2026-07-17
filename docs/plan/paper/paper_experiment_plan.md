# 论文实验工作计划（Paper Experiment Plan）

> 终点：产出一篇会议论文（目标 IEEE ICRA 2027）。本文档给出从当前代码状态到论文提交的完整工作顺序。
> 配套：撰写指南 [../paper/writing-guide.md](../../paper/writing-guide.md)；设计主文档 [../design/末端水平约束轨迹规划与闭环学习进化系统.md](../../design/末端水平约束轨迹规划与闭环学习进化系统.md)；论文骨架 [../paper/main.tex](../../paper/main.tex)。

## 当前状态基线（审计结论）

代码管线 Phase 0–9 已通、已发布 `v0.2.0-closed-loop-baseline`，但**支撑论文的证据几乎全部缺失**：

- **Collision 是 stub**：[../../level_planner_core/validators.py](../../../level_planner_core/validators.py) 的 `collision_safety` 硬编码 `checked:false`。带障碍成功率当前不可信。
- **数据是 smoke 规模**：in-repo 仅 44 候选 / 3 diffusion 正样本 / 6 请求。成熟 993 样本数据集只在旧项目 tashan_Manipulation，不在本仓库。
- **Benchmark 仅 3 请求**：文献标准是每设定 100 随机 start-goal、总 1791–2600 问题。
- **零外部 baseline**：[../../tools/dataset/run_closed_loop_benchmark.py](../../../tools/dataset/run_closed_loop_benchmark.py) 只比内部 4 策略（rule_only/diffusion_only/diffusion_critic/mixed_fallback）。
- **学习分支实测输给规则**：Phase 8 benchmark 3 请求下 rule_only 1/3、diffusion 0/3、mixed 1/3。
- **速度/加速度是无量纲 proxy**（`dt_sec: null`）。
- 可行性门 G1（数据契约可训练性）仍 `not_started`。

## 三项已定决策

1. **外部约束规划 baseline**：先查开源（OMPL 约束规划 / VAMP / cpRRTC / McVAMP），能借则借，否则自己复现投影式。调研用配置的 claude 模型，不用 fable5 搜索。
2. **数据规模目标**：3000+ 样本（远超旧项目 993）。
3. **collision 接入**：作为第一个动手项（A1）。

---

## 阶段 A · 能力地基（最先做，阻断一切带障碍实验）

**总目标**：在生成数据、跑对比之前，先让**度量层可信且达到论文级**。阶段 A 不改方法本身，只保证"每个实验、每个方法都用同一把可信的尺子"——成功率可信（含碰撞）、运动质量指标有真实量纲、benchmark 能扫 K 和时间预算、结果格式统一到论文表格/曲线。

**为什么 A 必须最先做（逻辑纠正）**：当前 [../../level_planner_core/validators.py:79-87](../../../level_planner_core/validators.py#L79-L87) 把 `collision_safety` 直接 pass 成 `valid:True`，意味着现有所有"成功"轨迹**可能是穿模的**。阶段 C 生成的数据用的就是这个成功标签——若不先修，C 会生产出"看似成功实则穿模"的正样本，污染整个学习闭环。因此 **A1 必须早于 C**，这是阶段间最硬的依赖。

关键背景（读代码确认）：世界几何**已建好**（[../../level_planner_core/planner.py:214-219](../../../level_planner_core/planner.py#L214-L219) `build_world`→`update_world`），CuRobo 里有障碍，只是**没沿轨迹查距离**；`self._planner.compute_kinematics(state)` 已存在（planner.py:735），但**世界距离查询接口尚未确认**，是 A1 的前置 spike；`validators.py` 目前是**纯函数**（不 import curobo），碰撞接入要保住这个边界。

### A1 · 碰撞距离回放接入（首项，阻断 C 的成功标签）
目标：把 `collision_safety` 从 `checked:False` 变成沿轨迹的真实最小距离检测，成为硬验收门的一项。
- **A1.1 接口 spike（时间盒化 + 决策门）**：按序尝试本地 fork 的世界碰撞查询（`RobotWorld`/`robot_world` 碰撞工具 → `MotionPlanner`/rollout 上的 `world_coll_checker`/`get_world_coll_dist` → primitive cost）。**硬门**：若 spike 预算内无任一接口暴露 per-config 世界签名距离，切到 A1.1b。全仓库/migration 快照/seed_repair 笔记均无先例代码，故这是真正的未知项。
- **A1.1b 兜底（自实现 SDF，plan B）**：用静态机器人碰撞球（`configs/robot/spheres/xms5_r800_w4g3b4c_spherized.yml`，64 球）+ 世界 cuboid（`world.py make_world_from_boxes`/`sample_tasks.py OBSTACLE_LAYOUTS`）算球-长方体签名距离，绕开 curobo 内部 API。**关键**：64 球分布在 base+link1..link6，`compute_kinematics`（planner.py:735）只返回 tool 帧位姿——A1.1 spike 必须一并确认 per-link FK 路径（kin_state 是否暴露 per-link poses/robot-sphere 张量；否则注册 6 连杆+base 为查询帧，或用轻量运动学库从 URDF 独立算 link FK），否则兜底不成立。
- **A1.2 实现碰撞评估**：批量遍历轨迹点 → 全 link 机器人 spheres → 查世界距离 → 取全路径最小签名距离。
- **A1.3 保边界接线**：由 planner（持有 `_planner`+world）算好碰撞结果作参数传进 `evaluate_hard_constraints`，**保持 validators 纯函数**不 import curobo。
- **A1.4 阈值定义**：activation/safety margin（参照 cuRobo 0–2.5cm）加入 `DEFAULT_THRESHOLDS` 与 config；`valid = min_distance >= margin`。
- **A1.5 更新 benchmark 统计**：[../../tools/dataset/run_closed_loop_benchmark.py:143-147](../../../tools/dataset/run_closed_loop_benchmark.py#L143-L147) 的 `collision_unchecked` 降为 0、`collision_failed` 变真实。
- **A1.6 驱动验证**：已知穿模 + 已知安全两条轨迹各跑一次观察结果；处理无障碍世界退化情形（距离置大、valid）。
- 设计决策：碰撞作**硬拒绝门**（与创新 A"硬验收后置"一致）；种子/repair 仍用 CuRobo 自身碰撞代价，我们加的是事后独立验收。
- **风险状态（已加兜底）**：A1 是最长链根节点，C0a/C2/C3/C4、B6、E1 密度轴全下游依赖它。A1.1 接口未确认，但 A1.1b 自实现 SDF 是明确退路（前提是 per-link FK 在 A1.1 一并探明），故不再是无退路单点失败。

### A2 · 真实时间参数化（与 A1 并行，独立）
目标：让 velocity/accel/jerk 从无量纲 proxy 变成 rad/s、rad/s²、rad/s³，与文献（cuRobo max jerk/accel、motion time）可比。**有量纲 jerk/accel/motion_time 是 A2 的硬出口判据，不是"dt 不可得时的软标注"。**
- **A2.1** 定位 `solve_pose` 结果里的 dt / interpolation_dt。
- **A2.2** 把 dt 贯穿 [../../level_planner_core/repair.py](../../../level_planner_core/repair.py) → candidate metrics。第一路径（便宜、优先）：修 `repair.py` `_extract_first_interpolated_trajectory`，除 position 外一并透传 cuRobo `get_interpolated_plan()` 的 `interpolation_dt` 与 velocity/acceleration/jerk（cuRobo 时间最优重定时已算好），A2.3 直接消费，无需自建重定时。
- **A2.3** 升级 `evaluate_velocity_acceleration_proxy` → 真实 velocity/accel/jerk + `motion_time=(n-1)·dt`。
- **A2.4** 写入 result schema 的 metrics。
- **A2.5（服务外部 baseline）** 对 B3b/B5/B3c 这类无时间参数化的几何路径，用按 SR5 关节速度/加速度限位的统一独立重定时（TOPP-RA 或梯形限速）算有量纲运动质量——scope 到真正需要它的 baseline，而非我方 cuRobo 轨迹。
- **出口判据接线**：D1 加一条——若 E1 任一方法 motion_quality 仍无量纲，则该列在 E1 结果表标"不可与文献比"并移除。

### A3 · Benchmark harness 升级（依赖 A1/A2 字段完整）
目标：产出 E1（成功率+约束误差+时间分位表）与 E2（Success@K vs 时间预算曲线）所需数据。
- **⚠ A3.0 预算真截断（CRITICAL 前置，先于 A3.1/A3.2）**：现状 `total_budget_ms` 只被记录、不截断——planner.py:294 读入，:471-482 只记 `timeout` 布尔量，`plan()` 分支流（288-378）顺序跑、遇成功即返回，中间无按预算中断；真正决定算力的是各策略**写死且不相等**的 `timeout_sec`（rule_only=0.5s vs learned=2.0s，run_closed_loop_benchmark.py:78-119）。→ **当前所谓"固定预算对比"是假的，直接摧毁 E2/Fig.4 与核心假设。** 修：(a) 让 `total_budget_ms` 成为 `plan()` 里真 wall-clock 硬截断（每分支/每次 solve_pose 前检查剩余预算，超时返回当前最优候选）；(b) `strategy_request` 删除写死的不等 `timeout_sec`，令 C4/E2 全策略共享由 `total_budget_ms` 驱动的**同一预算语义**，统一/参数化 `k_generate`；(c) `_summarize_strategy` 的成功改为**预算条件化**（只计 `elapsed ≤ budget` 的成功），使 `fixed_budget_success_rate ≠ final_success_rate`；(d) 若 CuRobo `solve_pose` 无法中途打断，退到迭代/solve 次数预算对齐，并在 E6w/E7w 明确 Fig.4 x 轴是 compute-budget（solve 调用数）而非 wall-clock。**此项落地前 E2 不能作为头号图。**
- **A3.1 K 扫描**：把硬编码的 `k_generate`（2/4/6）参数化为 `--k-values`，统一覆盖所有方法默认 K，使 K 成唯一受控轴（与 timeout/预算解耦，见 A3.0）。
- **A3.2 预算扫描**：`--budget-values` 循环（依赖 A3.0 真截断，否则扫描无意义）。
- **A3.3 Success@K 语义（同 K 可比）**：K 条种子中至少一条经硬验收通过；**跨方法一律在同一 K 网格 × 同一预算下比**；每方法至少生成 ≥K 个候选方可计入该 K 点，否则标 N/A。planner 已返回全部 `candidate_records`，**后处理计算**而非重跑，省算力。
- **A3.4 约束误差聚合**：已有真实 `max_alignment_deviation_deg` 聚合成分布（mean/p50/p95/max）+ 违约率；并入 A1 碰撞最小距离分布、A2 jerk/motion_time。
- **A3.5 延迟分位**：补 **p75**（文献 5.1/5.3）与 **p98**（当前只有 p50/p95/mean）。
- **A3.6 schema 升版** `closed_loop_curobo_benchmark.v1→v2`，加 K/budget 轴与约束误差块，向后兼容。
- **A3.7 方法轴接缝**：把 runner 抽象成通用"method"接口（rule/diffusion/external 同格式上报）——**通向阶段 B 外部 baseline 的接缝**。设计成 run_closed_loop_benchmark 与 run_lifecycle_batch 都能 import 的独立 dispatch 模块。
- **A3.8 统计重复（显著性前置）**：harness 加 `--repeats`/`--eval-seeds`，每设定用 N≥5 个不同 RNG 种子重复，重点是**扩散采样种子**；保留 D2 问题集种子冻结（供配对），eval-seed 重复叠加其上。范围提示：多种子只加在 C4/D4 的随机学习方法，不放大整个 D3 6000-plan 矩阵。

### A4 · 统一论文级结果格式（与 A3 共演化，A1/A2 字段定后定稿）
目标：一份所有实验（E1–E4）、所有方法（我们+baseline）都产出的规范记录，让表/图统一生成。
- **A4.1** 定义 `paper_result.v1`：`{method, constraint_class(LP/LPO/PP/PPO), obstacle_density, K, budget_ms, n_problems, n_success, success_rate, success_ci{low,high,method:"wilson"}, n_runs, seed, per_problem_outcomes(method×problem_id→bool，供事后 McNemar 配对，聚合会丢配对信息故必存), success_at_k, constraint_error{...,violation_rate}, collision{min_dist,collision_rate}, latency{p50,p75,p95,p98,mean}, motion_quality{max_jerk,max_accel,mean_vel,motion_time}, diversity{waypoint_variance_valid_only,n_valid_candidates,ik_branch_count,obstacle_topology_count}, continuity{reconfiguration_count}, hardware{device:cpu|gpu,gpu_model,cpu_model}}`。**关键：VAR 只在通过硬验收/可修复的候选子集内算**（否则"多样但全废"的方法 VAR 反而高、可灌水）；成功率带 Wilson/Clopper-Pearson CI；每条记录标运行硬件（CPU baseline 与 GPU 方法 wall-clock 不可直接比）。
- **A4.2** benchmark JSON → paper_result 转换器；从原始 candidate_records 提取逐问题成功位输出配对表。
- **A4.3** 表/图生成器 stub（喂阶段 E 的图表；mean±95%CI、误差带、显著性列）。
- **A4.4 出口核对**：schema 字段 ⊇ writing-guide §5.3 指标清单（p75/避障拓扑数/reconfiguration 数），二者不一致则改一处对齐。
- 这是阶段 B baseline 必须遵守的"契约"，故 A4 规格要早定。

**A 内部时序**：`A1.1 spike → (A1.2–A1.6 碰撞接入 ∥ A2.1–A2.5 时间参数化) → A4.1 结果格式规格 → A3.0 预算真截断 → A3 harness 完整化(含 A3.8 统计重复) → A4.2/A4.3/A4.4 转换器+图表 stub+出口核对`。
**出口判据**：无障碍+有障碍两种世界下对一批请求跑通，产出的 `paper_result` 记录里 collision 已 checked、jerk/motion_time 有量纲、Success@K 与时间预算曲线数据齐备、预算为真截断、成功率带 CI/per-problem/seed——即"尺子可信"。这样 C 生成的数据标签才可信，B/D 的对比才 apples-to-apples。

## 阶段 B · 外部 baseline 接入（A1 后可与 C 并行）

**总目标**：让完整闭环系统能与其它范式的规划器**在同一把尺子下**对比。核心是先建"方法轴接缝"，再让每个 baseline 都产出经**同一硬验收器**打分的 `candidate_records`——否则对比不是 apples-to-apples。

**读代码确认的关键事实（决定子任务形态）**：
- **方法轴目前不存在**：[../../tools/dataset/run_closed_loop_benchmark.py:25-30](../../../tools/dataset/run_closed_loop_benchmark.py#L25-L30) 的 `STRATEGY_ORDER` 硬编码 4 个内部策略，`strategy_request()`（:68-70）对未知名直接 `raise ValueError`；`run_benchmark` 直接 import 并实例化 `LevelConstrainedPlanner`（:229/241/260）。**没有任何外部 planner 接缝**——这就是为什么 B 硬依赖 A3.7。
- **种子接缝是好消息**：[../../level_planner_core/planner.py:653-726](../../../level_planner_core/planner.py#L653-L726) `_collect_external_seed_candidates` 对所有种子源统一处理（repair→validate→select），只按 `source_type` 分组（planner.py:446-464）。新的外部**种子**源只要返回 `SeedProviderResult` 就能无改控制流地流过——但 `_run_seed_providers`（planner.py:538-634）**硬编码只有 rule + diffusion 两个 provider**，无注册表，加第三个要改这里。
- **结果契约**：每个方法必须产出 `PlannerResult.to_dict()`（[../../level_planner_core/result_schema.py:91-104](../../../level_planner_core/result_schema.py#L91-L104)），至少含 status + selected trajectory + 带 `validator_metrics` 的 `candidate_records`，才能被同一 `_summarize_strategy` 统计。

### B0 · 方法轴契约（与 A3.7 耦合的前置）
目标：定义"方法 = 名称 + `callable(request,config)→result_dict`"，把 runner 从硬编码单一 planner 改成方法注册表，**不写任何外部 planner**。现有 4 策略变成 `ours/*` 方法，外部方法并列注册。保持向后兼容（[../../tests/test_phase8_artifact_and_benchmark.py](../../../tests/) 守护）。**设计成 run_closed_loop_benchmark 与 run_lifecycle_batch 都能 import 的独立 dispatch 模块**（两个 runner 现在各自实例化 planner，见风险）。依赖 A3.7 + A4.1。

### B1 · 开源调研 + 借用/复现决策（纯文档，可立即开始，不依赖 A）
读 [../paper/referpaper/](../../paper/referpaper/) 的 cpRRTC/McVAMP/VAMP/pRRTC PDF，产出"方法→{开源可得?/许可/SR5-URDF+碰撞世界桥接成本/借用 vs 复现}"决策表，喂 B3b/B3c/B5 定 scope。注意 referpaper/README 标注 OMPL 约束规划无 arXiv、需从 venue 引。调研用配置的 claude 模型，不用 fable5 搜索。**必须交付项**：为 B5 明确给出一条"最低成本硬约束投影 baseline"确定路径（不只对重型 OMPL 做 go/no-go），保证 B5 scope 不会归零；为 B3c 圈定经典约束优化对手（CHOMP/GPMP2/TrajOpt-TSR 之一）的可得实现与桥接成本。

### B2 · cuRobo 软约束 baseline（re-port，预期弱）
**纠正**：标准核心里**没有 lambda_level**（planner.py 搜 `lambda_level/update_tool_pose_criteria` 0 命中），对齐是**后置选择门**强制的，不是软代价。真实机制在**未移植的** migration 快照 `migration/source_snapshot/curobo_v2_planner/main.py:4107-4126` `_apply_shadow_non_terminal_pose_weight`（`update_tool_pose_criteria` 逐轴权重 `[0,0,0,w,w,w]`）。而该快照已记录它退化为 **horizon-1-only**（main.py:1700 `goalset_fallback_due_pose_cost_horizon1_only`）——即 cuRobo 软代价**本就压不住中途对齐**。所以 B2 是"soft-cost 会失败"的 baseline。**须跑一次 λ_level sweep 并把 sweep 曲线写进论文**（用数据而非未调单点预防稻草人指控），正文必须坦承 B2 是范式失效示例、非认真对手（真正的认真对手是 B3c/B5 的硬约束求解器）。依赖 B0 + A1。

### B3a · cuRobo 原生无约束下限（近乎免费）
**纠正**：`planner_native`（planner.py:801-806，cuRobo `plan_pose`）**不是 RRT-Connect**，是 cuRobo 自己的运动生成、对水平轴无约束。把它单独暴露为 `baseline/curobo_native_unconstrained`，产出 `source_type=planner_native` 的候选，由硬验收门量出其（预期高的）对齐违约。依赖 B0。

### B3b · 真实 RRT-Connect（外部，B1 定 scope）
按 B1 结论集成 OMPL RRTConnect 或 pRRTC，吃 SR5 URDF + 我们的碰撞世界；写 adapter 把路径转成我们的轨迹格式 + 构造 `CandidateRecord`（`source_type=external_rrt_connect`）过同一验收器；几何路径用 A2.5 独立重定时得有量纲运动质量。**显式 descope 终态**：若 B1 判定不可行或超期，降级为不集成，E1 少一列，**不阻断 B6/D3**。依赖 B0 + B1 + A1。

### B3c · 经典约束优化 baseline（认真对手，反 strawman）
**新增（论文完整性要求）**：设计文档:125 把 CHOMP/STOMP/TrajOpt/GPMP2 列为被批判范式却无一作 baseline，现有 baseline 集是"稻草人(B2)+无约束地板(B3a/B3b)+软学习(B4)"，缺一个**真正把水平当硬/软约束求解的经典优化对手**。按 B1 圈定的实现集成 CHOMP/GPMP2/TrajOpt-TSR 之一（优先 TrajOpt-TSR 或 CHOMP，因其对姿态约束有成熟表述），吃 SR5 URDF + 碰撞世界 + `local_axis/target_world_axis/tolerance_deg` 约束，输出转 `CandidateRecord`（`source_type=external_constrained_opt`）过同一验收器，几何/关节路径用 A2.5 重定时。**descope 终态**：若超期，与 B5 二选一保底（二者都是硬约束对手，至少保住一个）。依赖 B0 + B1 + A1。

### B4 · 单向学习种子（≈ 现有 diffusion_only，且 = E3 一档）
**纠正**：B4 几乎不是新代码——就是现有 `diffusion_only` 策略（run_closed_loop_benchmark.py:87-97，critic 自动关，:224）。真正差别：我们的 planner **总是**跑 CuRobo repair + 硬验收门；DiffusionSeeder 式的"剥掉验收门"**只能在报告层做**（记录 repair 后轨迹但不让 level-first 门拒绝它），从而让 baseline 在"到达目标"上成功、我们量它**真实的水平违约率**。这才是诚实的 apples-to-apples 切法。**实现一次，Phase D E3 复用**（设计文档:378 的 6 档消融）。依赖 B0；数字有意义需 C 的重训 checkpoint。

### B5 · 投影式约束规划 baseline（E1 硬底线，不可归零）
按 B1：包 OMPL ConstrainedStateSpace/ProjectedStateSpace（重）或复现切空间投影规划器。把 request 的 `local_axis/target_world_axis/tolerance_deg`（planner.py:518-524）桥接成投影约束函数，输出转 `CandidateRecord` 过验收器，几何路径用 A2.5 重定时。**提为 E1 硬底线**：E1 不得在缺少 ≥1 个约束强制型对手（B5 或 B3c）时交付。若重型 OMPL ConstrainedStateSpace 超期，**退到最小可行硬约束 baseline**——仅 keep-level 类的自实现切空间投影（复用 planner.py:518-524 的 local_axis/target_world_axis/tolerance_deg，不必覆盖全 4 约束类），把地板从"零"抬到"至少一个硬约束对手"。依赖 B0 + B1 + A1。

### B6 · 统一进 A3/A4 harness
所有 baseline 经**同一** `validators.evaluate_hard_constraints` + A1 真实碰撞打分，喂 A4.2 转换器成 `paper_result.v1`；扩展 `_summarize_strategy` 消费外部 source_type；在 curobo conda 环境小批 smoke。**硬依赖仅保底集** B2/B3a/B4（这 3 个已满足出口判据的 ≥3、均为低不确定）+ A3.6/A4.1/A4.2；**可选输入** B3b/B3c/B5：就绪则入表、未就绪不阻断 B6/D3。这样单个外部 baseline 失败只从 E1 表减少一列，不再切断实验+论文关键路径。

**B 内部时序**：`B1(即刻,纯文档) ∥ [B0 待 A3.7] → B3a(免费) → B2/B4(config 为主) → B3b/B3c/B5(外部,B1 定) → B6 统一`。
**出口判据**：≥3 个外部 baseline 能在统一 harness 下产出 `paper_result` 记录，与完整闭环同表可比；**且必须含 ≥1 个约束强制型对手（B5 或 B3c），否则 E1 不算完成**（无此则只剩稻草人+无约束地板，会被秒杀）。

## 阶段 C · 论文规模数据 + 重训（依赖 A1；与 B 并行）

**总目标**：产出 3000+ 样本、覆盖约束类 × 障碍密度的数据集，重训 diffusion+critic，并验证核心假设（固定预算下学习种子能否超过规则种子）。

**读代码确认的关键事实（多处纠正原计划）**：
- **⚠ 正样本标签依赖碰撞，故 C 硬依赖 A1**：`positive_for_diffusion = validator_valid`（[../../level_planner_core/planner.py](../../../level_planner_core/planner.py):1186），而 `validator_valid` 里 `collision_safety` 现被硬编码 `valid:True`（validators.py:79-87）。→ **A1 之前生成的任何数据都会把"穿模却标正"的样本烤进 diffusion 与 critic 训练**（critic 更甚：`collision_risk` 恒 0，critic.py:124）。这是正确性门，不是可选项。
- **⚠ LP/LPO/PP/PPO 约束类根本不存在**：sample_tasks.py:214 把 `strict_level` 硬设 True、固定水平轴（:212-213）；planner/config 只建模 keep-level 一种；condition.py 15 维里无约束类特征。→ 约束类不是"采样器小改"，而是**贯穿 采样器+planner 目标处理+validators+condition 向量+checkpoint schema 的多文件特性**（改 condition_dim 会同时改 diffusion condition_dim 与 critic input_dim=227，作废现有 checkpoint）。
- **⚠ 障碍密度对模型是"虚构"**：世界在 planner init 时只建一次（planner.py:214-219），`plan()`/`_normalize_request` 从不读 `request['world']`；condition.py 的 obstacle_count 特征取自那个**单一 config 世界**，恒定。→ 未接 per-request 世界重载前，密度轴对物理规划和条件向量都无效。
- **规模是候选级、非请求级**：当前 44 候选来自仅 6 请求（~7 候选/请求）。→ 冲 3000+ **样本**容易；冲可比的**正样本数**（旧 Phase10 ~140 diffusion 正样本 / 833 候选，~14-17%）才是关键，且 A1 后有效率会**下降**（开始拒绝穿模）。
- **C4 比较器已存在**：[../../tools/dataset/run_closed_loop_benchmark.py](../../../tools/dataset/run_closed_loop_benchmark.py) 的固定预算闭环 benchmark（strategies + `--total-budget-ms`）就是 C4 需要的"固定预算下学习 vs 规则"对比器，**无需新建，只需在重训 artifact 上跑**（但其成功标签同样过碰撞盲验收器，故 C4 也须在 A1 后）。
- **训练面已就绪**：`train.py`/`train_critic.py` 命令面支持 `--epochs/--batch-size/--hidden-dim/--diffusion-steps/--horizon/--samples`；`register_model_artifacts.py` 更新 `artifacts/current_artifacts.json`（保留旧档到 legacy_*），`write_dataset_pointer.py` 更新数据集指针。**注意 batch=64 会被 DataLoader clamp 到 len(dataset)**，正样本 <64 时"240 epochs/batch 64"失去意义——采用前先确认正样本数。

### C0a · A1 门（硬前置）
在生成任何 C 数据前，确认 Phase A 的 A1 碰撞回放已落地（`collision_safety` 能发出 failure_reason）。否则正样本碰撞盲。阻断 C2/C3/C4。

### C0b · per-request 世界应用（障碍密度轴的前置）
把 `request['world'].sampled_obstacles` 接进 `_normalize_request` 并 per-request（或按 obstacle_case 分组）调用 `update_world`。**未做前，障碍密度只能当分层标签，不得声称密度覆盖。** 阻断"×障碍密度"那半个目标，不阻断 keep-level scale-up。

### C0c · 冻结 request 级 train/val/test 划分（防 train-on-test，硬前置）
**新增（防拒稿）**：C4/E1/E2 的"学习 vs 规则"结论要求评测请求与训练请求**实例级不相交**，但现状无任何 split 任务——`export_lifecycle_dataset.py:83-95` 按 run_dir 分且单 run 时全进 train（:469），是 no-op。修：用互不相交的 RNG 种子与 `request_id` 命名空间**在生成任何训练数据前**预生成 train/val/test 三组请求，产出 split manifest（`request_id`→split，记 start-goal 哈希）。同分布采样保留（这是创新 B 该测的 in-distribution 泛化），只要求实例级 disjoint。C2 只吃 train 组；C4 显式在 test 组上跑；`run_closed_loop_benchmark.py` 加载 manifest 并断言评测 `request_id ∉ train`、相交即 fail-loud。exporter/dataset.py 须透传/按 split 过滤（现两者都忽略 split）。阻断 C2/C4/D2。

### C1a · 采样器：真实独立 start/goal（现为单基准 jitter）
把 sample_tasks.py:189-203 的单基准抖动换成**独立随机 start_joint + 可达 target_pose 采样（含可达/IK 预检过滤）**，或一组足够宽的 base 位姿库。产出真正多样的 100 对/类。依赖无（可与 A 并行起草，但数据实跑等 A1）。

### C1b · 采样器：LP/LPO/PP/PPO 约束类支持（**多文件特性，非小改**）
(1) 采样器加 `constraint_class` 字段发 4 类（切 `strict_level`、切 goal 朝向是否约束）；(2) planner 按 request flag 开关对齐约束、放松 goal 朝向（须核 `GoalToolPose` 构造 planner.py:637-751 支持 position-only）；(3) validators 的 `_evaluate_alignment/_evaluate_goal` 按活动类条件化；(4) condition.py 加类编码——**这会改 condition_dim → 改 diffusion condition_dim 与 critic input_dim(227) → 作废现有 checkpoint**（是全新训练，非 warm-start）。依赖无（feature 工作），是 C2b 的前置。
- **⚠ descope 分支（不是干净兜底）**：C1b 是最可能被砍的高风险项，但砍它**会塌掉 E1 的约束类主轴与 Table I 对 McVAMP 的 constraint-class 差异化列**（McVAMP 本身支持 LP/LPO/PP/PPO，我方退化为单约束反成劣势）。故若砍 C1b，**触发 reframe（非静默）**：优先退到**最小可发表子集**——keep-level + 一个放开 yaw 的 TSR(PPO) 两类（仍改 condition_dim/作废 checkpoint，非零成本，但保住 constraint-class 轴的最低可比性）；若连这也超期，则 Table I 删 constraint-class 列、差异化改由 projection/throughput/closed-loop/learning-target 承载（对 McVAMP 仍成立），Intro/Abstract/Limitations 显式收窄为 single orientation-path(keep-level) constraint。此分支的传播见 E4w/E8w 与关键路径。

### C2 · keep-level-only 首轮 scale-up 到 3000+（A1 后即可，today 可达）
现管线已能从少量请求产出大量候选。`sample_tasks.py --count N`（难度×障碍-case-as-label×seed 模式）→ `run_lifecycle_batch.py`（`--progress --resume` GPU）→ `export_core_results_dataset.py` → `validate_candidate_dataset.py` → `write_dataset_pointer.py`。**只吃 C0c 的 train 组，绝不含 test 实例**（不依赖 exporter 的 run_dir split 做隔离，在数据生成入口就排除评测实例哈希）。**按目标正样本数（非样本数）估请求数**；A1 后有效率会低于 ~14-17%，据此加请求。依赖 C0a + C0c + C1a。
- **⚠ 单世界标签警告**：C2 排在 C0b（per-request 世界）之前，故首切片全部候选的 A1 碰撞标签只对**单一 config 世界**成立，C3/C4 首切片结论仅限单世界；E1 的"ours"必须用 C2b(per-request) 数据重训后的模型，不得把 C2-单世界模型数字与 C0b 后的评测混用。

### C2b · 全 LP/LPO/PP/PPO × 密度 scale-up
仅在 C1b（约束类）+ C0b（per-request 世界）都就绪后可做，否则 4 类塌成同一 keep-level、密度恒定。只吃 C0c train 组。依赖 C1b + C0b + C2 + C0c。

### C3 · 重训 diffusion + critic（首切片，keep-level）
`train.py --epochs 240 --batch-size 64 --hidden-dim 128 --diffusion-steps 64 --horizon 32 --samples <new>`；`train_critic.py --epochs 160 --batch-size 64 --hidden-dim 128 --horizon 32 --samples <new>`（旧 Phase10 值；先按 C2 正样本数校准 batch）。**须实际启用复合损失**：在 `diffusion.py training_loss` 里接入 `L_level`（设计文档:330 全路径末端轴夹角误差的可微版）与 `L_collision`/guidance，并给 `train.py` 增 `--level-loss-weight`/`--collision-guidance`(或 `--no-*`) 开关——否则当前是纯去噪 MSE（diffusion.py:30-35），Method 会 over-claim 未实现组件、且 D5 的 −level-loss 消融相对纯 MSE 是空操作。**A1 落地后须把真实 min_distance 接进 critic.py:110-126 的训练标签**，否则 `collision_risk` 恒 0（critic.py:124），critic 碰撞盲。然后 `register_model_artifacts.py` + `write_dataset_pointer.py` 更新指针。**若 C1b 改了 condition_dim/input_dim，是全新训练。** 依赖 C2 + C0c。

### C3b · 全类数据重训（full-matrix checkpoint）
在 C2b 全约束类×密度数据上重训，生成与 C1b 新 condition_dim 兼容的 checkpoint（供 E1 全矩阵的"ours"）。依赖 C2b。

### C4 · 验证核心假设（用现成比较器，test 组）
`run_closed_loop_benchmark.py --strategies rule_only,diffusion_only,diffusion_critic,mixed_fallback --total-budget-ms <fixed>` 在 **C0c 冻结的 test 组**上跑（显式声明 train∩test=∅，manifest+"评测实例哈希不出现在任一训练 run"的自动校验写进 artifact，作为 D1 出口校验一条）。**贡献 (3) 的成败点；仅在 A1 + A3.0 预算真截断后有意义。**
- **统计协议（防"这差异显著吗"）**：在同一 test 组跑全部方法（天然配对），保存 `per_problem_outcomes`；对 diffusion vs rule 这对关键对比算 **McNemar 精确检验** p 值 + 效应量；随机学习方法用 **≥3 采样种子**报均值±std（对应设计文档:376 的方差必测）；成功率报 **Wilson CI**。**预注册 go/no-go 阈值**：判"学习超过规则"的最小百分点差 + 显著性水平须在跑之前写死（否则正负叙事都不可辩护）。依赖 C3 + C0a + C0c + A3.0。

### C5 · 消融变体重训（供 D5 第6档）
**新增**：D5 第6档消融（−level-loss / −collision-guidance / −多样性筛选）本质是不同训练/采样配置的模型变体，但 C 阶段原只有 C3 单模型、代码里也无这些损失开关。先在 C3 里实现缺失组件（见 C3 的 L_level/L_collision），再训 3 个 checkpoint：full（含两项）、−level-loss、−collision-guidance。−多样性筛选无需重训，仅以 `evaluate_critic.py diversity_weight=0` 的采样档产出。**若时间不允许实现这些损失项**，则 D5 第6档从 Table II 降级为 future work，同时 E1w 的 L_level/碰撞引导措辞从"已实现"改为"规划中"。依赖 C3。

**C 内部时序**：`C0a(A1 门) + C0c(split 冻结) → C1a → C2(keep-level, A1-aware, train 组) → C3(启用 L_level) → C4(test 组+统计) → C5(消融重训)` 作为去风险、可发表的**首切片**；再 `C0b + C1b → C2b → C3b` 补齐全 LP/LPO/PP/PPO×密度。这样把"已工作的 keep-level jitter scale-up"与"须新建的约束类/密度"隔离。

## 阶段 D · 跑完四组实验（依赖 A+B+C 全部就绪）

**总目标**：把可信 harness（A）× 外部 baseline（B）× 论文规模模型/数据（C）合起来，跑出四组实验的**已提交 artifact**（每个论文数字都能追溯到 `runs/` 或 `reports/` 里的一条记录，无 prose-only 数字）。

**读代码确认的关键事实（多处纠正原计划）**：
- **⚠ 机器人：已定为全 SR5（原计划的 CR7 假设已废）**：[../../configs/](../../../configs/) 只有 SR5——`sr5_level.yaml` → `robot/xms5_r800_w4g3b4c_v2.yml` + 6 连杆 URDF（6-DOF）。CR7 只作为历史文字出现（`phase0_lifecycle_inventory.py:314` "SR5 only; CR7 remains historical regression reference"；migration 里引 `rokae_cr7_dahuafuhe.urdf` 但**该文件不在本仓库**）。**CR7 的 44→92/100 是旧项目历史结果，本仓库不可复现，不作为本文实验。** 本文 sim + real 均用 SR5（6-DOF），做成单机器人论文。
- **障碍密度是 metadata-only、未接入规划**：[../../tools/dataset/sample_tasks.py](../../../tools/dataset/sample_tasks.py):230-235 把 `sampled_obstacles` 只作分层元数据（:234/:341 明确注明），planner 在 init 时**只从 config 建一次世界**（planner.py:214-219），**无 per-request 世界重载**。→ 现状下 3 个密度桶规划的是**同一个世界**，E1 密度轴是假的。这是必须新建的前置（归属 A/C，D2 依赖它）。
- **start/goal 是单基准位姿的 jitter，非独立随机对**：sample_tasks.py:189-203 围绕一个 base pose 抖动，不是"相隔一象限"的独立可达对。→ 现状会产出 100 个近重复请求，削弱 E1 说服力。C1 须加真实 start/goal 采样（工作/关节空间随机 + 可达/IK 校验）。
- **6 档消融只有 4 档现成**：`rule_only/diffusion_only/diffusion_critic/mixed_fallback` 对应第 1/3/4/5 档；**缺**第 2 档"规则+硬门 vs 无门"（无 flag 关验收器）、第 6 档组件消融（`-level损失/-碰撞引导/-多样性筛选`，属训练/种子变体，可能需 C 重训消融模型）、以及"时间匹配对照"（无 wall-clock 匹配机制）。
- **实机几乎无支撑**：`level_planner_ros/planner_node.py:45-116` 只是 `std_srvs/Trigger`→plan→写文件的 stub，**无关节流、无轨迹下发、无 SR5 端水/焊接任务接入、无杂乱场景**。E4 现有 20/20、18/20 仅存在于旧项目 prose，本仓库无 artifact。

### D0 · 机器人：全 SR5（已定，只需同步文档）
**决策已定**：仿真与实机**统一用 SR5**（6-DOF），做单机器人论文，删除 CR7——省掉整机上线子任务、sim/real 一致。落实项：
- 同步 [../paper/writing-guide.md](../../paper/writing-guide.md):188（把 "sim=CR7(7-DOF), real=SR5(6-DOF)" 改成 sim+real 均 SR5）与本文档实验矩阵中的机器人列。
- E1/E2/E3 仿真机器人 = SR5；不引 CR7 的 44→92 历史数字作为本文结果（如需可在 Related/背景以"既往工作"一句带过）。
- 无整机上线工作（CR7 URDF/球化/配置全部不做）。
依赖：无（须最先，但已无待决项）。

### D1 · A/B/C 出口校验 + 学习 go/no-go 门（花算力前的门）
逐条核验：A1 落地（`collision_safety` 返回真实最小距离、`collision_unchecked==0`）；A2 落地（`dt_sec!=null`、有真实 jerk/motion_time）；**A3.0 预算真截断落地**（全策略同预算语义、成功预算条件化）；A3/A4 落地（`--k-values`/`--budget-values`/p75/p98/方法轴/`paper_result.v1` 含 CI/per-problem/seed）；B（外部方法已注册可调，**含 ≥1 约束强制型对手 B5/B3c**）；C（重训 checkpoint 可解析且**非 smoke**、C0c split manifest 存在且"评测实例哈希不出现在任一训练 run"校验通过、C1 采样器能出约束类×密度且 per-request 世界生效）。
- **⚠ 学习 go/no-go 门（新增，避免烧算力）**：把 C4 的"完整闭环 vs 纯规则管线（匹配预算）"单点对比作为**硬判据**——若在匹配预算下完整闭环**不优于**纯规则管线（按 C4 预注册阈值+显著性），则**不投入** D3/D4/D5 的 6000-plan 重算力与 C2b/C3b 全类重训，转而走创新 B 降级叙事（见关键路径三层分层）。
任一不过 → 回对应阶段补齐。依赖 D0 + A3 + A4 + B6 + C3 + C4 + C2b（全矩阵项）。

### D2 · 生成 E1 请求集（每组 100 独立 start-goal 对，test 命名空间）
用 C1 扩展后的 sample_tasks.py 出请求集，**从 C0c 的 test 命名空间抽样**（与训练实例 disjoint），冻结**问题集种子**（供配对），并与**方法采样种子**（≥3 变化，供方差）区分。产出 eval-set manifest（每条 `request_id`/start-goal 哈希），`run_closed_loop_benchmark.py` 断言评测 id ∉ train。
- **⚠ 维度条件式（随 C1b/C0b 就绪度退化，非固定 1200）**：C1b+C0b 就绪 → `LP/LPO/PP/PPO × {none,sparse,dense} × 100`（=1200）；C1b 超期（退最小子集）→ `keep-level(+PPO 子类) × {none,sparse,dense} × 100`（=300，密度轴仍需 C0b）；C0b 也超期 → `keep-level × 100` 单世界。杜绝 1200 近重复请求。
依赖 D1 + C1a（+ C1b/C0b/C0c 视就绪度）。

### D3 · E1 横向仿真矩阵
方法集（完整闭环 + B2/B3a/B4 保底 + B3b/B3c/B5 就绪者，**须含 ≥1 约束强制型对手**）在 D2 请求集上跑（`--resume`）。收集成功率、约束误差分布（mean/p50/p95/max + 违约率）、碰撞最小距离/率、延迟 p50/p75/p95/**p98**/mean → 经 A4.2 转 `paper_result` → E1 表。
- **统计**：请求集由 D2 冻结共享（天然配对），加逐问题成功位日志 + 每方法 Wilson CI（仅需计数，单跑即可）；ours vs 关键 baseline 的成对成功率用 **McNemar**；连续量（水平误差/延迟）用配对检验或 bootstrap CI。
- **硬件可比**：同硬件类内比 wall-clock；跨类（GPU-ours vs CPU 投影/RRT-Connect）延迟并列必须附硬件规格、不作直接速度断言。
- **规模提示**：全矩阵 4 类×3 密度×100 对×N 方法 ≈ 6000 次 plan；退化档 keep-level×3 密度×100×N ≈ 1500。按方法分批过夜跑（单 A100 实测 ~0.5–3s/plan，全矩阵约 17–25 GPU-hr）。
依赖 D2 + D1 + B6 + A3 + A4 + C3（keep-level 子矩阵）/C3b（全矩阵）。

### D4 · E2 时间预算曲线
在密集障碍子集上，用 A3 `--budget-values 100,300,1000` × `--k-values` 扫描（**依赖 A3.0 真截断，否则曲线无意义**）；Success@K = K 条种子至少一条过硬验收、**同 K 网格跨方法可比**，后处理 candidate_records 计算不重跑（A3.3）；每预算点报 **Wilson CI**，学习分支 ≥3 采样种子的方差带；出 Fig.4 曲线数据。
- **登记为贡献 (2) 下界保证的验证/证伪实验**：固定预算下 mixed vs rule 的高低是 trade-off 而非保证；若出现 mixed<rule 交叉点，"下界不退化"须重述为"可加预算下的成功率下界"。
**结果不确定**：核心假设尚无正面结果，可能落成"假设+协议"。依赖 D1 + A3.0；可与 D3 共享请求集。

### D5 · 补齐 E3 缺失档 + 时间匹配对照，再跑 6 档消融
**新建**（非仅配置）：第 2 档硬门 on/off toggle（现无 flag 关验收器）；第 6 档 `-level损失/-碰撞引导/-多样性筛选`（**依赖 C5 消融 checkpoint**）；wall-clock 预算匹配的 rule-only 对照（**贡献 (2) 下界保证的证伪点**）。在混合场景跑全部档，算成功率单调链（mixed_fallback ≥ rule_only ≥ 单组件，**每档带 CI，证差异不在噪声内**，且单调链是固定预算断言、须加预算条件叙述）+ VAR 逐点方差多样性（**仅在有效/可修复候选内算**）。依赖 D1 + C5（第6档）；第 1/3/4/5 档已现成。

### D5b · 负面结果诊断分析（把 C4 负面转成正面贡献）
**新增**：设计文档:386 已承认"纯学习分支尚未超过 rule-only"，C4 负面是大概率结果。仅"写成假设+协议"是防御、非科学贡献。产出诊断证据：(1) **覆盖差**——把 `ik_branch_count`/避障拓扑数/`waypoint_variance` 按"成功/失败"×"学习种子 vs 规则种子"分层聚合，量化学习分布相对规则可达的"可修复种子集"覆盖了什么、缺了什么；(2) **失败模式 taxonomy**——聚合已记录的逐候选 `failure_reason`（validators.py:103-113）与 `failure_reason_counts`（run_closed_loop_benchmark.py），成按方法分列的计数表（偏离流形/对齐、分支不连续/joint-step、碰撞、超预算），近乎零新代码；(3) **机理对照**——把 B2 soft-cost 失败（horizon-1 退化）与学习失败 taxonomy 并列，写成统一叙事"为什么软代价与朴素学习都压不住路径约束"。C4 负面时喂 D7/E6w/E7w，使贡献 (3) 从纯兜底升级为分析性正面贡献。依赖 D4/D5。

### D6 · E4 实机
**纠正 E4 定位**：先决 scope——(i) **复用**旧硬件证据 20/20、18/20，但**标注为"既往工作对同一 SR5 平台的验证"、给旧项目引用，不计入本文第一方定量结果**（本仓库无 artifact、双盲外不可核验、违反"数字须可追溯"）；或 (ii) 跑新硬件试验（则须先建**轨迹下发桥**——现 ROS 节点只 plan+写文件、无执行）、布置无障碍+杂乱 SR5 工位、统一协议下 vs 最强 1-2 baseline，产出 `runs/`/`reports/` 下可追溯 log+视频作为本文 artifact。记录成功（目标位姿 + 全路径水平容差 + 无碰）+ 关节跟踪误差。
- **⚠ 实机安全 checklist（走 (ii) 的硬前置门）**：执行前 dry-run/可视化校验、关节速度/加速度硬限幅、执行期碰撞距离在线监控（复用 A1 距离查询）、物理 e-stop + 工作区围栏、首轮低速试运行。无此不上真机。
**若硬件/时间不可得，E4 走 (i) 并明确不作本文定量结果**。依赖 D3（定最强 baseline）+ 硬件可得性。

### D7 · 汇总成表/图 + 诚实分层叙事 + 独立几何校验
所有 `paper_result` 喂 A4.3 生成器（Table I/II、Fig.4）；按三层诚实分层写（见关键路径）；每个数字对齐 `runs/`/`reports/` 的 artifact（**扫描 E 正文/贡献列表，任何"已验证结论"数字必须指到本仓库一条记录，否则降级**）。
- **⚠ 独立几何校验（堵"自证循环"质疑）**：主成功判据与训练标签同源于同一 FK/validator（`positive_for_diffusion=validator_valid`，planner.py:1186）。新增一次性脚本——用**不参与训练标签的第三方 FK+FCL**（直接从 SR5 URDF 独立算末端轴偏差与最小碰撞距离）复核 E1/C4 成功样本的一个子集，报告验收器自证与独立校验的一致率，证明成功率非验收器自证。
依赖 D3/D4/D5/D5b/D6。

**D 内部时序**：`D0(机器人决策) → D1(出口校验+go/no-go 门) → D2(请求集) → [D3 ∥ D4] → D5 → D5b(若负面) → D6 → D7`。E3 内部档最不受阻（4/6 现成），E1 最受阻（需 A+B+C 齐）。

## 阶段 E · 撰写论文（终点）

**总目标**：产出符合 ICRA 2027 格式的可提交论文。**核心纠正**：原计划把 E 压成"论文撰写+图表+提交"一行，掩盖了一个真实的依赖切分——**约 60% 的 E（Method、Problem、Related Work、Fig.1-3、Table I 骨架、references.bib、去 cuRobo 化的标题/摘要重写）可与 A/B/C/D 并行、应尽早开写**；只有 Experiments 正文数字、Fig.4、Table I/II 的量化格才被 D 阻断。

**读文件确认的关键事实**：
- **main.tex 是完整 IEEEtran 骨架但正文是占位**：preamble/标题/作者/摘要/关键词已写（[../paper/main.tex](../../paper/main.tex):9-78），7 个 `\section` 都在，但 Intro/Related/Conclusion 正文字面就是 "Placeholder."（:89/97/163），Method 子节是注释大纲（:117-136），Experiments 是注释+一个空占位表（:138-159）。
- **⚠ 现摘要/标题违反去 cuRobo 化纪律**：标题含 "CuRobo Repair"（:36）、摘要 3 次点名 "CuRobo"（:36/64/70）。撰写指南（writing-guide.md:20-26）要求 Abstract/Intro/Problem **不得**点名 cuRobo（说"优化式规划"范式），只在 Implementation/Experiments 点名。**这是必须修的既存缺陷，不是新写。**
- **⚠ 章节结构与指南不符**：main.tex 现为独立 "Method"（3 子节 Seeds/Repair/Closed-Loop）+ 无结构性局限的 "Problem Formulation"；指南要求合并成单一 "System Design"（IV-A..IV-D）+ "Problem Formulation & Structural Limitations"（含 Observation 1/2）。**E 首步是重构骨架，不是填空。**
- **水平约束 `zᵉᵉ·zʷ ≥ cos ε` 已在 Problem 定义一次**（eq:level，:107-112，宏 `\zee \zw` 已定），符合指南"只定义一次"。
- **references.bib 缺 6 篇**：cpRRTC/McVAMP/VAMP/pRRTC/DiffusionSeeder/IKLink 无条目，PRESTO 是占位（author='Author, A. and others'），IKFlow 年份 2022 与文件名 2021 冲突，缺 GPMP2。→ Related Work + Table I 依赖先补 bib。
- **figures/ 只有 .gitkeep（零资产），仓库无任何绘图工具链**（grep matplotlib/pyplot 无命中）。4 图全从零画；Fig.4 还依赖一个 A4.3 尚未建的 Success@K 绘图生成器。`\graphicspath{{figures/}}` 已接好。
- **既有一个占位表**（tab:main，:144-159，全 `--`）映射到 Table II 消融（5 行已对齐 E3）；Table I 关联工作矩阵是**新表**。

### E0 · 重构 main.tex 骨架（先于任何正文）
把 "Problem Formulation" → "Problem Formulation & Structural Limitations"；三个 Method 子节合并成单一 "System Design"（IV-A 总览 / IV-B 解耦约束满足 / IV-C 从可靠规划器到数据引擎 / IV-D 自改进闭环）；加 Fig.1（teaser,Intro）/Fig.2（系统总览,跨栏 figure*）/Fig.3（种子构造）/Fig.4（Success@K）/Table I（关联工作矩阵）的 float 占位 + `\label`，保留 tab:main 作 Table II。保留 eq:level 单一定义。依赖无。

### E1w · 起草 Method / System Design（IV，两个创新）—— 最高置信、不受 D 阻断，**最先写**
IV-A 总览段（pipeline = 执行器+数据生成器+带回退的验收器）；IV-B 创新 A（流形感知预置种子：笛卡尔插值 + 目标相对 twist `Q_i=Q_g·T_y(θ_i)` + 分支一致 IK 代价；批量精修；硬后验门 + level-first 排序键，逐条挂到 Observation 1/2）；IV-C 桥接段（两个局限→数据引擎）；IV-D 创新 B（可修复种子分布学习目标、1D U-Net 扩散 + `L_level` 损失、失败即一等数据 + success critic、五阶段闭环 + 下界不退化保证）。
- **⚠ IV-D punchline 须先声明预算语义再给下界句**：下界保证限定为"**可加/无限预算下的成功率下界**（失败后再回退规则）"，固定预算下 mixed vs rule 的高低是效率 trade-off、非保证。
- **⚠ critic 维度按实现择词**：critic 现输出中 `collision_risk` 恒 0（critic.py:124），除非 A1 后把真实 min_distance 接进训练标签（C3），否则 Method 不得声称预测碰撞风险——要么在 C3 落实碰撞维、要么把 critic 输出写成 `[P_success, e_h, n_iter]` 3 维。
- **⚠ L_level/碰撞引导按 C5 落地择词**：若 C5 未实现这些损失项，措辞从"已实现组件"改为"规划中"。
**本节用"批量轨迹优化器"，不点名 cuRobo。** 源：设计文档 2.2-2.4/3.1-3.10 + `level_planner_core/{rule_seed,repair,constraints,validators}.py`。依赖 E0。

### E2w · 起草 Problem Formulation & Structural Limitations（III）
约束轨迹优化模型（变量/平滑目标/约束含 eq:level）+ 流形视角（dim M=n−m、体积坍缩、IK 多分支连续）+ **Observation 1（约束退化）/Observation 2（初值敏感）**（指南 :32-35 已预写文本），保持范式级不点名 cuRobo。依赖 E0。

### E3w · 补全 references.bib（Related Work + Table I 的前置）
补 6 篇 + GPMP2 的真实 author/venue/arXiv ID（cpRRTC 2505.06791、McVAMP 2604.13323、VAMP 2309.14545、pRRTC 2503.06757、DiffusionSeeder 2410.16727、IKLink 2402.16154、GPMP2 1707.07383、PRESTO 2409.16012）。**去所有 `and others` 截断**——不只 PRESTO，还有 `curobo2023`(bib:9)、`trajopt`(:66)、`mpd2023`(:83)、`diffusionpolicy2023`(:98) 同问题，逐条枚举补全作者。修 IKFlow 年份（PDF masthead 为 ACCEPTED MAY 2022 / RA-L，year=2022 以发表年计成立）。**CSVTO 二选一决策**（Composable Diffusion for Constrained Trajectories，与本文"约束+扩散"最相关的对照之一，referpaper/README 标 arXiv 未解析）：找到 preprint 则补 bib 并进 Related Work 学习式一类；确无则正文一句说明并引其 workshop/venue——不留悬空 TODO。依赖无。

### E4w · 起草 Related Work（II）+ 建 Table I 骨架
四类关联工作（采样式约束；IK 方法；轨迹优化；学习式 seeding/扩散），每类以"我们如何不同"收尾 + 指南 :104-105 的差异化句。Table I 列：projection/throughput/closed-loop/constraint-class/learning-target × {cpRRTC,McVAMP,DiffusionSeeder,PRESTO,ours}，先填定性格，量化 throughput 格待 D。**constraint-class 列 conditional on C1b**：若 C1b 被砍（keep-level-only 终稿），删该列、差异化改由 projection/throughput/closed-loop/learning-target 承载（对 McVAMP 仍成立），且 ours 的多类若仅 design-only 须标注非实测。**补一行"约束强制(投影)"**确保 Table I 中 "constraint-enforcing optimization" 格非空（对应 B5/B3c）。依赖 E3w。

### E5w · 画三张概念图（Fig.1/2/3，不需 D 数据，用 dataviz 技能）
先调 dataviz 保持风格一致。Fig.1 teaser（约束流形薄面 + 坏种子偏离收敛到违约 vs 多种子近流形+硬门+回退）；Fig.2 系统总览（跨栏 figure*，三阶段 pipeline + 五阶段闭环 + 回退箭头，标执行器/数据生成器/验收器）；Fig.3 种子构造（twist 插值 + 连续 IK 分支选择示意）。导出到 [../paper/figures/](../../paper/figures/)。依赖 E0。

### E6w · 起草 Experiments 非数字部分（V）—— D 跑的同时写
E1-E4 协议段（按范式列 baseline；指标：最终成功率、约束误差分布、延迟 p50/p75/p98、Success@K、VAR 多样性）。**此处才点名 cuRobo 作为优化器实例**。
- **⚠ 预算语义先声明再报曲线**：协议段先声明预算语义（可加 vs 固定）再报 Success@K；Fig.4 x 轴标清是 wall-clock 还是 compute-budget（solve 调用数，若 A3.0-(d) 走迭代预算）。
- **⚠ 统计方法写明**：比例用 Wilson/Clopper-Pearson CI、成对二元用 McNemar、连续量用 bootstrap/配对检验；给出 n 与种子数 + 样本量/功效说明——使"假设+协议"成为完整协议。
- **⚠ 硬件公平性声明**：CPU 采样/投影 baseline 与 GPU 方法 wall-clock 不可直接比，报硬件规格，GPU 吞吐作为显式系统性权衡。
- **诚实分层覆盖到贡献 (2)**：下界措辞按预算语义择词（可加预算下的成功率下界），不只覆盖贡献 (3)。
数字格与曲线留 `\TODO` 挂 D。依赖 E0。

### E7w · 填 Experiments 数字 + Fig.4 + 表 I/II 量化格（**阻断于 D**）
建/接 A4.3 的 Success@K 绘图生成器（含 CI 误差带），用 E2 `paper_result` 渲 Fig.4；填 Table II 消融 6 行（证单调链 mixed≥rule，**带 p 值/CI/效应量/种子方差**）；填 Table I throughput/约束误差格 + E1 成功率+约束误差+延迟表（带 Wilson CI 与 McNemar p 值）。**印刷前用 A1 修好的碰撞重核 SR5 数字（仅 SR5，删 CR7）**。依赖 E6w + E3w +（D 结果，而 D 依赖 A/B/C）。

### E8w · 最后写 Intro（I）+ Abstract，并去 cuRobo 化标题
四段式 Intro（动机/范式局限（不点名 cuRobo）/洞见+方法/3 条贡献）；重写摘要去 3 处 "CuRobo"、去标题 "CuRobo Repair"（用指南候选标题）。
- **⚠ 贡献 (1) 措辞**：只声明 E1（新 SR5 仿真）+ E4（按 D6 选定 provenance）能支撑的内容；旧 SR5 硬件 20/20、18/20 若沿用须标"prior validation of the same SR5 platform in earlier work"并给旧项目引用，**不得作本文第一方结果**；**删除 44→92/100**（D0 已裁定 CR7 不作本文实验）。
- **⚠ 贡献 (2)+(3) 措辞按证据择词**：Abstract/Intro/§IV-D 标题里的 "self-improving" 须以正面结果为门——无 C4/D4/E2 正面结果则不作为卖点，创新 B 降级为 "a lower-bound-safe integration architecture"；贡献 (3) 按 C4/E2 结果择"已验证"或"假设"；贡献 (2) 下界句按预算语义择词。
依赖 E1w+E2w+E4w+E6w。

### E9w · 内审 + ICRA 格式/页数核对 + 构建 + 提交
`cd docs/paper && latexmk -pdf main.tex`（+ bibtex），清所有 `\cite/\ref`/未定义引用/浮动告警；核对 ICRA 2027（6 页、IEEEtran、PaperPlaza，**官网确认确切页数/截止**）；按诚实分层与去 cuRobo 化通读润色；备 PaperPlaza 提交。依赖 E7w+E8w+E5w。

**E 内部时序**：`E0 → [E1w,E2w,E3w 早写] → [E4w(待 E3w),E5w,E6w 与 A/B/C/D 并行] → E7w(待 D) → E8w → E9w`。**关键提醒**：指南的写作顺序（Method→Experiments→…）是"置信度顺序"，与依赖顺序冲突（Experiments 最受 D 阻断）——按依赖调度，Experiments 数字等 D 落地再填。

---

## 关键路径与风险

- **最长链**：`A1(spike→SDF兜底) → A3.0(预算真截断) → C0a/C0c → C1a → C2 → C3(启用 L_level) → C4(test集+统计) → D1(go/no-go 门) → D(E1–E4) → E7w → E9w`。碰撞（A1）、预算真截断（A3.0）与正样本规模是前置闸门。
- **并行**：B 轨与 C 轨在 A1 后同时推进；A2/A3/A4 穿插在 B/C 期间但须早于 D；E 的 E0/E1w/E2w/E3w/E4w/E5w/E6w 可与 A/B/C/D 并行早写（约 60% 的写作不受 D 阻断）。
- **机器人已定为全 SR5（D0）**：仿真与实机统一 SR5，单机器人论文；CR7 不作为本文实验。落实项见 D0，且传播范围须扩到 writing-guide :46/:140/:162/:188/:206-208 与 E7w（删 CR7），非仅 :188。
- **两个被原计划隐藏的前置**：(1) **per-request 世界应用**（C0b）——否则 E1 的障碍密度轴是假的；(2) **约束类支持**（C1b）——LP/LPO/PP/PPO 是贯穿多文件、会作废 checkpoint 的特性，且链路 `C1b→D2→D3→Table I constraint-class 列/McVAMP 可比性`，砍它触发 reframe（见 C1b descope 分支）。
- **一个致命前置（审查新发现）**：**A3.0 预算真截断**——现状 `total_budget_ms` 只记录不截断、各策略 timeout 写死不等（rule 0.5s vs learned 2.0s），使"固定预算对比"是假的、直接摧毁 E2/Fig.4。此项落地前 E2 不能作头号图。

- **兜底（三层诚实分层，修正原两层）**：原计划把"贡献 1+2 稳、只有贡献 3 可能塌"当前提是错的——贡献 2（自改进闭环"系统"）的新价值依赖学习能改进，而这正是贡献 3。故拆三层：
  - **L1 已验证**：创新 A 的解耦范式（C1 解耦 + 硬门 + 回退）——**但其第一方证据是 E1 的新 SR5 仿真矩阵**（44→92 是 CR7 旧项目、不可复现，20/20 是旧仓库 prose、无本仓 artifact），不是既有数字；
  - **L2 已验证但价值有限**：创新 B 的"可加预算下界不退化架构"（架构上成立，固定预算下是 trade-off）；
  - **L3 与 C4/D4/E2 结果绑定**：创新 B 的 "self-improving" 卖点——若匹配预算下完整闭环不优于纯规则管线，则创新 B 整体降级为 "a lower-bound-safe integration architecture"，学习故事移入 Limitations/future work，论文单靠创新 A（+贡献 3 假设+D5b 诊断分析）投稿。
  D1 的 go/no-go 门据此决定是否投入 D3/D4/D5 重算力。**不把论文押在 C4 必须成功上。** B5/B3c 至少保住一个约束强制型对手；E4 硬件不可得则走"既往验证、不计第一方结果"。
- **最大不确定项（biggest_uncertainties）**：
  1. **A1.1 世界碰撞查询 API**（最上游根节点，接口未确认；兜底 A1.1b 自实现 SDF，前提是 per-link FK 在 A1.1 一并探明）；
  2. **A3.0 预算能否真截断**（若 CuRobo solve_pose 不可中途打断，退迭代/solve 次数预算，Fig.4 x 轴改 compute-budget）；
  3. **C1b 约束类特性 + checkpoint schema 变更**（砍它触发 Table I/novelty reframe）；
  4. **C4 重训后仍可能负面**（固定预算下 mixed<rule 交叉点会证伪"下界不退化"punchline，须重述为可加预算下界）；
  5. **D6 实机无执行桥**（走既往验证兜底则不计第一方结果）。
- **规模纠正**：3000+ 是**候选/样本级**（44 候选来自 6 请求）；按**目标正样本数**估请求数，A1 后有效率会下降。算力**不是约束**——全 D 矩阵单 A100 约 17–25 GPU-hr（实测 ~0.5–3s/plan），几个过夜跑。

- **时间线框架（不锁会议，用户决策：先打磨好再定投哪）**：当前**不锁定目标会议**，故无硬 deadline 与会议 pivot 规则。改用**相对里程碑**驱动去风险首切片，拿到 C4/E1 数据后再决定投 ICRA/IROS/RA-L：
  - **M1 尺子可信**：A1(+A1.1b 兜底) + A2 + **A3.0 预算真截断** + A4 出口判据过（"两个世界跑通、collision checked、量纲齐、预算真截断、CI/per-problem/seed 就位"）。
  - **M2 首切片数据**：C0a + C0c + C1a + C2(keep-level, train 组) + C3(启用 L_level)。
  - **M3 假设裁决**：C4(test 组 + McNemar + 预注册阈值) → 触发 D1 go/no-go 门；同时 B 保底集（B2/B3a/B4）+ ≥1 约束强制对手（B5/B3c）就绪。
  - **M4 首切片论文**：E0/E1w/E2w/E3w/E4w/E5w/E6w 早写 + E1/E4(SR5) + D5b(若 C4 负面) → 可投稿的 keep-level 单约束切片。
  - **M5 全量增量**（时间/结论允许才做）：C0b + C1b → C2b → C3b → E1 全 LP/LPO/PP/PPO×密度矩阵 + D6 新硬件。
  每个里程碑标注人日估计待用户定投稿窗口后回填；descope 决策绑到"M3 结论 + 用户选定窗口"，而非日历硬 freeze。

## 进度追踪

| 阶段 | 任务 | 状态 |
|---|---|---|
| A | A1 碰撞距离回放（A1.1 spike + A1.1b SDF 兜底/per-link FK） | 待开始 |
| A | A2 真实时间参数化（有量纲为硬出口 + A2.5 TOPP-RA） | 待开始 |
| A | A3 benchmark harness（**A3.0 预算真截断** · K/预算扫描 · p75/p98 · 方法轴 · A3.8 统计重复） | 待开始 |
| A | A4 统一 paper_result.v1 schema（+CI/per-problem/seed/hardware/topology/reconfig）+ 转换器 | 待开始 |
| B | B0 方法轴契约（待 A3.7） | 待开始 |
| B | B1 开源调研 + 借用/复现决策（可即刻，须给 B5/B3c 最小路径） | 待开始 |
| B | B2 cuRobo 软约束 baseline（re-port + λ-sweep 曲线） | 待开始 |
| B | B3a cuRobo 原生无约束下限（近免费） | 待开始 |
| B | B3b 真实 RRT-Connect（外部，descope 终态） | 待开始 |
| B | **B3c 经典约束优化 baseline（CHOMP/GPMP2/TrajOpt-TSR，反 strawman）** | 待开始 |
| B | B4 单向学习种子（=E3 一档） | 待开始 |
| B | B5 投影式约束规划 baseline（E1 硬底线 + 最小投影兜底） | 待开始 |
| B | B6 统一进 harness（硬依赖保底集 + B3b/B3c/B5 optional） | 待开始 |
| C | C0a A1 门（硬前置） | 待开始 |
| C | **C0c train/val/test 冻结划分（防 train-on-test，硬前置）** | 待开始 |
| C | C0b per-request 世界应用（密度轴前置） | 待开始 |
| C | C1a 采样器：真实独立 start/goal | 待开始 |
| C | C1b 采样器：LP/LPO/PP/PPO 约束类（多文件特性，含 descope 分支） | 待开始 |
| C | C2 keep-level scale-up 到 3000+（train 组） | 待开始 |
| C | C2b 全约束类×密度 scale-up | 待开始 |
| C | C3 重训 diffusion + critic（**启用 L_level**，接碰撞维） | 待开始 |
| C | **C3b 全类数据重训（full-matrix checkpoint）** | 待开始 |
| C | C4 核心假设验证（test 组 + McNemar + 预注册阈值 + CI） | 待开始 |
| C | **C5 消融变体重训（供 D5 第6档）** | 待开始 |
| D | D0 机器人：全 SR5（已定，待同步 writing-guide/矩阵） | 已决策 |
| D | D1 A/B/C 出口校验 + **学习 go/no-go 门** | 待开始 |
| D | D2 E1 请求集（test 命名空间，维度条件式 300/1200/100） | 待开始 |
| D | D3 E1 横向仿真矩阵（含约束强制对手 + McNemar/CI + 硬件披露） | 待开始 |
| D | D4 E2 时间预算曲线（预算语义 + Wilson CI + 方差带） | 待开始 |
| D | D5 E3 消融（补 2 档 + 时间匹配，依赖 C5） | 待开始 |
| D | **D5b 负面结果诊断分析（覆盖差 + 失败 taxonomy + 机理对照）** | 待开始 |
| D | D6 E4 实机（或既往验证不计第一方 + 安全 checklist） | 待开始 |
| D | D7 汇总成表/图 + 三层诚实分层 + **独立几何校验** | 待开始 |
| E | E0 重构 main.tex 骨架 | 待开始 |
| E | E1w Method/System Design | 待开始 |
| E | E2w Problem + Observation 1/2 | 待开始 |
| E | E3w 补全 references.bib | 待开始 |
| E | E4w Related Work + Table I 骨架 | 待开始 |
| E | E5w Fig.1/2/3 概念图 | 待开始 |
| E | E6w Experiments 非数字部分 | 待开始 |
| E | E7w Experiments 数字 + Fig.4 + 表量化格（待 D） | 待开始 |
| E | E8w Intro + Abstract + 去 cuRobo 标题 | 待开始 |
| E | E9w 内审 + 格式核对 + 构建 + 提交 | 待开始 |
