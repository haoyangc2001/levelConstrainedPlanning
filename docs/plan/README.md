# 计划文档索引（Plans）

本目录存放项目的**工作计划类文档**。计划按**背景**分成两类子目录：一类是当前正在推进的论文工作，另一类是此前已完成的闭环系统实施工作。请勿把两类文档混在同一层。

```text
docs/plan/
├── README.md                 # 本索引
├── paper/                    # 当前计划：把已完成的系统做成一篇会议论文
└── closed_loop_impl/         # 历史计划：闭环学习优化系统的实施（已完成）
```

项目主线是闭环学习优化系统：

```text
数据生成 → 模型学习 → 优化验收 → 失败回退 → 数据更新
```

`closed_loop_impl/` 记录的是**如何把这个系统实现出来**（已完成）；`paper/` 记录的是**如何用这个系统产出一篇论文**（进行中）。两者顺序衔接：先有系统，再写论文。

---

## paper/ —— 论文实验工作计划（当前活跃）

终点是产出一篇会议论文（目标会议未锁定，先打磨去风险首切片再定投 ICRA / IROS / RA-L）。记录从当前代码状态到论文提交的每一个可追踪任务、当前状态审计与关键决策。

- [paper/paper_experiment_plan.md](paper/paper_experiment_plan.md)：**叙述版**。人类可读的完整工作顺序，含 `file:line` 依据、诚实分层、风险与里程碑。先读这一份。
- [paper/paper_experiment_plan.json](paper/paper_experiment_plan.json)：**机器可读版**。上一份的结构化镜像——逐 Phase / 逐 task 的 id、依赖、出口条件、审查改动（`review_change`）与关键路径。经 35-agent 对抗性审查后大改（v1.1）。

> 两份内容一一对应：`.md` 用于阅读，`.json` 用于程序化消费与依赖校验。

配套（不在本目录，位于 `docs/` 其他子目录）：撰写指南 [../paper/writing-guide.md](../paper/writing-guide.md)、设计主文档 [../design/末端水平约束轨迹规划与闭环学习进化系统.md](../design/末端水平约束轨迹规划与闭环学习进化系统.md)、论文骨架 [../paper/main.tex](../paper/main.tex)。

---

## closed_loop_impl/ —— 闭环系统实施计划（历史，已完成）

把轻量 SR5 水平约束规划项目从「可验证规划核心 + 离线学习工具链」推进为完整闭环学习优化系统的实施计划及其配套契约。Phase 0–9 已完成，规划管线已发布 `v0.2.0-closed-loop-baseline`。保留这些文档是为了追溯设计决策与数据/接口契约。

- [closed_loop_impl/closed_loop_system_implementation_plan.json](closed_loop_impl/closed_loop_system_implementation_plan.json)：**主实施计划**。Phase 0–9 的逐阶段任务、验收与产出，是本目录的入口文档。
- [closed_loop_impl/closed_loop_schema_contract.json](closed_loop_impl/closed_loop_schema_contract.json)：**闭环数据契约**。冻结 run / candidate / dataset / label / validation 的共享 schema 与存储策略（小产物入 git、大数据落 `/pub/data/caohy/levelConstrainedPlanning`）。
- [closed_loop_impl/rule_seed_migration_inventory.json](closed_loop_impl/rule_seed_migration_inventory.json)：**规则种子迁移清单**。定义从源项目 tashan_Manipulation 迁移可靠规则化水平种子生成所需的最小闭包（只迁移确定性种子数学与 CuRobo IK/trajopt 修复适配，不带 ROS / 状态机 / 产品依赖）。
- [closed_loop_impl/seed_repair_api_notes.md](closed_loop_impl/seed_repair_api_notes.md)：**种子修复接口笔记**。冻结 Phase 0 结论——如何通过 `TrajOptSolver.solve_pose(seed_traj=...)` 把外部轨迹种子喂入 CuRobo 修复。

后三份是主实施计划的支撑契约/结论，被 `closed_loop_system_implementation_plan.json` 直接引用。
