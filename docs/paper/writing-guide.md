# 会议论文撰写指南

> 本文档汇总本项目会议论文（目标：IEEE ICRA 2027，截稿约 2026-09-15，IEEEtran 会议双栏格式）的完整撰写分析。
> 配套文件：设计主文档 [../design/末端水平约束轨迹规划与闭环学习进化系统.md](../design/末端水平约束轨迹规划与闭环学习进化系统.md)；论文骨架 [main.tex](main.tex)；参考文献 PDF [referpaper/](referpaper/)。

---

## 0. 全文定调原则

### 0.1 核心主张（全文围绕这一句）

> 在一个自带验收与兜底的闭环里，学习一个能被已验证优化器快速修复、且贴近约束流形的种子分布——面向末端姿态路径约束这一特定难题。

支撑这句话的关键事实：**同一条约束流水线同时扮演三个角色**——在线执行器、离线数据生产器、学习种子的验收器兼兜底。正是这个"一体三用"把"约束前移—批量优化—硬验收后置"（创新 A）和"闭环学习"（创新 B）焊成一个系统，而不是两个并列的点子。

### 0.2 去具体化原则：不点名 cuRobo 讲痛点（贯穿全文）

软约束 + 非凸局部下降是**基于优化的运动规划范式**（CHOMP/STOMP/TrajOpt/GPMP/cuRobo…）的**共性结构属性**，不是某个实现的缺陷。全文按位置分层处理：

| 位置 | cuRobo 怎么提 |
|---|---|
| Abstract / Introduction / Problem | **完全不提 cuRobo**。讲 "optimization-based trajectory planning" 这一大类，两个失效模式是范式的**结构性属性（structural properties）** |
| Method / System Design | 讲 "a batch trajectory optimizer" / "the underlying optimizer"，泛化描述，不绑定实现 |
| Implementation / Experiments | **这里才点名 cuRobo**——"we instantiate the optimizer with cuRobo"，作为具体实现选择 |

一句话原则：**cuRobo 是我们的 instantiation，不是我们的 problem。** 好处：贡献自动泛化到任意软约束优化器；审稿人更难说"这只是 cuRobo 调参"。

### 0.3 两个失效模式的"定理化"表述

在 Problem 里用带编号的 Observation 把两个失效模式写成范式分析，措辞克制、像分析而非吐槽：

> **Observation 1 (Constraint degradation).** When a path constraint is encoded as a weighted penalty in a scalar objective, a local optimum can minimize total cost while violating the constraint, because the optimizer trades constraint satisfaction against competing terms.
>
> **Observation 2 (Initialization sensitivity).** For a non-convex objective solved by local descent, the returned solution is confined to the basin of its initial seed; a seed off the constraint manifold converges to a constraint-violating local optimum.

这两条是所有方法组件的论证支点。方法的每个组件都要能一一追溯到对这两条之一的针对性回应。

---

## 1. 标题与两个创新

**标题候选**（须同时含创新 A 的"约束满足解耦/流形种子" + 创新 B 的"闭环/自进化"）：
- *Manifold-Aware Seeding with Verified Fallback: A Self-Improving Loop for Orientation-Constrained Manipulation Planning*
- *Decoupling Constraint Satisfaction from Optimization: A Closed-Loop Learned-Seeding System for Level-Constrained Manipulation*

**创新 A（基础，已验证）**：约束满足的"解耦"范式——把约束满足从"优化过程内的单一软惩罚"，拆成优化前（多种子族贴近流形，回应 Obs.2）+ 优化后（硬验收门，回应 Obs.1）。已实测（CR7 44→92/100；SR5 20/20、18/20）。可迁移到任意"soft-cost 优化器 + 硬约束需求"场景。

**创新 B（拔高）**：自进化闭环——把流水线的两个局限（串行 IK 慢 / 覆盖受人工规则族上限约束）转化为数据引擎。三个差异化设计：(1) 学习目标是"可修复种子分布"而非"最优轨迹分布"；(2) 失败是一等公民数据，喂独立 Success Critic 而非扩散模型；(3) 学习失败必然回退规则种子 → **引入学习永不低于规则基线（下界不退化）**。

---

## 2. 章节结构（ICRA/IROS 6 页）

方法合并成**一个** System Design 节，两个创新作为子节——结构本身传达"一个系统"的主张。

```
I.   Introduction
II.  Related Work
III. Problem Formulation & Structural Limitations of the Optimization Paradigm
IV.  System Design                                      ← 合并的方法节
     IV-A. Overview: A Verified Self-Improving Loop      （系统总览 + Fig.2，先立"一个系统"）
     IV-B. Decoupled Constraint Satisfaction             [创新 A]
           - Manifold-aware pre-seeding（多种子族）
           - Batch refinement
           - Hard post-verification gate & level-first selection
     IV-C. From a Reliable Planner to a Data Engine       （桥接，~1 段：两个局限 + "规划器即数据引擎"）
     IV-D. Self-Improving Closed Loop                     [创新 B]
           - Learning target: the repairable-seed distribution
           - Diffusion seeder
           - Failures as first-class data & the success critic
           - Lower-bound guarantee via rule fallback
V.   Experiments
VI.  Conclusion & Limitations
```

**为什么合并方法节**：一个 Method 节 = 一个系统；两个平级节会暗示"两件事"，与主张矛盾。IV-A 先用一段 + Fig.2 统领全局（明说 pipeline 同时是 executor / data generator / verifier-with-fallback），读者带着"这是一个系统"的框架进入 IV-B/IV-D，两个创新自然被读成"系统的两个层次"。风险是这节偏长（约 2 页），用 IV-A 统领 + 子节 label 化对冲。

**命名**：节标题用 "System Design" 或 "Approach"，不用 "Method A and B"；子节保留 "Decoupled Constraint Satisfaction" 和 "Self-Improving Closed Loop" 作为两个创新的 label（出现在目录里就是卖点）。

---

## 3. 逐节撰写思路

### I. Introduction（约 1 页）

四段递进：

1. **动机**：末端全路径水平约束的真实任务（端托盘/焊接/喷涂）。一句点难点——加入约束后可行空间从"体积"坍缩为嵌入的低维流形，可行解稀缺 + IK 多分支需连续。
2. **范式局限（不提 cuRobo）**：主流做法是把约束写进优化目标做软惩罚 + 非凸局部下降，由此产生两个结构性失效：约束退化、初值敏感。**这是去 cuRobo 化的关键落点。**
3. **洞察 + 方法**：与其改优化器，不如把约束满足**从优化过程内解耦**到优化前（贴近流形的多样种子）+ 优化后（硬验收门）；进一步发现——这条可靠流水线本身就是数据引擎，于是封装成自进化闭环。
4. **贡献列表**（见 §4）。

**Intro 末尾放 teaser 图（Fig.1）**：左=约束流形薄面 + 坏种子落在流形外收敛到违约局部最优；右=多种子贴近流形 + 硬验收门 + 失败回退箭头。一图讲清两个创新。

### II. Related Work（约 0.75 页，四类 + 明确定位）

用设计主文档的四类分法，每类结尾一句"我们与之的差异"：

- **采样式约束规划**（CBiRRT/TSR、cpRRTC、McVAMP）：它们**在搜索中投影**到流形；我们把约束**移出搜索**（前移+后验），换 GPU 吞吐。
- **IK 方法**（IKFlow、IKLink、RelaxedIK）：是我们**种子构造的一个组件**，非全部。
- **轨迹优化**（CHOMP/STOMP/TrajOpt/GPMP/cuRobo）：**两个失效模式的来源**；我们不改它，做前后处理。
- **学习式种子/扩散**（DiffusionSeeder、PRESTO、Diffuser、MPD）：它们是**单向一次性**、面向一般规划、学最优轨迹；我们是**带验收+兜底+失败回流的闭环**、面向流形约束、学**可修复种子分布**。

**关键差异化句**（放本节末，反复打磨）：
> Unlike prior learned-seeding pipelines that map a generative sample to an optimizer in a single open-loop pass, our seeder is embedded in a loop whose hard-verification gate and rule-based fallback guarantee that adding learning never degrades the success rate below the verified baseline.

### III. Problem Formulation & Structural Limitations（约 1 页）

1. **约束轨迹优化建模**：优化变量、平滑目标、约束（起点/目标/**核心水平约束 $z_{ee}^\top z_w\ge\cos\epsilon$**/限位/连续性）。全文只在这里定义一次水平约束。
2. **流形视角**：$\dim\mathcal{M}=n-m$，形式化难点 1（体积→薄面）、难点 2（IK 多分支需连续）。
3. **范式的两个结构性局限**：放 Observation 1/2（§0.3）。强调是软约束优化**这一类**的属性，cuRobo 只是其中一员——**一个字都别提 cuRobo**。

### IV. System Design（约 2 页，见 §2 结构）

- **IV-A 总览（合并成功的关键）**：先一段 + Fig.2（三段流水线 + 五阶段闭环 + 回退箭头），明说 pipeline 同时是 executor / data generator / verifier-with-fallback。
- **IV-B 解耦约束满足（创新 A，写得扎实自信，这是已验证部分）**：
  - 每段标注回应哪个 Observation（前移↔Obs.2 初值，后验↔Obs.1 退化）。
  - 前移：重点讲基准种子机理——笛卡尔位置线性插值 + 姿态限制在水平族内做 goal-relative twist 插值 $Q_i=Q_g T_y(\theta_i)$ + 逐点连续 IK 分支选择的代价式；强调"一步同时处理保持水平 + 分支连续"。种子族用表格列。
  - 批量修复：一句带过，用范式化措辞 "the optimizer refines each seed within its basin"。
  - 后验硬门 + 水平优先选择：硬验收项 + 多级排序键（对齐→起点跳变→关节突跳→twist 平滑→路径代价）；强调把"优化成功"与"约束满足"解耦。
- **IV-C 桥接（压到一段）**：Method 有两个局限（串行 IK 慢 / 覆盖受人工规则上限）；而它每次运行都产出带标签数据 → 于是闭环。让读者觉得闭环是"逼出来的"。
- **IV-D 自进化闭环（创新 B，拔高部分）**：
  - 学习目标重定义（单独成段，最有说服力）：学 repairable-seed distribution 而非 optimal-trajectory distribution；正样本 40/40/20 由此决定。
  - 扩散种子模型：输出表示、条件分层、1D U-Net、约束损失（$L_{level}$ 是核心约束的可微版）、独立噪声采样 K 条的理由。
  - 失败是一等公民 + Success Critic：失败**不喂扩散**（会抬升失败概率），训 critic 预测 $[P_{success},\hat e_h,\hat d_{coll},\hat n_{iter}]$ 做筛选。
  - 五阶段闭环 + 下界不退化保证（本节 punchline）：学习失败必然回退已验证规则种子 → 引入学习**永不低于**规则基线。

### V. Experiments（约 1 页，见 §5）

### VI. Conclusion & Limitations

一段总结两个咬合创新；Limitations 正面写核心假设待验证、当前 primitive 障碍、固定机器人——作为 future work 自然入口。

---

## 4. 贡献列表（Intro 末尾，逐字建议）

三点，分别对应"已验证范式 / 闭环架构 / 可证伪假设"：

1. **A decoupled paradigm** that moves constraint satisfaction out of the soft-penalty objective into manifold-aware pre-seeding and a hard post-verification gate, addressing two *structural* limitations of optimization-based planning (constraint degradation, initialization sensitivity) — validated on hardware (SR5: 20/20 obstacle-free, 18/20 cluttered) and in simulation (44→92 / 100).
2. **A self-improving closed-loop system** in which the constraint pipeline serves simultaneously as executor, labeled-data generator, and verifier-with-fallback; a diffusion seeder learns the *repairable-seed distribution* while a success critic consumes failures — with an architectural guarantee that learning never degrades the verified success-rate lower bound.
3. **A falsifiable hypothesis and initial baseline** for whether learned seeds cover the optimizer's convergence basin more efficiently than hand-designed rule seeds under a fixed time budget, with a full evaluation protocol and ablations.

**写作纪律**：贡献 (1)(2) 有证据、说满；(3) 是假设+初步基线，**不要把"扩散超过规则"写成已完成结论**。审稿人会尊重这种诚实。

---

## 5. 实验设计（横向对比 + 纵向消融）

**核心判断：绝不能只做自己的消融。** 三篇最接近的前作（DiffusionSeeder、PRESTO、McVAMP）都是"横向对比 + 纵向消融"两层展开；没有横向对比的学习类规划论文基本过不了 ICRA/IROS。**完整闭环系统作为一个整体参赛。**

### 5.1 文献给出的三条硬规律（来自 referpaper/ 实验章节精读）

1. **核心对比是"成功率 vs 时间预算"曲线，不是单点成功率。** DiffusionSeeder / PRESTO 都以此为头号图，画法是**扫优化器的计算旋钮**：PRESTO 扫 trajopt 迭代数(1–8)、cuRobo 扫尝试数 Natp(1/10/100)、DiffusionSeeder 扫迭代数 Niters(25–475) 和种子数 K(1–32)。DiffusionSeeder 头条："DS-50 比 cuRobo-100 快 12× 且成功率高 10%"——**匹配成功率下比速度**。
2. **采样式约束规划器"投影保证约束满足"，所以不报残差误差，只比成功率+规划时间。** cpRRTC、McVAMP 约束满足是 binary。这带来公平性问题（见 5.4）。
3. **时间/成功率用 mean / 75th / 98th 分位报，不是只报均值；实机对比精简 baseline。** cuRobo、DiffusionSeeder 都报分位数（尾部延迟才是真痛点）。实机成本高，只跑最强的 1–2 个 baseline。

### 5.2 横向对比：baseline（按范式选代表）

| 范式 | Baseline | 文献出处 | 对比要回答什么 |
|---|---|---|---|
| **优化器裸跑（软约束）** | cuRobo 原生种子 + 大 $\lambda_{level}$ | DiffusionSeeder/McVAMP | 前移+后验相对"纯软惩罚"的增益（规范化 CR7 44→92） |
| **采样式约束规划（投影）** | McVAMP 或 cpRRTC / OMPL 约束 RRT-Connect | McVAMP/cpRRTC | 约束满足"理论最干净"的对照；换吞吐/覆盖是否值得 |
| **单向学习种子** | DiffusionSeeder 式（扩散→cuRobo，无回退无 critic） | DiffusionSeeder | 证明"闭环+验收+回退"优于"单向 pipeline"——**最关键的对手** |
| **纯采样规划（无约束基线）** | RRT-Connect / BIT* | 几乎所有论文 | 下限参照 |

- 纯 IK 方法（IKFlow/IKLink）**不做端到端 baseline**（解子问题）。IKLink 放 Related Work 定位，或借它的 reconfiguration 数作连续性指标对照。
- **最关键的一组**：完整闭环 vs 单向学习种子。二者同下游优化器、同扩散种子思路，差别正好是创新 B（验收门 + critic + 回退 + 失败回流），能干净隔离"闭环"本身的贡献。

### 5.3 对比维度（比什么）——附文献确切定义

1. **最终成功率（主指标）**：collision-free ∧ 末端位姿误差 < 容差（cuRobo 5mm/5%；DiffusionSeeder δt=5mm、δr=2.86°）**+ 全路径水平约束在容差内**（我们的任务定义）。
2. **约束满足质量（我们的强项，重点做）**：全路径**最大水平误差**(度)、**水平违约率**。投影类（McVAMP）约束是 binary、不报残差，所以**我们报连续水平误差分布反而是差异化优势**。McVAMP 实机用**关节跟踪误差**(度)作约束可行性代理(mean<10°,max 12°)，可借。
3. **固定时间预算下 Success@K / success-vs-time 曲线（学习分支主战场）**：100ms/300ms/1s 预算，扫 K 和去噪步数。
4. **规划时间**：mean / 75th / 98th 分位。相对采样式约束规划的速度卖点。
5. **候选多样性/覆盖（支撑"覆盖更广"）**：借 MPD 的 **Waypoint Variance (VAR)**（轨迹间逐点 L2 方差）作多模态度量 + IK 分支数 + 避障拓扑数。区别于单向学习种子的证据。
6. **执行质量 + 连续性**：cuRobo 的 max jerk / max accel / motion time；IKLink 的 **reconfiguration 数**（关节速度超限即计一次突跳）。

### 5.4 公平性陷阱（必须处理）

我们（软优化+验收，不保证约束）和投影类约束规划器（McVAMP/cpRRTC，保证约束）直接比成功率不完全公平。处理办法（文献通行做法）：**分维度扬长**——用"成功率 vs 时间预算曲线"体现我们的速度/覆盖优势；用"约束误差分布"坦诚展示我们是软约束+验收（在容差内即可）而非硬保证。**正文明确写清这个 trade-off**，审稿人认可这种诚实。

### 5.5 场景与任务（借文献命名体系增强可比性）

- **约束类型分级**：借 McVAMP 的 **LP/LPO/PP/PPO** 命名 + **TSR** 表达。我们的"保持水平"= 约束 roll/pitch、放开 yaw/z 的 TSR，属 PPO/LPO 类。
- **障碍密度扫描**（cpRRTC/McVAMP 通用）：无障碍 → 稀疏 → 密集（窄通道），每档 **100 个随机 start-goal**（文献标准样本量）。
- **真实端水任务**（借 IKLink 焊接/拧螺丝/关阀 + MPD Panda Shelf 的末端保向代价）：对应端托盘/焊接，做实机 demo。
- **机器人**：仿真 CR7(7-DOF)，实机 SR5(6-DOF)。实机精简到最强 1–2 个 baseline。

### 5.6 纵向消融（内部，证明每个组件有用 + 下界不退化）

- 规则种子 only（第二幕基线）
- + 后验硬门（vs 无后验）— 隔离创新 A 后端
- 纯扩散种子 only（无 critic 无回退）= DiffusionSeeder 式
- 扩散 + critic
- 扩散 + critic + 规则回退（**完整闭环**）
- 去组件：−水平损失 / −碰撞引导 / −多样性筛选（借 PRESTO 三个 ablation claim）
- **时间匹配对照**（借 IKLink 的 MultiGIK×30 思路）：给规则种子同等 wall-clock 预算，证明闭环优势不只是"算得多"

数据展示单调链：mixed_fallback ≥ rule-only ≥ 各单项 → 证明"引入学习永不低于规则基线"。

### 5.7 实验矩阵一览

| 实验 | 方法集 | 场景 | 主指标 | 对应论点 | 状态 |
|---|---|---|---|---|---|
| E1 横向-仿真 | 完整闭环 vs cuRobo原生 / McVAMP / 单向学习 / RRT-Connect | CR7，LP/LPO/PP/PPO × 障碍密度，各100对 | 成功率+约束误差+时间分位 | 系统整体优越性 | 待补 |
| E2 时间预算曲线 | 完整闭环 vs 单向学习 vs 规则 | CR7 密集障碍 | Success@K vs 100ms/300ms/1s | 创新B核心假设 | 待补（假设） |
| E3 消融 | 内部6档 + 时间匹配对照 | CR7 混合 | 成功率单调链 + VAR多样性 | 下界不退化 + 各组件贡献 | 待补 |
| E4 实机 | 完整闭环 vs 最强1-2 baseline | SR5 端水/焊接，无障碍+多障碍 | 成功率（20/20、18/20 已有） | 工程可信度 | 部分已有 |

**诚实分层**：E1/E4（创新 A + 系统）写成已验证结论；E2/E3（创新 B）写成"假设 + 初步基线 + 完整协议"。E2 的核心假设——"固定预算下扩散 K 条种子能否比规则种子更高效覆盖收敛域"——目前**尚未拿到正面结果**。

---

## 6. 图表规划（6 页至少 4 图 1-2 表）

- **Fig.1** teaser：流形薄面 + 两创新一图（Intro）。
- **Fig.2** 系统总览：三段流水线 + 五阶段闭环 + 回退箭头（IV-A 开头，跨栏）。
- **Fig.3** 种子构造：twist 插值 + 连续 IK 分支选择示意（IV-B）。
- **Fig.4** 主结果：Success@K vs 时间预算曲线（Experiments，借 DiffusionSeeder/PRESTO 画法）。
- **Table I** 相关工作四类对比矩阵（投影/吞吐/闭环/约束类型/学习目标）。
- **Table II** 消融（借 PRESTO/IKLink 的表格组织）。

配色与可读性可用 dataviz 规范统一（本仓库有 dataviz skill）。

---

## 7. 写作顺序建议

**Method(IV) → Experiments(V) → Problem(III) → Related Work(II) → Intro(I) → Abstract。**
先写最有把握、证据最实的 IV-B（创新 A）稳住地基，最后写 Intro/Abstract 提炼主张，避免开头把话说太满。

---

## 8. 会议目标与时间线

| 会议 | 时间 | 截稿 | 模板 | 适配度 |
|---|---|---|---|---|
| **IEEE ICRA 2027** ⭐ | 2027，首尔 | **约 2026-09-15** | IEEEtran conference（本仓库已备） | 主攻，时间与格式最合适 |
| CoRL 2026 | 2026-11，Austin | 通常已过 | 自有模板 | 学习角度备选（需换模板 + 补 E2 正面结果） |
| IEEE RA-L | 滚动期刊 | 无固定截稿 | IEEEtran journal | 可扩展成期刊版 |

主攻 ICRA 2027。若截稿前补出 E2 正面结果（扩散在固定时延超过规则种子），可把贡献 (3) 从"假设"升级为"验证结果"，分量最大；否则按诚实分层投稿，贡献 (1)(2) 已足够扎实。

> 提交前务必到 ICRA 2027 官网核对确切截稿日期、页数限制和 PaperPlaza 规则——会议日期可能变动。
