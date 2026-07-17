# 文档索引

本目录按**用途**分组存放项目文档，而不是把所有 Markdown 平铺在一个文件夹里。

项目主线是闭环学习优化系统：

```text
数据生成 → 模型学习 → 优化验收 → 失败回退 → 数据更新
```

理解项目方向请**先读** [guides/project_mainline.md](guides/project_mainline.md)。其余文档都应作为这个闭环的组成部分来读，而不是彼此无关的工具说明。

```text
docs/
├── README.md      # 本索引
├── guides/        # 使用与工程指南
├── reference/     # 接口 / schema 参考
├── design/        # 系统设计文档（archive/ 存历史版本）
├── plan/          # 工作计划（按背景分 paper / closed_loop_impl）
├── reports/       # 各阶段 smoke / benchmark 报告
└── paper/         # 论文写作素材（LaTeX 骨架、撰写指南、参考文献）
```

## 指南 Guides

- [guides/project_mainline.md](guides/project_mainline.md)：顶层闭环项目方向与工程契约。
- [guides/environment.md](guides/environment.md)：机器、conda、ROS、CuRobo、离线生成与在线验证的无头环境。
- [guides/runtime.md](guides/runtime.md)：在线闭环行为、CLI / Python 入口、学习种子验收与规则回退。
- [guides/dataset_training.md](guides/dataset_training.md)：离线闭环的数据集导出、artifact 指针、扩散 / critic 训练与数据更新流程。
- [guides/ros_adapter.md](guides/ros_adapter.md)：围绕同一规划核心的可选在线任务入口。

## 参考 Reference

- [reference/api_schema.md](reference/api_schema.md)：纯规划核心、CLI、数据集记录与反馈闭环使用的 request / result schema。

## 设计 Design

- [design/末端水平约束轨迹规划与闭环学习进化系统.md](design/末端水平约束轨迹规划与闭环学习进化系统.md)：**当前系统设计主文档**。
- `design/archive/`：历史设计文档，保留以供追溯。
  - [design/archive/机械臂带末端位姿约束的轨迹优化.md](design/archive/机械臂带末端位姿约束的轨迹优化.md)：早期「末端 / 水平位姿约束轨迹优化」设计。
  - [design/archive/末端约束扩散学习模型设计.md](design/archive/末端约束扩散学习模型设计.md)：早期扩散种子学习模型设计。

## 计划 Plans

计划按**背景**分成两类，详见 [plan/README.md](plan/README.md)：

- [plan/paper/](plan/paper/)：**当前计划**——把已完成的系统做成一篇会议论文（论文实验工作计划，叙述版 + 机器可读版）。
- [plan/closed_loop_impl/](plan/closed_loop_impl/)：**历史计划**——闭环学习优化系统的实施计划及其数据 / 接口契约（Phase 0–9，已完成）。

## 报告 Reports

`reports/` 存放各阶段的 smoke / benchmark 报告，按阶段命名（如 [reports/phase8_closed_loop_benchmark_report.md](reports/phase8_closed_loop_benchmark_report.md)）。其中 [reports/source_phase10_training_report.md](reports/source_phase10_training_report.md) 是源项目 Phase 10 的 SR5 成熟离线模型报告与 artifact 摘要。

## 论文 Paper

`paper/` 存放论文写作素材：

- [paper/writing-guide.md](paper/writing-guide.md)：逐节撰写指南、图表规划、去 CuRobo 化纪律与诚实分层。
- [paper/main.tex](paper/main.tex)：IEEEtran 论文骨架（正文多为占位）。
- [paper/references.bib](paper/references.bib)：参考文献。
- `paper/referpaper/`：相关论文 PDF（只读）。
- `paper/IEEEtran/`、`paper/figures/`：LaTeX 模板与图目录。
