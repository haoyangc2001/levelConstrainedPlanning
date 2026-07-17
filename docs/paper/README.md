# Paper: SR5 Level-Constrained Planning

Conference-paper workspace for the level-constrained trajectory planning /
closed-loop learning–optimization system in this repository.

## Layout

```
docs/paper/
  main.tex          # paper skeleton (IEEE conference format, section stubs mapped to the design docs)
  references.bib    # starter bibliography (constrained planning, IK, trajopt, diffusion seeding)
  figures/          # put figures here (\graphicspath already points to it)
  IEEEtran/         # official IEEEtran package from CTAN (class + BibTeX styles)
```

`main.tex` loads the class as `\documentclass[conference]{IEEEtran/IEEEtran}`
and the BibTeX style as `\bibliographystyle{IEEEtran/bibtex/IEEEtran}`, so the
template is self-contained — no system-wide IEEEtran install is required.

## Build

```bash
cd docs/paper
latexmk -pdf main.tex          # preferred (runs pdflatex + bibtex as needed)
# or manually:
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Output: `main.pdf`.

## Target venues

Today's date context: mid-2026. The 2026 IEEE robotics deadlines (ICRA 2026,
IROS 2026, RSS 2026) have already passed, so the nearest realistic target is
listed first.

| Venue | When / where | Paper deadline | Format | Notes |
|---|---|---|---|---|
| **IEEE ICRA 2027** | ~mid-2027, Seoul, Korea | **Sep 15, 2026** (first submission, PaperPlaza) | IEEEtran conference (this template) | Nearest deadline; flagship robotics venue; strong fit for constrained manipulation planning. |
| **IEEE IROS 2026** | Oct 2026, Hangzhou, China | passed (spring 2026) | IEEEtran conference | Same template; consider IROS 2027 if ICRA 2027 is missed. |
| **CoRL 2026** | Nov 9–12, 2026, Austin, TX, USA | typically closed by mid-2026 | CoRL style (separate) | Best fit for the **diffusion-seed / success-critic learning** contribution; uses its own template, not IEEEtran. |
| **RSS** | mid-year | ~late Jan/early Feb | RSS style (separate) | Single-track, selective; good for the seed-construction + manifold-planning story. |
| **IEEE RA-L** | rolling (journal) | rolling | IEEEtran journal | No fixed deadline; can be presented at ICRA/IROS. Switch `\documentclass` to `journal` and `bare_jrnl.tex` layout. |

### Recommendation

- **Primary target: ICRA 2027** — deadline Sep 15, 2026, and this template is
  exactly the required format. This is what `main.tex` is configured for.
- If the learning-loop results (diffusion + critic beating rule seeds under a
  fixed time budget) mature in time, **CoRL** is a strong alternative for that
  angle, but it needs its own style file (swap the template).

Verify the exact deadline, page limit, and PaperPlaza submission rules on the
official ICRA 2027 site before submitting — conference dates move.

## Section map

The `main.tex` stubs mirror the repository's design documents:

- **Problem Formulation** ← `docs/design/机械臂带末端位姿约束的轨迹优化.md`
  (constraint-manifold formulation, level constraint, IK-branch continuity).
- **Method / Seeds + Repair** ← rule seed families, CuRobo repair, level-first
  selection (`level_planner_core/{rule_seed,repair,constraints,validators}.py`).
- **Closed-Loop System** ← `docs/design/末端约束扩散学习模型设计.md` and
  `docs/guides/project_mainline.md` (diffusion seed model, success critic,
  data-generation → learning → validation → fallback → update loop).
- **Experiments** ← the CuRobo benchmark and SR5 stress-test results.
