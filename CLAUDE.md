# CLAUDE.md

This file gives Claude Code project-specific guidance for `/home/yyf/proj`.

## Current Authority

Read these first:

1. `INDEX.md` -- project entrypoint
2. `agents/README.md` -- collaboration SOP
3. `agents/STATE.md` -- current confirmed facts
4. `agents/TASKS.md` -- next actions
5. `workspace/README_先看这个.md` -- daily research workspace

If older archived files conflict with `agents/STATE.md`, trust `agents/STATE.md`.

## Project Position

SafeConf is a single-cell perturbation prediction reliability project.

It is not a new perturbation predictor.

It scores the reliability/risk of existing prediction records:

```text
predicted_effect + historical task features
  -> SafeConf risk/confidence score
  -> prioritize which predictions should be experimentally checked first
```

## Minimal Layout

```text
proj/
├── README.md
├── INDEX.md
├── START_HERE_FOR_GPT.md
├── code/       formal SafeConf code
├── docs/       stable docs, results, beginner guides
├── agents/     current AI collaboration state
├── workspace/  daily research workspace
├── tools/      maintenance scripts and server tools
└── runtime/    temporary output entrypoint
```

Historical material is centralized outside the repository at
`/home/yyf/archive/safeconf/`. Do not use it as current truth unless explicitly
asked to inspect history.

## Core Evidence Boundaries

- Frozen v0.2 is interpretable and fixed; McFarland remains a frozen-v0.2 failure boundary.
- Learned reliability results are a stronger extension, not proof that frozen v0.2 succeeded everywhere.
- E8b is an external benchmark method-error association, not vector-level scoring for 27 architectures.
- Do not claim SafeConf is proven universal for GEARS, CPA, scGPT, or all deep predictors.
- Do not call historical AI outputs final evidence unless backed by files in `docs/实验结果/`.

## Split Protocol

Formal evaluations use 5-fold held-out `(context, perturbation)` pair splits.

In each fold:

- test `(context, perturbation)` pairs do not appear in train;
- the context and perturbation may each appear separately in other train pairs;
- feature computation must use fold-local training statistics only.

## User Preference

The user is learning the project and prefers:

- concrete data examples;
- flowcharts and tables;
- plain Chinese explanations;
- clear path references;
- no vague paper-sounding prose when teaching basics.
