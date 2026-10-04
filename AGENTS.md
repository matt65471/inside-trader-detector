# Agent guidance

Before planning or implementing work in this repository, read
[`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md). It records the agreed scope,
architecture, data sources, constraints, and phased plan for the informed-flow
detector.

The most important guardrails are:

- Keep the system read-only. Do not place trades or add execution code.
- Model whether a public trade appears informed and whether following it after a
  realistic delay would have been profitable; do not claim to identify legal
  insiders.
- Do not add wallet linking, reinforcement learning, or a trading bot unless the
  user explicitly expands the scope.
- Store broadly before scoring; scores must be recomputable.
- Ask before undertaking a large new build unless the user explicitly says to
  start implementation.

