# Memory Bank — Universal Agent Rules

> This file applies to all AI coding agents working on this project.
> Read this file and all files under `memory-bank/` at the start of every session.

## File Structure

| File | Purpose |
|------|---------|
| `RULES.md` | This file — rules (immutable, do not modify) |
| `projectbrief.md` | Project scope, vision, requirements |
| `productContext.md` | Product context, user goals, experiment framing |
| `systemPatterns.md` | Architecture, experiment workflow, design decisions |
| `techContext.md` | Tech stack, environments, runtime constraints |
| `activeContext.md` | Active work, recent changes, next steps |
| `progress.md` | Completed work, in-progress items, known issues |
| `dependencies.md` | Key package/runtime versions |
| `observability.md` | Logging, metrics, Zeus/NVML observability |
| `technicalDebt.md` | Known debt and follow-up items |
| `contextCoverage.md` | Memory-bank coverage status |
| `events.md` | Important runtime and experiment event flows |

## Required Protocols

### Session Start
1. Read `memory-bank/RULES.md`.
2. Read `memory-bank/activeContext.md`.
3. Read `memory-bank/progress.md`.
4. Read additional files as needed for the task.

### Update Triggers

| Event | File(s) to Update |
|-------|-------------------|
| Experiment result added | `activeContext.md`, `progress.md`, `observability.md` |
| Architecture/workflow change | `systemPatterns.md` |
| Environment/runtime change | `techContext.md`, `dependencies.md` |
| New user preference | `activeContext.md` |
| Sweep coverage expanded | `contextCoverage.md` |

### Session End
1. Capture what changed.
2. Capture decisions made.
3. Capture next steps.
4. Ask whether the memory bank should be updated if major changes occurred.

## Update Rules
- Do not modify `memory-bank/RULES.md` after setup.
- Do not store secrets, tokens, passwords, or private keys.
- Use date stamps like `[2026-03-11]` when adding session-specific facts.
- Keep files concise and avoid unnecessary duplication.
- Prefer stable facts and clearly label temporary or incomplete findings.
