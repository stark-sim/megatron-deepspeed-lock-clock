# AGENTS.md

## Memory Bank Protocol (Required)

This project uses a Memory Bank system in `memory-bank/` for cross-session context continuity.

### At Session Start — ALWAYS:
1. Read `memory-bank/RULES.md`.
2. Read `memory-bank/activeContext.md`.
3. Read `memory-bank/progress.md`.
4. Read additional memory-bank files as needed for the task.

### During Work — Update When:
- Experiment result added → update `activeContext.md` + `progress.md` + `observability.md`
- Architecture/workflow decision made → update `systemPatterns.md`
- New dependency or runtime fact learned → update `techContext.md` + `dependencies.md`
- New user preference learned → update `activeContext.md`

### Never:
- Modify `memory-bank/RULES.md`
- Store secrets in memory-bank files
- Skip reading the memory bank at session start
