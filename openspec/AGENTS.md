# OpenSpec Instructions

Instructions for AI coding assistants using OpenSpec for spec-driven development.

## Skill Routing (技能优先)

本文件仅为格式与 CLI 速查，流程契约以 skills 为准。冲突时 skills 优先。

| 意图 | 使用 Skill |
|------|------------|
| 创建/修订可执行提案 (`proposal/design/spec/tasks`) | `openspec-proposal` |
| 按已批准 `Execution Contract` 实施 (`INITIAL`) 或修 implementation finding (`REPAIR`) | `openspec-apply` |
| 审查实现 (`FULL`/`RECHECK`/`DIAGNOSE`) | `openspec-review` |
| 用户明确授权后归档、提交、推送 | `openspec-archive` |

各 skill 硬性前置均要求先读本文件与 `openspec/project.md`，故本文件不可删。

## TL;DR Quick Checklist

- Search existing work: `openspec spec list --long`, `openspec list` (use `rg` only for full-text search)
- Decide scope: new capability vs modify existing capability
- Pick a unique `change-id`: kebab-case, verb-led (`add-`, `update-`, `remove-`, `refactor-`)
- Scaffold: `proposal.md`, `tasks.md`, `design.md` (only if needed), and delta specs per affected capability
- Write deltas: use `## ADDED|MODIFIED|REMOVED|RENAMED Requirements`; include at least one `#### Scenario:` per requirement
- Validate: `openspec validate [change-id] --strict` and fix issues
- Request approval: Do not start implementation until proposal is approved

Skip proposal for: bug fixes restoring spec behavior, typos/format/comments, non-breaking dependency updates, config changes, tests for existing behavior. Unclear → create proposal.

## Directory Structure

```
openspec/
├── project.md              # Project conventions (must read)
├── specs/                  # Current truth - what IS built
│   └── [capability]/
│       ├── spec.md         # Requirements and scenarios
│       └── design.md       # Technical patterns
├── changes/                # Proposals - what SHOULD change
│   ├── [change-name]/
│   │   ├── proposal.md     # Why, what, impact
│   │   ├── tasks.md        # Implementation checklist + Execution Contract
│   │   ├── design.md       # Only if needed (see below)
│   │   └── specs/          # Delta changes
│   │       └── [capability]/
│   │           └── spec.md # ADDED/MODIFIED/REMOVED
│   └── archive/            # Completed changes
```

Stage indicators: `changes/` = proposed, `specs/` = built, `archive/` = completed.

Create `design.md` only if: cross-cutting change, new dependency/data model, security/perf/migration complexity, or ambiguity needing decisions before coding.

## Naming

- Change ID: kebab-case, verb-led, unique (`add-two-factor-auth`; taken → append `-2`).
- Capability: verb-noun, single purpose (`user-auth`, not `user-auth-and-notify`).

## Spec File Format (Normative)

### Scenario format (exact)

CORRECT:

```markdown
#### Scenario: User login success
- **WHEN** valid credentials provided
- **THEN** return JWT token
```

WRONG: `- **Scenario: ...**`, `**Scenario**: ...`, `### Scenario: ...`. Every requirement MUST have at least one `#### Scenario:`.

### Requirement wording

Use SHALL/MUST for normative requirements. Avoid should/may unless intentionally non-normative.

### Delta operations

- `## ADDED Requirements` — new standalone capability. Prefer for orthogonal additions.
- `## MODIFIED Requirements` — paste full updated block (header + all scenarios). Archiver replaces whole requirement; partial deltas drop details.
- `## REMOVED Requirements` — include **Reason** + **Migration**.
- `## RENAMED Requirements` — `- FROM: \`### Requirement: Old\`` / `- TO: \`### Requirement: New\``. Behavior change also needs MODIFIED under new name.

Headers matched with `trim(header)`. If not changing existing requirement, use ADDED instead of MODIFIED.

## CLI Essentials

```bash
openspec list                  # active changes
openspec list --specs          # capabilities
openspec spec list --long      # verbose specs
openspec show [item]           # details; deltas: --json --deltas-only; spec: --type spec
openspec validate [item] --strict                 # comprehensive check
openspec validate <id> --strict --json --no-interactive  # for skills/scripts
openspec archive <change-id> --yes [--skip-specs] # --skip-specs only if no spec deltas
```

Flags: `--json`, `--type change|spec`, `--strict`, `--no-interactive`, `--skip-specs`, `--yes`/`-y`.

## Search Guidance

- Enumerate specs: `openspec spec list --long` (or `--json` for scripts)
- Enumerate changes: `openspec list`
- Full-text: `rg -n "Requirement:|Scenario:" openspec/specs`; changes: `rg -n "^#|Requirement:" openspec/changes`
- Debug delta: `openspec show [change] --json --deltas-only`

## Before Any Task

- [ ] Read `openspec/project.md`, relevant `specs/[capability]/spec.md`, pending `changes/` for conflicts
- [ ] Run `openspec list` / `openspec list --specs`; use `openspec show` before creating specs; prefer modifying existing capability

## Troubleshooting

- `Change must have at least one delta` → check `changes/[name]/specs/**/*.md` + `## ADDED Requirements` prefix.
- `Requirement must have at least one scenario` → must be `#### Scenario:` (4 hashtags).
- Silent parse failure → `openspec show [change] --json --deltas-only`, then `openspec validate [change] --strict`.

Remember: Specs are truth. Changes are proposals. Skills own process; this file owns format.
