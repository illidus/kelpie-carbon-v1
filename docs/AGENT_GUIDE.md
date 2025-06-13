# 🤖 Agent Guide – Working with Kelpie-Carbon

_Last updated: 2025-06-12_

Welcome, automated collaborators! This guide explains **how AI agents (like ChatGPT, Claude, etc.) should interact with the repository** so that changes remain predictable, reviewable, and high-quality.

---

## 1. General Principles

1. **Read before you write**
   Always inspect relevant files (code, docs, config) _before_ proposing an edit.
2. **Small, atomic PRs**
   Group related changes together; avoid sweeping refactors in a single patch.
3. **Follow the Roadmap**
   All work should map to an open checkbox in `docs/ROADMAP.md` (or add one).
4. **Stay deterministic**
   Prefer pure functions and deterministic pipelines; avoid randomness unless seeded.
5. **Idempotent scripts**
   CLI commands and notebooks should be safe to run multiple times.

## 2. File & Directory Conventions

| Path | Purpose |
|------|---------|
| `src/kelpie_carbon/core/` | Domain logic & shared utilities |
| `src/kelpie_carbon/data/` | Imagery/geo data loaders & preprocessing |
| `src/kelpie_carbon/validation/` | Metrics, validation pipelines |
| `src/kelpie_carbon/reporting/` | Report generation & templates |
| `tests/` | Unit, integration & e2e tests |
| `docs/` | Markdown docs (served by MkDocs) |

*Use these paths when creating new modules.*

## 3. Patch Formatting Guidelines

1. **Use unified diffs** – One file per `edit_file` call.
2. **Context comments** – Replace unchanged code with `// ... existing code ...` (see Roadmap examples).
3. **Explain why** – The `instructions` field should state the intent (not the code).
4. **Keep imports tidy** – Alphabetical, no unused imports (ruff will complain).
5. **Respect linters** – Run `poetry run ruff .` & `poetry run mypy src/` locally before committing.

## 4. Testing Expectations

* Every new module **must** have at least one unit test in `tests/`.
* Long-running tests → mark with `@pytest.mark.slow`.
* Ensure `pytest -m "not slow"` passes in CI (<5 min target).

## 5. Documentation Workflow

1. **Update docs** alongside code.
2. Add new pages under `docs/` and reference them in `mkdocs.yml` nav.
3. Keep the "Last updated" line at the top of each doc.

## 6. Typical Agent Workflow

```
1. Identify Roadmap task → read related code/tests
2. Call read_file/list_dir to inspect context
3. Use edit_file to implement minimal diff
4. Add/adjust tests
5. Update docs & Roadmap checkbox
6. Run tests & linters
7. Commit with clear message
```

## 7. Common Pitfalls to Avoid

- Editing large files blindly → always preview relevant sections first.
- Forgetting to update imports after moving files.
- Introducing non-deterministic randomness (fix with a constant seed).
- Pushing notebooks with execution output; clear before commit (`nbstripout`).

## 8. Need Help?

Open a GitHub Discussion or check the [Contributing Guide](https://github.com/kelpie-carbon/kelpie-carbon-v1/blob/main/CONTRIBUTING.md).

Happy shipping! 🚀
