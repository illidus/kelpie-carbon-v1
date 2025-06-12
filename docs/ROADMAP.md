# Kelpie‑Carbon Roadmap  
_Last updated: 2025‑06‑11_

---

This **ROADMAP.md** is the **single source of truth** for outstanding technical work.  
*Rule of thumb*: every pull‑request must tick at least one box below (or add a new one).

> **How to use with Cursor / Claude 4**  
> 1. Copy the code‑block shown under the next unchecked item.  
> 2. Paste it as a prompt in Cursor.  
> 3. Accept the generated patch, run `pytest`, commit.  
> 4. Tick the checkbox here in a follow‑up commit.

---

## Track 1 · Refactor & Repo Layout

- [ ] **T1‑001** Move modules into 4‑package layout (`core/`, `data/`, `validation/`, `reporting/`)
  ```text
  # Cursor prompt – Refactor skeleton
  1. mkdir -p src/kelpie_carbon/{core,data,validation,reporting}
  2. git mv src/kelpie_carbon_v1/indices.py src/kelpie_carbon/core/indices.py
  3. For every module in kelpie_carbon_v1 that isn't tests, relocate into the four‑package tree; update imports.
  4. Add __init__.py files exporting public symbols.
  5. Update pyproject.toml to point to new namespace.
  ```

- [ ] **T1‑002** Unify configuration into **`config/kelpie.yml`**
  ```text
  # Cursor prompt – Single YAML config
  CREATE config/kelpie.yml merging:
    * validation/config.json
    * validation_config.json
    * hard‑coded constants in research_benchmark_comparison.py
  Write loader: kelpie_carbon.core.config.load() returns OmegaConf DictConfig.
  Replace all json loads with this.
  ```

- [ ] **T1‑003** Add type‑safety & lint
  * Adopt **ruff**, **mypy**, **black**, **isort** via `pre‑commit`.  
  * Add `pyproject.toml [tool.ruff]` to mirror Black prefs.

---

## Track 2 · Validation Layer

- [ ] **T2‑001** Implement `ValidationResult` & metric helpers (MAE, RMSE, R², IoU, Dice).

- [ ] **T2‑002** Config‑driven thresholds in `kelpie.yml → validation:`.

- [ ] **T2‑003** Functional validation CLI
  ```text
  # Cursor prompt – Validation CLI
  1. new file src/kelpie_carbon/validation/cli.py with Typer app:
       kelpie validate --dataset data/val --out validation/results
  2. Implement metrics in metrics.py.
  3. ValidationResult (Pydantic) saved as JSON & pretty Markdown via Jinja template.
  4. Hook into poetry scripts: [tool.poetry.scripts] kelpie = "kelpie_carbon.cli:app"
  ```

- [ ] **T2‑004** Replace narrative benchmark script
  * Make `research_benchmark_comparison.py` read latest validation results  
    and benchmarks from YAML; **exit ≠ 0** if any metric fails.

---

## Track 3 · Tests & CI

- [ ] **T3‑001** Profile & cache heavy fixtures; slice data to minimal sample.

- [ ] **T3‑002** Enable parallel execution & markers
  ```text
  # Cursor prompt – Speed‑up tests
  1. Create tests/conftest.py with session‑scoped fixtures sentinel_tile(), rf_model().
  2. Monkeypatch time.sleep & httpx.get in integration tests.
  3. Add pytest.ini:
       addopts = -n auto -m "not slow" --cov=kelpie_carbon --cov-report=term-missing
  4. Mark >10 s tests with @pytest.mark.slow.
  ```

- [ ] **T3‑003** Coverage gate ≥ baseline ‑ 1 %.

- [ ] **T3‑004** CI matrix & nightly job
  ```yaml
  # Cursor prompt – GitHub Actions CI
  jobs:
    lint:   {runs-on: ubuntu-latest, steps: [ {run: ruff .} ] }
    type:   {runs-on: ubuntu-latest, steps: [ {run: mypy src/} ] }
    unit:   {runs-on: ubuntu-latest, steps: [ {run: pytest -m "not slow"} ] }
    validate: {runs-on: ubuntu-latest, steps: [ {run: kelpie validate --dataset test_data} ] }
    docs:   {runs-on: ubuntu-latest, steps: [ {run: mkdocs build --strict} ] }
    nightly:
      if: ${{ github.event_name == 'schedule' }}
      runs-on: ubuntu-latest
      steps:
        - run: pytest -m slow
  ```

---

## Track 4 · Documentation

- [ ] **T4‑001** Adopt **MkDocs Material**; auto‑generate API docs with `mkdocstrings[python]`.

- [ ] **T4‑002** Validation CLI writes Markdown into `docs/reports/`; MkDocs picks it up.

- [ ] **T4‑003** Keep this ROADMAP.md up to date  
  *CI fails if a PR touches code but not this file.*

---

### Changelog template

When ticking a box, add to PR description:

```markdown
### ✔ Completed tasks
- [x] T1‑001 Move modules into four‑package layout
```

Happy shipping! 🚀
