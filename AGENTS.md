# Project collaboration instructions

## Scope and current facts

- Repository: `Solvement/26-Deep-Learning-Project-1-Kaggle`.
- Competition: https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/overview
- Development and CI use CPU. Real training may use Kaggle Notebook GPU, Google Colab, or a local GPU, subject to verified competition rules, availability, performance, and cost. Keep training platform-independent.
- The competition requirements have not yet been retrieved. Read `docs/competition.md`; never invent architecture constraints, deadlines, dataset layouts, or submission columns.
- Two humans will maintain this project. Their GitHub usernames are not yet configured. Do not invent CODEOWNERS entries or count an agent review as a human approval.

## Work isolation and conflicts

- Each cloud task is isolated already. Use the existing checkout. Do not create a Git worktree unless the user explicitly asks.
- Inspect `git status` and preserve pre-existing changes before editing, switching branches, or resolving conflicts.
- For an authorized coding task, use one topic branch and one coherent PR. Do not push directly to `main`.
- Divide concurrent tasks by files or modules, state ownership, and integrate through PRs. Concurrent agents must not edit the same files without coordination.
- Before updating a branch, fetch the target and inspect changes. Use a merge when that preserves shared history; rebase only branches whose history you alone control. Never force-push or discard another contributor's work without explicit authorization.
- Resolve conflicts by checking both intended behaviors. Rerun affected validation. Escalate conflicts that require a product, architecture, or experiment decision.

## Development and validation

- Read `docs/cloud-development.md` for the CPU development environment and the selectable GPU training backends.
- Run `python -m pip check` and `python scripts/smoke_environment.py --output-dir /tmp/kaggle-dl-smoke` after changes to the environment smoke workflow.
- This smoke check validates dependencies and CPU training only. It is not a competition baseline or evidence of competition compliance. Add focused tests when actual data and model code is introduced.
- Keep datasets, checkpoints, credentials, notebook outputs, and submission files out of Git. Use ignored paths and private artifacts consistent with competition rules.
- Do not print environment or credential values. Reuse existing authentication, checking names/presence and the required operation before requesting more credentials.

## PR review and merging

- For an authorized PR task, include the problem, changes, actual validation results, relevant competition constraints, and remaining limitations.
- Attach every PR created or actively reviewed to the task using the available Codex artifact tool.
- Agents may prepare changes, inspect diffs, run checks, suggest fixes, and provide review findings. A separate human maintainer supplies the required approval.
- Never approve your own PR, impersonate a reviewer, bypass branch protections, or dismiss another person's review merely to merge.
- Merge only when the user has authorized merging, required checks pass for the current revision, the PR is conflict-free, and required approvals are valid. Use squash merge when repository settings permit it.
- Changes to dependencies, workflows, competition assumptions, or merge policy need explicit attention in the PR description.

## Releases and Kaggle submissions

- Do not submit to Kaggle, publish a dataset/notebook, enable GPU training jobs, alter repository security settings, or spend cloud resources without task authorization.
- No automatic Kaggle submission is configured. Confirm the rules and build/validate a real competition pipeline before adding delivery automation.
- Record the commit, configuration, seed, environment, dataset version, validation score, and artifact checksum for each real experiment.
