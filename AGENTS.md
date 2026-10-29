# Project collaboration instructions

## Scope and current facts

- Repository: `Solvement/26-Deep-Learning-Project-1-Kaggle`.
- Competition: https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/overview
- Development and CI use CPU. Real training may use Kaggle Notebook GPU, Google Colab, or a local GPU, subject to verified competition rules, availability, performance, and cost. Keep training platform-independent.
- Competition page requirements were verified through Kaggle CLI. Read `docs/competition.md`: residual CNN, at most 5,000,000 trainable parameters, from-scratch training, permitted CIFAR-10 data only, no external pretrained weights/data. The course PDF adds Nov 6 competition / Nov 9 report deadlines (timezone still unconfirmed), an 8-page report, and executed-notebook/public-repository deliverables. Official data metadata is fully audited in docs/data-audit.json; see docs/course-requirements.md and research/charter.md.
- The user is preparing to train individually first; teammates may join later, with a competition maximum of three members. Reviewer identities are not configured. Do not invent CODEOWNERS entries or count an agent review as a human approval. Check official team membership and any deadline/approval requirements before privately sharing competition work with new members.

## Team operating model

- Read `docs/team-workflow.md` for task ownership, acceptance, review, experiment records, delivery, and escalation.
- Link substantive changes to an Issue with one owner and observable acceptance criteria. Use the development or experiment Issue template.
- Follow `research/charter.md` for literature-backed hypotheses, frozen three-way splits, GPU budgets, valid comparisons, and one-time confirmation. The current authorized round is capped by `research/queue.json`; continue within that scope without asking again, and stop at its budget or failed gates.
- Record actual outcomes separately from plans. Pure experiment runs may update an Issue and artifact index without an empty code PR.
- Scope unattended remediation to at most two verified repair attempts before reporting a blocker; this does not activate a background agent or authorize extra spending.

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
- Keep datasets, checkpoints, credentials, and submission files out of Git. Curated executed notebooks with non-sensitive plots and explicitly labeled smoke/validation results are permitted and required by the course; never commit data, model bytes or credentials in them. Use ignored paths and private artifacts consistent with competition rules.
- Do not print environment or credential values. Reuse existing authentication, checking names/presence and the required operation before requesting more credentials. The current vault variable is `Kaggle_API_Token`; the cloud activation script maps it to CLI-standard `KAGGLE_API_TOKEN`. Inspect runtime variable names/readiness as well as proxy secret names before diagnosing a missing credential.

## PR review and merging

- For an authorized PR task, include the problem, changes, actual validation results, relevant competition constraints, and remaining limitations.
- Attach every PR created or actively reviewed to the task using the available Codex artifact tool.
- Agents may prepare changes, inspect diffs, run checks, suggest fixes, and provide review findings. A separate human maintainer supplies the required approval.
- Never approve your own PR, impersonate a reviewer, bypass branch protections, or dismiss another person's review merely to merge.
- Merge only when the user has authorized merging, required checks pass for the current revision, the PR is conflict-free, and required approvals are valid. Use squash merge when repository settings permit it.
- Changes to dependencies, workflows, competition assumptions, or merge policy need explicit attention in the PR description.

## Releases and Kaggle submissions

- Do not submit to Kaggle, publish a dataset/notebook, enable GPU training jobs, alter repository security settings, or spend cloud resources without task authorization.
- The user authorized appropriate Kaggle submissions after full training and validation. Do not ask again within this scope; enforce a conservative research-loop cap of two per New York day plus the actual whole-team cap of five, verify all 2,000 official IDs/order and labels, model/data compliance, provenance and duplicate hashes. No prediction has been submitted yet. The latest priority is training; persistent cloud-service setup is paused.
- Record the commit, configuration, seed, environment, dataset version, validation score, and artifact checksum for each real experiment.
