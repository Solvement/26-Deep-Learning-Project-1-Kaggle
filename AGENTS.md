# Project collaboration instructions

## Scope and current facts

- Repository: `Solvement/26-Deep-Learning-Project-1-Kaggle`.
- Competition: https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/overview
- Development and CI use CPU. Real training may use Kaggle Notebook GPU, Google Colab, or a local GPU, subject to verified competition rules, availability, performance, and cost. Keep training platform-independent.
- Competition requirements are verified in `docs/competition.md`: residual CNN, <=5,000,000 trainable parameters, scratch training, official competition inputs only. Do not substitute external CIFAR data or pretrained weights. Deadline verified in Kaggle UI: Nov 6, 2026 23:59:59 EST. Course report due Nov 9, timezone unspecified.
- Current owner is Kevin Yang (ky2738), GitHub Solvement, Kaggle kevin1220, registered team Saber. Currently solo; a teammate may join later subject to the written three-member limit. Do not invent CODEOWNERS entries or count an agent review as a human approval.

## Team operating model

- Read `docs/team-workflow.md` for task ownership, acceptance, review, experiment records, delivery, and escalation.
- Link substantive changes to an Issue with one owner and observable acceptance criteria. Use the development or experiment Issue template.
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
- Keep datasets, checkpoints, credentials and submission files out of Git. Full generated outputs stay ignored. The user requested completion of the course PDF, which requires an executed results notebook: a curated notebook containing only actual summary metrics, configurations and plots may be committed after inspecting every cell/output for data, weights, prediction rows and credentials. Do not publish raw artifacts inside notebook cells or image outputs.
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
- The user explicitly authorized local training, use of available free resources, code pushes, public repository visibility, and validated Kaggle submission. Do not request those permissions again. Paid resources and merging main are not authorized by that statement.
- Formal local campaign: fixed official 40k train / 5k development / 5k one-time confirmation split (seed 2026), three matched 160-epoch recipes, selection by development only, one confirmation, then frozen-recipe full-50k training. `run_campaign.py` enforces artifact checks and a three-hour wall-time cap. Do not retune after confirmation or expose hidden labels.
- Keep original private Kaggle input-export jobs private. One independent free GPU repeat was attempted but aborted before training because CUDA was unavailable; no seed-stability score exists for that attempt. Local training is the verified backend.
