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
- All five permitted UTC-day submissions on 2026-10-31 completed scoring: baseline 0.835, architecture 0.854, regularized 0.874, primary final_full 0.871, and exploratory robust_full 0.862. The API confirmed five used and zero remaining. Best observed public entry is regularized; private scores are unreleased. Check the actual whole-team quota before any future authorized upload and preserve every scored receipt/hash. The exploratory 300-epoch recipe was prompted by official public feedback and does not reuse or inherit the 96.64% one-time confirmation result. Preserve the original frozen 160-epoch study and never retune it or rewrite its results after confirmation.
- Keep original private Kaggle input-export jobs private. One cloud seed-repeat attempt aborted before training because CUDA was unavailable; its replacement local seed-2027 repeat achieved 96.36% development accuracy and was excluded from selection, confirmation and hidden-test prediction. Local RTX 5060 training and the separate free T4 robust_full run both completed. Final sanitized evidence is in docs/results.json; data, checkpoints, predictions and full runtime records remain private.

## Five-day follow-on authorization (2026-10-31)

- The owner explicitly authorized merging PR 6; it was squash-merged as e582b7b13c8d18aaebbb451a92896fc497bad0c6. This is not blanket authorization to merge future PRs or bypass required reviews/protections.
- The owner requested five days of autonomous research to maximize score, with local/cloud benchmarks before every new submission. Work is linked to Issue 7, on feat/five-day-autoresearch. Read research/campaign.json and research/README.md, and continue the recorded jobs rather than restarting them. Stop new experiments/submissions at 2026-11-05 00:00 America/New_York, then finalize artifacts and pause the associated automation.
- The user considered paid compute but selected existing Colab first. Use existing Colab entitlement/units and existing available resources; no additional subscription, units, top-up or external GPU purchase is currently authorized. Verify actual balance/availability. Colab CLI is prepared in existing Ubuntu WSL because the official tool does not support native Windows.
- All new runs are exploratory. Never reuse the primary confirmation for tuning or call full-50k overlapping development an independent benchmark. Keep train.py/model.py/data.py/runtime.py frozen, retain all old scores, and keep stability_seed2027 excluded from hidden prediction/upload helpers.
- Promotion requires a completed fixed official development benchmark and predeclared clean/stress thresholds, plus artifact/source/config hashes, parameter budget, official row checks, quota and duplicate/ambiguous-upload guards. Do not spend slots on smoke runs or unqualified candidates merely to fill the daily quota. The independent research uploader now exists and has passed CPU checks plus actual offline validation of the qualified WRN34 delivery; only real API receipts establish submission/scoring.
- The current chat heartbeat is saber-benchmark, every 30 minutes through the stop time. Detailed progress/leases/resource records belong under ignored results/research; only audited aggregate evidence and source enter Git. A successful scheduler configuration is not proof that any GPU job or submission completed.
- WRN34-3 is the first benchmark-qualified follow-on candidate. Its primary seed2026 checkpoint/policy and paired evidence are frozen before independent seed2027 outcomes. The independent research/delivery.py and research/submit_candidate.py paths enforce repeat validation, official CSV hashes, actual quota, duplicate/reference and ambiguous-upload guards. Do not bypass them or extend the original uploader allowlists. GPU prediction and diagnostic work use research/run_diagnostic.py under the shared local lease/budget. No new research submission has yet been scored; inspect actual receipts rather than assuming helper implementation means success.
- ResizeMix completed a source-bound T4 benchmark and passed the point gate, but needs consistent local rebenchmark, paired evidence and an independent seed before prediction. Use paired_compare_v2.py/delivery_v2.py for its versioned audit; never edit files bound to the existing WRN34 freeze. Its independent local seed repeat uses run_resizemix_local.py with explicit FP16 to preserve the primary training method/precision. A separate run_refit.py/refit_delivery.py path supports only already-qualified frozen-core WRN full-50k recipes; a prepared plan is not completed training or independent development evidence. Read actual state/receipts for current progress.
