# One fixed label-smoothing experiment

Issue 7 / draft PR 8, prospective 2026-11-03. Test WRN34-3/160 with epsilon
0.1. This is the only remaining conditional candidate in this registration;
there is no epsilon sweep, policy search, teacher, extra view, or new engine.
No accuracy, speed admission, prediction, or submission is claimed here.
All previously executed sources, negative experiments, and receipts stay frozen.

## Evidence and exact target convention

[He et al., Bag of Tricks, sections 5.2, 5.4 and 5.5](https://arxiv.org/html/1812.01187v2)
evaluate combinations including smoothing and Mixup on ImageNet. They use
epsilon 0.1 and Mixup alpha 0.2; Mixup also changes their training duration.
This motivates a cheap loss-only test, not an isolated result for adding
smoothing to this project's CIFAR/Cutout/probability-0.5 Mixup recipe. No paper
accuracy or ImageNet improvement is used as a competition forecast.

The paper-linked [GluonCV author implementation at commit
567775619f3b97d47e7c360748912a4fd883ff52](https://github.com/dmlc/gluon-cv/blob/567775619f3b97d47e7c360748912a4fd883ff52/scripts/classification/imagenet/train_imagenet.py)
uses the uniform-all-classes convention: on-target is 1-epsilon+epsilon/K,
off-target epsilon/K, then mixes the two targets. This is the convention in
[PyTorch 2.12 CrossEntropyLoss](https://docs.pytorch.org/docs/2.12/generated/torch.nn.CrossEntropyLoss.html).
It differs from the paper's literal equation assigning epsilon/(K-1) to only
the other classes. The pinned author file is a semantic reference; no author
source or dependency is imported by training.

The unchanged core already passes config.label_smoothing to both original
cross-entropy branches. By linearity, its loss equals soft-target CE with

`q = (1-epsilon)*(lambda*one_hot(y)+(1-lambda)*one_hot(y_donor)) + epsilon/K`.

Thus epsilon 0.1 and K=10 give 0.91/0.01 on an unmixed example. Smoothing is
applied once; it does not replace the feature-dependent mixed target. The
registered CPU tests check the formula and gradients, endpoint lambdas and
same-class donors, plus the unchanged original batch AST under this config.

## Fixed recipe and honest historical reference

The new primary raw config differs from frozen `wrn34_3.json` only in
label_smoothing 0.0 to 0.1; the predeclared seed27 raw config differs only in
seed. This entry implements primary26 engineering only; a seed27 config is not
an implemented repeat-admission path. Keep depth34,
width3/dropout0, 4,166,298 total/trainable parameters, 160 epochs, B128,
SGD/Nesterov lr0.1/min0.0001/warmup5/momentum0.9/WD0.0005, original crop/flip,
normalization, Cutout16, Mixup alpha0.2/probability0.5, EMA0.995, split2026 and
original checkpoint selection. Train scratch on the same official40k only;
the independent5000 development set is selection/evaluation. No confirmation
evaluation, hidden inputs, external data, pretrained weights or ensemble.
Formal expected workload is 50,080 direct SGD/EMA updates, zero skips,
6,400,000 source examples, one student view and zero teacher passes.

The historical WRN34-3/160 actually trained locally with native BF16,
FP32 master/EMA, Python3.14.3/Torch2.12.0+cu130/CUDA13.0/cuDNN92000 on
NVIDIA GeForce RTX 5060 Laptop GPU. Its environment JSON SHA256 is
`63f182c467c2d2126e6224363f5378d8105a5f6e272e6dbffad67f205e90d84c`.
Its original PID/watchdog supervisor lacks newer trainer/precision/schema
fields; do not retrofit a modern OSJob receipt. The new preflight has its own
actual OSJob protocol. Process-local DLL PATH removal count may differ;
the recorded software/GPU/native flags must match the fixed environment.

| Fixed weighted reference | selected | clean | NLL | mean synthetic stress | noise | blur | contrast | brightness | resize |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical WRN34-3/160 |152 model|.9632|.1246351960|.78592|.7532|.4286|.9550|.9602|.8326|
| Current WRN34-3/300 incumbent |298 model|.9654|.1175310477|.79716|.7768|.4394|.9594|.9648|.8454|

Historical parent benchmark SHA256
`cf8cfa9fbb103db3ad705a5971b993d187e055dff6e6d0a0a3c1ac5964168690`,
checkpoint `1787237fb5f4566043843f9008516e911b9a6d867bc6ac4720c52448871c7b9a`.
Incumbent benchmark `e3c586afc302d1422c80eb3831562f6b67f684d53c7b2b5390c9d4623001570b`,
checkpoint `53d8d64df899ebab79ae77144ff36e912c45e65178c0b50bd9af381dc48187bc`.
Both fixed complete18-case JSON references have matching local native-v3
evaluation context, split and per-case pixel hashes. The incumbent's original
300-epoch training was T4 FP16; matching local evaluation does not make its
training platform match. The historical160 and new LS160 share the specified
training platform/precision, but the old run is not a newly contemporaneous
randomized causal control or current timing evidence. Report signed differences
without attributing them solely to smoothing. No new fresh160 control is
registered for this final bounded test.

## Actual engineering admission, never borrowed speed

`python research/preflight_label_smoothing.py --run UNIQUE --seed 2026`
only performs primary26 engineering preflight. No args show help. Seed27 and
all frozen-manifest arguments fail before allocation; neither a valid-looking
freeze nor raw27 config can authorize a repeat through this entry. Own schema/script,
argument vector/nonce, canonical lease, parent/child PID creation times, exact
source/config/data binding, and native OSJob/session membership are checked
before runtime, then Torch. Reuse only frozen generic guard/accounting/input
primitives and `preflight_core_control.core_step(..., future_trainer='frozen')`;
never dynamically rebind an old script-specific validator or copy learning math.

Use actual official40k uint8 resident input, original GPU randperm and B128,
3 warm plus20 timed updates. Time the complete gather/augmentation/Mixup,
loss/backward/directSGD/EMA/metrics, all98 gradient finite checks and norm,
all480 state checks (191 model+191 EMA+98 momentum), all31 BN reads,
actual parameter/EMA probes, dtype and allocation checks. Initial staging/H2D
is charged in setup/supervisor wall time rather than invented per-step transfer.
Check 23 actual successful updates/zero skips/2944 unique source rows, actual
native BF16 without emulation, FP32 model/EMA and no GradScaler, one forward
per update, epoch-end-only EMA buffer copying (zero copies in this preflight).
Preserve the parent RNG sequence. Save no weights or reusable RNG state.

Reserve exactly180 seconds. Keep local concurrency1/day21600s/job10800s and
hard stop2026-11-05 05:00UTC. Unknown cleanup retains the lease. All setup,
proofs and actual use are charged; actual elapsed is prorated over UTC midnight
for the live budget read. Numerical correctness is separate from timing:
`1.5*median_full_step_seconds*313*160+300` must fit job/day/hard-stop time.
The unconstrained3h median ceiling is .139776357827476s. Old WRN/KD speeds
cannot admit LS; no multiplier/overhead change, split/resume workaround or
performance profiling rescue. Timing rejection closes this candidate.

## Conditional formal and delivery chain

Root alone may launch unchanged `run_local.py --trainer frozen --precision bf16
--config research/configs/wrn34_3_label_smoothing.json` after current CI,
independent review and actual admission. Preserve its genuine original
PID/watchdog receipt. This preflight never invokes formal training or uploads.
Only one primary26 and, if qualified, one independent fresh27 are registered.

Complete5000dev/18 native-v3 cases with tensor/history/source/config/parameter
and workload audit. Fixed policy is center80_shift20 with original BN. Retain
the old original-baseline gate and all [prospective incumbent rules](incumbent-promotion-plan.md):
clean delta>=.001 with stress>=-.002, or stress>=.005 with clean>=-.002, and
every stress case delta>=-.01 against the pinned incumbent. NLL and all cases
remain visible. A failed point/case gate stops immediately, without pair,
repeat, prediction, epsilon variation or policy substitution.

If qualified, compute the actual10000-draw original-image cluster comparison
with all five stress views kept together, separately from the required legacy
paired-v2 comparison against original-baseline flip. Report all conditional
CIs including zero-crossing; do not silently add a new significance threshold.
Keep the original robustness-driven legacy CI delivery rule. The existing
`paired_incumbent.py` exposes a pure result validator, but no complete offline
native-guard/cleaned-receipt validator. This primary-only entry therefore does
not implement seed27 admission rather than pretending a legacy-only check
proves the new matched-incumbent requirement. If the primary actually qualifies,
prepare a separate minimal repeat-preflight entry and finalize its complete
source before freezing the primary. That freeze must bind actual completed
incumbent-pair path/hash, exact candidate/incumbent policies and checkpoints,
10000 original-image clusters, all CI/marginal/context/pixel/source and closed
native-guard evidence, as well as the separate legacy pair, raw27 config and
exact planned27 source map. No positive-clean-CI threshold is added.
All this precedes every seed27 outcome. Seed27 requires its own actual
engineering preflight; primary26 or other-method speed never substitutes.
Its complete fixed-gate repeat remains mandatory; repeat weights never supply
hidden predictions. If the independent repeat path cannot be completed within
the deadline, stop. A source/hash JSON audit does not reload tensors.

Generic core benchmark/paired-v2/freeze/native-repeat/predict-v2/strict submit
can consume this legal smoothing config without new markers or allowlists;
root must separately verify added incumbent/case/source requirements and real
budget settlement. Only the previously frozen primary may produce2000 official
ID rows. Existing CSV/hash/quota5 perUTCday/intent/reference/duplicate/unknown
guards remain mandatory. No score is promised: TA's stress.91168 still scored
.867, below incumbent.878. Proxies and papers do not establish hidden accuracy.
Allow at most the two formal3h caps plus separately charged preflights,
pair/prediction/audit before hard stop, potentially across UTC days. If that
complete path no longer fits, stop rather than lower proof or repeat standards.
