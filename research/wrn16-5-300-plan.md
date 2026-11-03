# WRN16-5 / 300: one schedule-length experiment

Issue7 / draftPR8, prospective2026-11-03. This is a new scratch40k experiment,
not continuation of the completed160-epoch checkpoint. No300-epoch timing,
training, accuracy, prediction or submission exists at registration. Existing
core, old preflights/configurations, all receipts and confirmation stay frozen.

## Evidence and fixed controls

The [WRN paper, Implementation details](https://arxiv.org/html/1605.07146v4)
uses200 CIFAR epochs with step reductions at60/120/160. It supports width/depth
tradeoffs, not a claim that our300-epoch cosine recipe will improve. The
[SGDR paper, section4.2](https://arxiv.org/html/1608.03983v5) establishes that
learning-rate schedules matter for WRNs; its large-model results do not predict
this competition. This experiment adds no restart or new optimizer.

The project's WRN34/160 and WRN34/300 scored0.869 and0.878. Their training
platform/precision also differ, so this is motivation, not isolated duration
causality. The completed WRN16/160 is the matched architecture/learning-engine
parent. Fixed native-v3 original-BN weighted references are:

| Reference | selected | clean | NLL | mean synthetic stress | noise | blur | contrast | brightness | resize |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WRN16-5/160 parent |141 EMA|.9610|.1243274645|.79264|.7060|.5134|.9542|.9578|.8318|
| WRN34-3/300 incumbent |298 model|.9654|.1175310477|.79716|.7768|.4394|.9594|.9648|.8454|

Parent benchmark SHA256 `c5d457e6e249052851b042c5f7fa19267cb44e62613ecb8b28ff55b6238cb211`,
checkpoint `426a986e34a8179e94fe3065eaf8e0db444dab7bbc52f236f35dec75b055aacb`.
Incumbent benchmark `e3c586afc302d1422c80eb3831562f6b67f684d53c7b2b5390c9d4623001570b`,
checkpoint `53d8d64df899ebab79ae77144ff36e912c45e65178c0b50bd9af381dc48187bc`.
The preflight binds those complete18-case same-context/split/pixel JSON records;
it never reloads their tensors or evaluates images.
The parent was retrospectively not prioritized after the recorded incumbent
comparison; no later prospective gate is attributed to its historical run.

Only change raw parent `wrn16_5.json` epochs160 to300. Fix WRN16-5/dropout0,
4,289,754 total/trainable parameters, B128/313 updates per epoch, SGD/Nesterov
lr.1/min.0001, warmup5, momentum.9, WD.0005, original crop/flip/normalization,
Cutout16/Mixup.2 probability.5, EMA.995, original checkpoint selection and native
BF16/FP32 master weights. Only seed differs between the new primary26/repeat27
configs. Warmup remains5 epochs; cosine is recomputed over300 epochs rather
than appending140 epochs at the old terminal LR. Thus this tests the complete
longer schedule, not extra low-LR steps alone. Expected formal workload is
93,900 updates/zero skips/12,000,000 source examples, single view.

## New bounded engineering entry

`preflight_wrn16_300.py --run UNIQUE --seed 2026` performs only an engineering
preflight. No arguments show help. It owns its schema/script/argv/nonce and
authenticates canonical lease, live PID/creation times, source/config/data
binding and actual OSJob/session membership before runtime/Torch. It reuses
only the frozen `preflight_core_control.core_step(..., future_trainer='frozen')`
and generic accounting/guard/input primitives, never an old script validator.

Use the official resident40k uint8 pool and original GPU randperm, B128,
3warmup+20timed actual successful updates. Time the full original augmentation,
gather/Mixup/loss/backward/SGD/EMA/metrics, every gradient/state finite check,
13BN host reads and actual parameter/EMA probes. Staging/H2D is included in
the supervisor/setup charge as in the unchanged parent protocol; data is not
restaged every step. WRN16 has44 gradient tensors and210 combined model/EMA/
SGD state tensors after each update, not WRN34's98/480. Verify actual counts,
all23 updates, nativeBF16/no emulation, FP32 master/EMA, unchanged epoch-end-only
EMA buffer rule, source identity hashes and complete environment. Save no
weights/RNG cache and discard all state. RNG consumption remains the original
parent sequence; no additional random draw is introduced by proofs.

Reserve exactly180 seconds under concurrency1/day21600s/job10800s and the
2026-11-05 05:00UTC hard stop. Unknown cleanup keeps the lease. Numerical
correctness does not automatically admit formal training. The fixed guard is
`1.5*median_full_step_seconds*313*300+300`; it must fit10800, the actual day
remaining and hard-stop time. With300s overhead median must be<=.07454739084s.
The old160 preflight .0493127s would arithmetically imply7245.69s, but is not
new300 timing or admission. No multiplier/overhead relaxation or split-resume
workaround. Formal setup/dev/checkpoint/benchmark remain charged by supervisor.
This WRN16 timing can never admit the separately rejected WRN34/300 recipe.
Each budget read uses one captured current UTC day; a live preflight crossing
midnight prorates its actual elapsed time by its overlap with the current day.
The stop timestamp must equal2026-11-05 05:00UTC, not a later edited deadline.

Repeat27 requires a separate fresh preflight with `--seed 2027 --frozen-manifest
PATH`, after the qualified primary's actual pair and freeze. Its decision must
already pin exact raw27 config and this actual preflight source map. Primary26
preflight is never repeat27 admission. Freeze validation here is a pure
source/hash/chronology boundary, not a replacement for the full delivery audit.

## Formal and decision chain

Root launches only after real admission. Reuse unchanged `run_local.py
--trainer frozen --precision bf16 --config research/configs/wrn16_5_300.json`
and the exact new run directory. Its original native receipt is preserved;
do not add a schema/mode, relabel a new controller or reuse a cloud attestor.
The new preflight does not train, launch that command or authorize an upload.

Complete5000dev/18 native-v3 cases and tensor/history/source checks are required.
Freeze the above parent and incumbent references before formal. Report parent
comparison as matched diagnostic, without changing its historical decision.
For candidate and fresh27, retain the original fixed weighted baseline gate
and [prospective incumbent protocol](incumbent-promotion-plan.md), including
each stress-case loss<=1pp and complete conditional image-cluster bootstrap.
No best-policy choice, hidden examples, confirmation, repeat weights or public
score may rescue a failed fixed policy. NLL and all signed case changes stay
visible. Synthetic stress is not hidden accuracy; TA's real0.867 despite
stress.91168 and SAM's0.871 despite clean.9674 demonstrate proxy limitations.

The ordinary core benchmark/paired-v2/delivery-v2 loaders support legal depth,
width and completed epochs from actual configs; native repeat validation
expects the real original local receipt. No new trainer/marker/benchmark/
predictor/uploader is needed. Produce current-incumbent paired evidence and
separate legacy-baseline paired-v2; freeze exact primary/policy/raw27 config
and planned source before fresh27, then only that frozen primary can predict.
Existing quota/official2000ID/hash/intent/reference/duplicate/unknown protections
remain mandatory. Report adapters need an explicit eventual new300 row;
prepared engineering evidence is not a score or delivery qualification.
