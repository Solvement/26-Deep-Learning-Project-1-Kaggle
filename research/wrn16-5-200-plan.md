# WRN16-5 / 200: independent schedule experiment with a fresh160 control

Issue7 / draftPR8, preregistered2026-11-03. No200 result or admission exists.
The completed300 preflight is an immutable timing rejection:23 correct updates,
median.09585395s and guard13801.0289s>10800. It stays closed. This experiment
actually changes the full schedule to200; it does not rename/resume a300 job,
borrow its admission or lower a multiplier, proof requirement or resource cap.

## Question, references and fixed recipe

Does a200-epoch cosine schedule improve this legal shallow/wide architecture
over a fresh160-epoch control while also improving the fixed incumbent? The
[WRN paper, Implementation details](https://arxiv.org/html/1605.07146v4)
uses200 CIFAR epochs and step reductions at60/120/160. Its schedule differs
from ours. [SGDR section4.2](https://arxiv.org/html/1608.03983v5) motivates
testing schedule effects; it does not establish that our200 recipe will win.
No restart, optimizer, augmentation or teacher is added. Paper accuracy is not
a project result. TA's actual0.867 despite stress91.168% and SAM's0.871 despite
clean96.74% show that these local proxies do not reliably rank hidden accuracy.

Only raw parent `research/configs/wrn16_5.json` epochs160 changes to200.
WRN16-5/dropout0 remains4,289,754 total/trainable parameters. Keep B128,
SGD/Nesterov lr.1/min.0001, warmup5, momentum.9, WD.0005, crop/flip/normalization,
Cutout16, Mixup.2 probability.5, EMA.995, label smoothing0 and original online/
EMA development selection. Source is the same official40k, splitseed2026,
5000dev, nativeBF16 with FP32 master/EMA. Two new raw configs differ only in
training seed2026 versus2027. Recompute the entire cosine over200 epochs;
do not initialize from160 weights or append40 low-LR epochs. Expected formal
work is62,600 updates/zero skips/8,000,000 source examples, single view.

Fixed historical native-v3, original-BN, weighted references before outcomes:

| Reference | selected | clean | NLL | stress | noise | blur | contrast | brightness | resize |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical WRN16-5/160 |141 EMA|.9610|.1243274645|.79264|.7060|.5134|.9542|.9578|.8318|
| Incumbent WRN34-3/300 |298 model|.9654|.1175310477|.79716|.7768|.4394|.9594|.9648|.8454|

Historical parent benchmark SHA256
`c5d457e6e249052851b042c5f7fa19267cb44e62613ecb8b28ff55b6238cb211`,
checkpoint `426a986e34a8179e94fe3065eaf8e0db444dab7bbc52f236f35dec75b055aacb`.
Incumbent benchmark `e3c586afc302d1422c80eb3831562f6b67f684d53c7b2b5390c9d4623001570b`,
checkpoint `53d8d64df899ebab79ae77144ff36e912c45e65178c0b50bd9af381dc48187bc`.
Both complete18-case JSON/context/pixel identities are preflight inputs.
The historical parent was retrospectively not prioritized; preserve that
decision. It is neither a newly executed control nor a later prospective reject.

## New actual-image preflight only

`python research/preflight_wrn16_200.py --run UNIQUE --seed 2026` requests
one new engineering preflight. No arguments show help. It has its own schema,
script, argv, nonce, canonical lease and authenticated OSJob/session before
runtime/Torch. It calls unchanged `preflight_core_control.core_step` with
`future_trainer='frozen'`, never an old script-specific owner validator.

Require original official40k uint8 residency, actual GPU randperm, B128,
3warmup+20timed real successful updates. Keep gather, crop/flip/Mixup, losses,
backward/SGD/EMA/metrics, all44 gradient tensors, all210 model/EMA/SGD state
tensors,13BN counters, norms and parameter/EMA probes within whole-step timing.
The same original staging/H2D setup is charged to the180-second supervisor;
resident data is not staged every step. Record2944 unique source rows, zero
skips, actual Laptop GPU/nativeBF16, FP32 master/EMA, full actual environment
and identity hashes. No extra RNG draws, accuracy, saved weights, logit cache,
confirmation or hidden inputs. Discard all preflight state and reseed formal.

Exactly180 seconds, concurrency1, day21600s/job10800s and hard stop
2026-11-05 05:00UTC remain fixed. Unknown cleanup retains the lease. Recompute
each budget from one captured UTC timestamp; prorate actual live elapsed over
midnight. The200 guard is `1.5*median_full_step_seconds*313*200+300`; median
must be<=.11182108626s for the3h limit, and actual remaining day/stop may be
tighter. Neither old160 nor rejected300 speed admits this job. Their speeds
differed despite identical recorded software/GPU/RNG-order metadata, so clock/
thermal/load/algorithm causes remain unknown. No profiling/retry branch is added.

## Preregistered order and stop conditions

1. New200 primary26 passes its own actual preflight and fresh scratch formal
   training, with complete5000dev/18-case typed benchmark. Fixed primary policy
   is `center80_shift20`. Require the unchanged original weighted baseline
   gate and [incumbent mean plus every-case floor](incumbent-promotion-plan.md).
   Any failure stops this candidate before spending on control or repeat.
2. Only a passing primary gets actual current-incumbent10000 original-image
   cluster pairing and separate legacy-baseline flip paired-v2 evidence.
   Freeze exact200 primary checkpoint/policy/source, raw27 config and all
   planned repeat sources **before any control or27 outcomes**. Also freeze
   explicit fresh160 control formal/preflight artifact targets, original raw
   config/source closure and its validation-only purpose. No repeat/control
   weights, scores or logits may replace or influence hidden predictions.
3. Execute fresh160 seed2026 contemporary control, using original160 config
   and original control entry's **own new real preflight** and source/actual
   budget admission. It is not300 completion, old160 replay evidence, a200
   repeat, a submission candidate or a replacement primary. Its original
   `run_local --trainer frozen --precision bf16` receipt stays unchanged.
4. Root audits complete control evidence/source/config/tensor/context/pixels
   and chronology against the frozen targets. Compare the frozen200 primary
   to this fresh160 under the same weighted policy: clean gain>=.001 with
   stress change>=-.002, or stress gain>=.005 with clean change>=-.002;
   additionally no stress case loses more than.01. Record NLL and every signed
   case delta. A control comparison failure rejects200; do not switch to the
   control, alter gates/policy or choose a more favorable historical control.
5. Only after control supports200, run a separate actual200 seed27 preflight
   `--seed 2027 --frozen-manifest PATH`, then fresh scratch200 repeat with exact
   frozen recipe. Require complete original/incumbent point and case gates;
   root also reports its same-policy comparison to the frozen fresh160 control.
   Only the already frozen qualifying200 primary26 may ever predict.

The decision JSON supplied to unchanged delivery-v2 freeze must explicitly
contain `planned_repeat_config_sha256`, this200 preflight's complete
`planned_repeat_source_sha256`, and `planned_fresh160_control` with schema
`wrn16_200_fresh160_control_plan_v1`, role`development_control`, seed2026,
epochs160, trainer`frozen`, precision`bf16`, config path/hash, effective config,
original control `source_snapshot()` map, controlled `run_path` and
`preflight_run_path`, and `hidden_prediction_authorized:false`. Freeze these
targets while absent. Later validators must permit their completed existence.
The200 seed27 preflight verifies these declarations/source bytes and the real
primary legacy pair/freeze. Root separately audits the actual fresh control
and extra gate before repeat and before upload. **The old uploader does not
automatically enforce this new control contract**; no schema/allowlist or old
receipt is changed to pretend otherwise. These additional checks are required
durable independent evidence, not optional metadata or delivery qualification.

Formal runs retain the unchanged original `run_local.py` PID/watchdog protocol;
only this new preflight has an OSJob. Generic core benchmark, incumbent pair,
legacy paired-v2, freeze-v2, fresh native27, predict-v2 and strict submit can be
reused because ordinary depth/width/complete epochs are configuration-driven.
No new trainer/marker/predictor/uploader is introduced. Preserve official2000
IDs/hashes, real UTC team quota, intent/reference/duplicate/unknown protections.
Prepared outputs are not scores. Original confirmation remains untouched.

## Resource planning, not admission

At the rejected300 preflight's observed median, arithmetic would give200 guard
9300.6859s and160 guard7500.5487s. Three formal tasks would total26101.9205s
before separate preflights/pairing/prediction; this is neither measured200
throughput nor a reservation guarantee. Each task gets its own actual admission,
receipt and settlement, potentially across UTC days. If all required control,
repeat and final checks cannot finish before the fixed stop, skip delivery.
Do not split/resume to evade per-job limits or buy new compute. The new report
must explicitly record this experiment and any negative, preserving all old
160/300 recipes, source bytes and public/private distinctions.
