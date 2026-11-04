# Independent synchronous TA training and typed benchmark — Issue7 / PR8

Prepared prospectively2026-11-02. This adds source/configuration/tests only, not
a controller, allocation, speed admission, trained model or score. Controller
and actual official-image preflight are separately owned and reviewed. The
frozen TA component and its implementation plan remain byte-for-byte unchanged.

## Fixed recipe and parent relationship

Own scratch WRN34-3,4,166,298 total/trainable parameters, dropout0,160 epochs,
batch128, seed2026 or exact separately authorized2027 repeat, official40k train /
5000dev with split2026. The base is frozen `research/configs/wrn34_3.json`, raw
SHA `2901f0a7d717083c8811aac9cff1910c7a7493eb3c474c260bb9b7f7eb45df02`.
New configurations add only `ta_space=fixed_standard` and `ta_strengths=31`;
primary and planned repeat differ only in seed. Original LR0.1/min0.0001,
warmup5/cosine160, SGD Nesterov0.9/WD0.0005, EMA0.995, Cutout16, Mixup0.2/prob0.5
and raw/EMA accuracy/NLL selection are retained. Native localBF16 uses the same
disabled GradScaler/finite-gradient update helper; CPUFP32 is for math tests.
No SAM, JSD, SE, extra views, changed TTA or external data/pretrained weights.

For each epoch retain the original device `torch.randperm(40000)`. The matching
CPU optimization-pool original uint8 pixels are indexed in that same order.
For each minibatch, synchronously apply exactly one CPU/PIL TA operation per
original32x32 RGB image using an independent explicitPCG64(seed). Stack uint8,
transfer to the device, then call unchanged parent crop/flip/normalize/Cutout
and original Mixup/loss/SGD/EMA. There is no DataLoader, worker, queue or pipeline
framework. Test actual runtime before assuming this synchronous design fits.
Only optimization-pool sources and within-batch Mixup donors are used. Dev5000
appears only in epoch selection/full dev benchmark, never as an augmentation
source. Original confirmation and hidden inputs are never evaluated.

TA sampling never consumes the parent's Python, NumPy-global or Torch streams.
After each epoch record14 operation counts totaling40000, source/view/update
counts and independent TA RNG fingerprint. All160 epochs must total6.4M source
examples,6.4M single views,50,080 successful updates/forward passes and zero
skips. Exact state includes model/EMA/SGD, disabled-scaler state, original four
parent RNG fields and the independentPCG64 state. Explicit resume authenticates
raw/effective config, source, split, complete partial history, actual tensor /
momentum topology and TA RNG; preflight states are rejected and never refitted.

This prospective recipe is the previously delivered implementation plan's
WRN34/pre-crop/+Mixup adaptation. The older unexecuted `trivialaugment-plan.md`
proposed WRN22/post-crop/noMixup; it is preserved as historical research and is
not silently substituted here. Paper scores are not project results.

## Interfaces and state audit

`train_trivialaugment.py --config PATH --output-dir FRESH --data-dir data
--device cuda --precision bf16` supplies math only. Explicit `--resume` and
`--stop-after-epoch` are epoch-boundary facilities for a separately authorized
controller/test, not a way around job/day/hard-stop limits. Controller must hold
the canonical lease and establish its actual OS boundary before importing or
executing Torch. No unattended GPU call is authorized by these new files.

`train_step` is also the actual-math entry for a separately reviewed preflight.
It includes CPU TA, transfer, parent augmentation/Mixup, one forward/loss,
backward/SGD finite checks and EMA. It returns private tensors to its caller;
controllers may persist only audited aggregate proof, not images/prediction rows.

`benchmark_trivialaugment.py --checkpoint PATH --baseline PATH --output FRESH
--data-dir data --device cuda` accepts complete native-v3 evaluation only.
Before scoring, validate the actual selected and final model/EMA tensors,
all SGD momentum tensors and optimizer settings, parent/TA RNG state, raw and
effective config, method/core source, history/counters and selected weight epoch.
It uses the explicit `research_trivialaugment_training_v1` schema, never a
renamed shadow/SAM/ResizeMix marker. Full last-state/tensor auditing is performed
by this typed engine; an outside JSON-only audit must not claim tensor reloads.

## Fixed benchmark decision and delivery boundary

Run unchanged core.score on all5000 independent dev images, all3 policies and
all6 cases, batch128/nativeBF16. Preserve source/split/context/pixel hashes and
all per-case NLL/accuracy. Original baseline gate remains unchanged; only
`center80_shift20` is primary. Same weighted incumbent is frozen WRN34-3/300
checkpoint `53d8d64df899ebab79ae77144ff36e912c45e65178c0b50bd9af381dc48187bc`,
local benchmark `e3c586afc302d1422c80eb3831562f6b67f684d53c7b2b5390c9d4623001570b`.
Load/verify its actual bound source/checkpoint evidence using the existing typed
shadow audit, without retraining or changing it.

Apply the unchanged prospective plan raw SHA
`10e59e56f8c2a3c1f9ae0ad2681c3db5b3129ffd2d33c6f4d3087710e2860ec7`:
old fixed weighted gate, plus same-policy incumbent clean/stress mean criteria,
plus no individual stress decrease beyond1pp. NLL and noise delta are explicit;
neither a mean nor an alternate policy can rescue a failed case. This same
benchmark supports exact fresh2027 repeat metadata without pretending seed27 is
primary2026. Any passing flags remain screening evidence: new typed TA pairing,
primary freeze before repeat, full repeat and explicit delivery support are
still required. This implementation neither extends uploader allowlists nor
predicts/submits hidden data.

## CPU validation scope

Pillow/stdlib tests may check config/source/RNG/count boundaries without Torch.
Torch math tests run only in the approved LinuxCI or separately coordinated CPU
window: forced-identity TA must exactly match parent loss/SGD/EMA and parent RNG;
real TA changes pixels but replays from saved independent state. Also test
source/split/raw/preflight-state rejection and actual tensor/momentum/RNG
corruption. These synthetic small tests are not GPU throughput or accuracy.
