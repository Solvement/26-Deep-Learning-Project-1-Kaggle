# WRN34-3/300 full-data refit: predeclared extension

Issue 7 / PR 8. This new route extends the already frozen and independently
repeated WRN34-3/300 primary. Its actual public submission was 0.878, reference
57854294. That is a 40k-parent result, not an outcome of this proposed refit.
The original study, parent/repeat weights, manifests, attestors, validators and
all proof-bound source remain unchanged, including `full_refit_supported=false`.
Each new plan records raw source SHA256 separately from the recorded Git blob
content SHA256 and serialization. Exact bytes are required except for the two
already frozen CRLF files `delivery_v2.py` and `shadow_repeat_attestation.py`:
their original raw fingerprints stay unchanged, and only their explicitly pinned
CRLF-to-LF Git serialization is accepted. No other content/newline differences
are ignored. Later docs-only HEAD changes do not replace the recorded source commit.

The new recipe starts from scratch with seed2026 and uses all 50,000 official
training images. It has zero validation images and never evaluates or selects
using development, confirmation or hidden images. The former confirmation IDs
may participate as training examples only after this recipe is frozen. There is
no claim of independent development accuracy for the refit.

Fix WRN34-3 (4,166,298 parameters), 300 epochs, source batch128 (last batch80),
SGD lr0.1/momentum0.9/weight-decay0.0005/Nesterov, original five-epoch warmup and
cosine schedule, MixUp alpha0.2/probability0.5, Cutout16 and EMA0.995. Reuse the
frozen model/data/augmentation/schedule and shadow optimizer/RNG primitives.
Training is explicit native CUDA FP16 with GradScaler initial scale65536,
growth2/backoff0.5/interval2000. Preserve the original cuDNN BN backend. Final
weights are the online **model at epoch300**, fixed before training; EMA is kept
only as resumable training state. No best-epoch or model-versus-EMA selection.
The original parent happened to select epoch298/model; its weights are never
loaded as initialization. Future prediction retains the parent's separately
frozen local BF16 inference context and center80_shift20 policy.

Each epoch has391 attempted updates and exactly50,000 source examples; 300 epochs
have117,300 attempts and15,000,000 examples. Successful plus skipped updates must
equal attempts; EMA advances only with successful SGD. Finite-loss overflow skips
must lower the scale and preserve model/SGD/EMA probes. Complete final audit also
checks all model/EMA/momentum/scaler/RNG tensors, ordered epoch history, exact
source coverage/counts and final-state equality. Checkpoints and receipts have
new schemas and bind the immutable parent delivery/freeze/typed repeat evidence.

Before formal training, run a new source-bound actual-official-image preflight
with full50k GPU residency, original augmentation/mixing/loss/backward/SGD/EMA,
at most64 normal attempts until23 successful updates (3 successful warmup,20
successful timed), preserving init65536 and recording every natural overflow
skip, scale backoff, finite loss and SGD/EMA preservation probe. The guard uses
the20 successful full-pipeline step durations; all attempted timings and source
coverage are retained separately. A separate injected overflow proof preserves
all online and EMA parameters and the complete SGD state. Training-forward BN
buffers may update normally even when SGD is skipped. All proof costs
are included in step timing; no synthetic or old timing can admit this route.
Preflight scratch state is discarded and formal training reseeds from scratch.
It must fit `1.5 * median_step_seconds * 391 * 300 + 300` within the actual remaining
local budget, match GPU/Torch/CUDA/cuDNN/precision/source/config, and run under the
shared local lease and independent OS process-tree boundary. Each job is at most
three hours, local UTC-day usage at most six hours, with the campaign hard stop
2026-11-05 05:00 UTC. Preflight is at most180 seconds. Unknown cleanup retains the
lease. Resume requires stopped owner/child, verified tree cleanup, unchanged
source/config/inputs and exact epoch-boundary RNG/SGD/scaler/EMA restoration;
every attempt and its conservative budget accounting is preserved. The1percent
overflow bound applies only to the complete300-epoch run; partial epochs retain
exact counts, positive successful updates and finite-state/RNG checks without
misclassifying initial init65536 backoff as a final skip-rate failure.

Existing timing is planning context only: the parent's two T4 FP16 40k/300 runs
took5,259 and5,612 seconds; step-count scaling estimates6,570–7,010 seconds for
50k/300 before a new measurement. The same architecture's local40k/160 BF16 run
took1,867 seconds, a different precision/context that cannot admit FP16. Full50k
refit performance is unknown until actual receipts, and competition improvement
is unknown until an actual scored submission through a separately audited new
delivery/uploader route. Data, weights, RNG states and prediction rows stay private.
