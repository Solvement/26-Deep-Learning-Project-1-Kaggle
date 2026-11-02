# Scratch-to-scratch distillation preregistration draft

Issue 7 / PR 8. Recorded 2026-11-02. This is one proposed experiment, not an
implemented trainer, admission receipt, compliance certification or qualified
submission. No KD training, soft-target generation or numerical test has run.
Keep existing source, checkpoints, gates, receipts and the original confirmation
study unchanged. The campaign's resource limits and 2026-11-05 05:00 UTC hard
stop apply. Confirm the current rules cover this own-team derived-target method
before execution; a fresh student initialization alone does not establish that.

## Hypothesis and fixed recipe

An independently initialized residual student may benefit from the relative
class probabilities of the team's strongest frozen scratch teacher. This could
improve generalization without enlarging the inference model. Teacher errors
can also be transferred, and no hidden-score improvement is assumed.
[Hinton et al., section 2](https://arxiv.org/html/1503.02531v1) motivates softened
targets with a hard-label term and temperature-squared gradient scaling.
[Furlanello et al., sections 3-4](https://proceedings.mlr.press/v80/furlanello18a/furlanello18a.pdf)
studies fresh students of equal or similar capacity. This proposal combines
those ideas; it is not an exact BAN replication. Paper results from other
architectures/datasets or ensembles are not project results.

Freeze these defaults in raw primary/control/repeat configurations before any
new outcome. There is no temperature, coefficient, teacher or architecture grid.

| Item | Proposed fixed value |
| --- | --- |
| Student and matched hard-label control | WRN16-5, dropout0, 10 classes; 4,289,754 total/trainable parameters each |
| Training | Fresh scratch, 160 epochs, source batch128, primary seed2026, split seed2026, official40k only |
| Existing recipe | SGD lr0.1/min0.0001, warmup5, momentum0.9, weight decay0.0005, Cutout16, MixUp alpha0.2/probability0.5, EMA0.995, label smoothing0; otherwise parsed `configs/wrn16_5.json` |
| New KD settings | Temperature `T=4.0`, soft-loss weight `alpha=0.5`, constant through training; control `alpha=0.0` |
| Native precision and selection | Explicit native BF16, FP32 master/EMA and FP32 losses; existing dev-based online/EMA checkpoint rule; fixed `center80_shift20` primary policy |

WRN16-5 is a near-capacity student chosen for a feasible step budget, not a claim
that it is equivalent to WRN34-3. Its matched control isolates the KD loss from
architecture and training-duration changes. The completed old WRN16 run is
historical context; a fresh control through the new engine is required for a
method-attribution claim. Same-architecture WRN34/300 would be a different,
currently unadmitted proposal and is not a silent fallback if this fails.

## Teacher identity and source boundaries

Use only the frozen primary2026 WRN34-3/300, 4,166,298 parameters, selected
epoch298 **online `model`** weights, trained from scratch on the same40k pool:

```text
checkpoint SHA256 53d8d64df899ebab79ae77144ff36e912c45e65178c0b50bd9af381dc48187bc
freeze SHA256     b6314d7f8b05f3f1978c682115d9cb4185646a32771db2826f95b675161045a3
official archive d7def227362a3c930c86a9ab1ad9180c15ac5477847f4b5c34657bec61ff2a60
split SHA256      73c9dbbf99a8c85944a314e1679e6116da066b7a4f40cd794e10aceb6bafea7e
40k indices SHA   b8472b8b293ad1aacad02abb7fcb7bfe8d37fd8065f280f675dee6ae8f9d2afd
```

The private teacher path is resolved from
`results/research/wrn34_3_300_delivery/frozen.json`, not selected by best-looking
repeat results or filename. Verify its exact bytes and complete existing
training/source/split lineage before trusting any derived targets. The teacher
was selected using development; this is exploratory reuse of development, not
new independent confirmation.

- Soft targets may be produced only for minibatches whose source IDs are in
  the fixed40k training pool. MixUp donors must also belong to that pool.
  Development may only score/select; confirmation and hidden-test loaders are
  absent from the trainer. No hidden-image transduction or pseudo-labeling.
- Never use `stability_seed2027`, any renamed copy of its SHA, any repeat2027
  weights, their logits, targets, descendants or teacher ensembles. The selected
  primary teacher is the same fixed teacher for a future fresh student27 repeat.
- Teacher weights/BN buffers stay unchanged in `eval()` with gradients disabled
  and no optimizer, EMA, BN recalibration, feature alignment or teacher updates.
  Construct/load the teacher before reseeding student initialization; prove that
  no teacher weights, optimizer, EMA or RNG state initialize the student.
- The professor ZIP has no extra images or labels; it is the same official
  input. Derived logits remain private and add no externally sourced examples.
  Default implementation uses online targets and saves no logits cache.

At inference retain only one student residual CNN, with total and trainable
parameters verified <=5M. The temporary teacher/student/EMA training memory and
compute must be disclosed and budgeted; do not report their sum as the deployed
model or claim the simultaneous training state contains <=5M parameters.

## Exact view and loss contract

Apply the existing crop/flip/normalization/Cutout and MixUp once. Let `x_m` be
that final student input and `q_y = lam*one_hot(y)+(1-lam)*one_hot(y_perm)`;
without mixing, `lam=1`. The frozen teacher receives **exactly `x_m`**, including
the same Cutout mask, normalized values and donor pairing. One plain teacher
forward supplies targets: no teacher TTA and no separate random weak view.
`softmax(teacher(x_m)/T)` is generally different from mixing the probabilities
of the two unmixed images. Clean per-image cached logits cannot substitute for
targets of these augmented/mixed inputs.

For batch size B, student logits `z_s` and detached teacher logits `z_t`:

```text
log_p_t = log_softmax(float32(z_t.detach()) / T, dim=1)
log_p_s = log_softmax(float32(z_s) / T, dim=1)
L_hard = lam*CE(float32(z_s), y) + (1-lam)*CE(float32(z_s), y_perm)
L_soft = T*T * KLDiv(log_p_s, log_p_t, reduction="batchmean", log_target=True)
L = (1-alpha)*L_hard + alpha*L_soft
```

The divergence is `KL(p_teacher || p_student)`: sum over10 classes, mean over
actual B, not mean over B*10. Hard CE uses temperature1; both soft distributions
use the same T. Compute log-softmax/KL/CE outside autocast in FP32. Do not
softmax twice, reverse KL arguments, average logits/probabilities interchangeably,
detach student logits, omit T squared or clamp a bad loss to conceal nonfinites.
[PyTorch2.12 KLDivLoss](https://docs.pytorch.org/docs/2.12/generated/torch.nn.KLDivLoss.html)
documents the log-input convention and `batchmean` reduction. The coefficient
defaults are controlled choices, not purported optima from either paper.

## Admission, controls and prospective comparison

1. Implement a new separately reviewed typed trainer/preflight/benchmark path.
   Preserve the frozen core and every already bound method. A method name,
   marker edit or additional field in an old receipt is not KD evidence.
2. Run numerical/equivalence checks in Linux CPU CI, with no local Torch job
   during another active GPU task. Then one <=180s supervised actual-image
   preflight after the GPU lease is free: complete40k resident source pool,
   teacher identity/forward, student/EMA/SGD/proofs,3 warmup+20 timed successful
   updates and all state/gradient/BN checks inside whole-step timing. Discard
   preflight state. Native BF16 capability excludes emulation.
3. Admit formal KD only when the fresh uninstrumented measured guard
   `1.5*median_step_seconds*313*160 + 300` fits the3h job cap, actual remaining
   UTC-day6h budget and hard stop. The measurement includes the teacher and all
   proofs. Account for model load, per-epoch dev, checkpoint I/O and complete
   benchmark; increase a deficient overhead allowance before outcomes, never
   reduce the safety factor or split a job to bypass the cap. Profile timings,
   old student-only timings and paper FLOPs cannot admit KD. Budget the fresh
   hard-label control and any repeat separately; do not start a study whose
   required comparison cannot reasonably finish in the remaining campaign.
4. Complete the same5000-image native-v3 suite: clean/NLL,5 stress cases,3
   policies, unchanged BN, full source/config/seed/data/checkpoint/update and
   elapsed-time evidence. Check the unchanged campaign baseline gate for fixed
   weighted policy, plus an analogous prospective gate against the **matched
   hard-label control and WRN300 incumbent**: `(clean_delta>=0.001 and
   stress_delta>=-0.002) or (stress_delta>=0.005 and clean_delta>=-0.002)` for
   each comparator. Also report every case regression and NLL; stress is a
   synthetic proxy, not a hidden-distribution diagnosis.
5. Use [the incumbent comparison plan](incumbent-comparison-plan.md) for the
   fixed WRN300 local-BF16 same-policy reference. Its aggregate helper has no
   promotion authority and may need a new typed KD adapter. Before a repeat,
   require a separately audited original-ID paired comparison to both matched
   control/incumbent (10,000 resamples,5 views clustered per image), record
   uncertainty without treating it as a new holdout, and freeze the primary
   checkpoint/policy, all source/config hashes, comparator decisions and exact
   planned seed2027 recipe. Then fresh scratch27 must complete the identical
   full benchmark and prospective gates. Only frozen primary2026 student
   predictions may reach a new audited delivery path.

The seed2027 experiment repeats student optimization conditional on the same
fixed teacher. It does not measure teacher-training seed variance or supply an
independently trained second teacher. Do not describe it as an independent
repeat of the entire teacher-plus-student pipeline.

## Implementation checks required before admission

| Boundary | Required evidence |
| --- | --- |
| Loss math | FP64 small-tensor analytic reference for forward KL and gradients; identical logits give near-zero KL; additive-logit-shift invariance; `T=1` identity and T-squared gradient scaling; B=1 and short last batch64; class mean vs batch mean discrimination; alpha0 equals hard loss exactly |
| Finite safety | Ordinary/extreme finite logits, invalid T/alpha, NaN/Inf in either side; reject nonfinite loss/grad/state without updating student/EMA; finite FP32 reference vs BF16 forward; no teacher gradients |
| View/data ownership | Teacher sees byte/value-identical final mixed student tensor; lambda0/1 and no-mix cases; same donor IDs; reject dev/confirmation/hidden IDs and wrong teacher/split/hash; no clean-logit cache substitution |
| State and replay | Fresh student initial-state hash equals its matched control under same seed; teacher parameters/all BN buffers unchanged after23 steps; every student BN counter advances once per train step; real positive finite student gradients, SGD and successful-step-only EMA; fixed RNG replay/unchanged teacher eval RNG; model/optimizer/EMA finite and shape checks |
| Delivery boundary | Typed proof links the teacher lineage and student training to benchmark/freeze/repeat/predictions; strict2000 official IDs/hash, actual whole-team quota, duplicate and ambiguous-upload guards; exclude teacher/repeat weights from prediction and final artifact |

No implementation or successful check is claimed here. In particular, the old
`submit_verified` allowlist and current research uploader must not be renamed,
relabeled or weakened to accept an unsupported KD artifact. Until new lineage,
benchmark, repeat and delivery validation exist, this plan permits no upload.
