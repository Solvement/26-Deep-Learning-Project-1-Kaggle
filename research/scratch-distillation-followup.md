# Scratch distillation: evidence and execution decision

Issue7 / draftPR8. Research-only follow-up recorded 2026-11-02. This supplements
[the original preregistration](scratch-distillation-plan.md), whose bytes and
fixed recipe remain unchanged. No KD trainer, target cache, GPU measurement,
numerical test, training result or submission qualification is claimed here.

## Decision for the remaining campaign

Retain **one fixed KD recipe as a conditional backup**. Finish the admitted
ResizeMix300 comparison and evaluate the already reviewed TrivialAugment path
first. KD is scientifically defensible, but a new trainer, typed benchmark and
lineage-aware repeat/delivery support still have to be implemented and tested.
Starting that work does not by itself justify spending a GPU slot. If the
matched control, primary KD run and a qualifying fresh student27 repeat cannot
reasonably finish under the existing daily budget and Nov5 05:00UTC stop, skip
KD in this campaign. Do not shrink the evidence requirements to fit the window.

This is a priority judgment, not a measured prediction of benefit. The teacher
is already available, so teacher retraining has zero *incremental* cost; every
student step still needs an additional teacher forward. No runtime or score
gain is established by that observation.

## What the primary evidence establishes

[Hinton, Vinyals and Dean, section2](https://arxiv.org/html/1503.02531v1)
supports combining correct-label loss at temperature1 with softened teacher
targets, using the same soft temperature on both models and multiplying that
term by T-squared. It supports reusing the original training inputs. It does
not identify T4/alpha0.5 as optimal for this project; those remain the existing
controlled choices.

[Furlanello et al., sections3-4](https://proceedings.mlr.press/v80/furlanello18a/furlanello18a.pdf)
shows that a fresh student can outperform a same-capacity teacher, including
some weaker-teacher transfers. Thus teacher top1 accuracy is not a strict
student ceiling. Their CIFAR experiments use different training settings from
our MixUp/Cutout recipe, and this proposal is not an exact BAN reproduction.
Their repeated generations, shared-layer variants and ensembles are outside
this proposal. None of their numerical results predicts our competition score.

An author-owned implementation cross-check is the classic `DistillKL` function
in [Tian et al.'s RepDistiller](https://github.com/HobbitLong/RepDistiller), the
CRD authors' repository, not a BAN/Hinton author implementation. Its
[pinned KD.py](https://github.com/HobbitLong/RepDistiller/blob/b84f547c5db6a35318d4671d7d5c4de74c822403/distiller_zoo/KD.py)
uses student log-softmax, teacher softmax, class-summed/batch-averaged forward KL
and T-squared. Inspected commit:
`b84f547c5db6a35318d4671d7d5c4de74c822403`; raw493-byte file SHA256:
`2a55763457438f6654922ebb0e9fe00a0cf2c88c0fd07d931e41493cf5275f7e`.
No repository models, data, pretrained checkpoints or code were imported.
A verified BAN-author code release was not located in this bounded check;
third-party BAN implementations are not treated as author evidence.

## Keep the exact small experiment

Use the preregistered WRN16-5 student/control: 4,289,754 parameters, dropout0,
160 epochs, B128, seed2026, fixed40k official pool, original crop/flip,
normalization, Cutout16, MixUp0.2/probability0.5, SGD/EMA/schedule unchanged.
KD has constant T4 and alpha0.5; the new matched hard-label control has alpha0.
There is no architecture/temperature/weight search, teacher replacement,
feature-matching branch, extra view, JSD, SAM or successive student generation.

The sole teacher is the frozen primary2026 WRN34-3/300 online-model checkpoint,
epoch298, 4,166,298 parameters, SHA256
`53d8d64df899ebab79ae77144ff36e912c45e65178c0b50bd9af381dc48187bc`.
It was trained from scratch on the same40k. Validate the original freeze,
training/config/source/data/split/40k-index lineage and exact checkpoint bytes
before constructing soft targets. Its local native-BF16 reference is fixed;
T4FP16 training does not make a different-precision evaluation interchangeable.

Same-context historical weighted metrics provide a reason to test transfer:

| Existing primary | Clean | NLL | Mean of five synthetic stress cases |
| --- | ---: | ---: | ---: |
| WRN16-5/160, historical student architecture | 0.9610 | 0.1243274645 | 0.79264 |
| WRN34-3/300, fixed teacher/current incumbent | 0.9654 | 0.1175310477 | 0.79716 |

These are completed native-v3 local-BF16, same5000/policy observations, not a
new matched-control experiment. The old student run cannot replace alpha0
through the new engine. KD must improve past the incumbent, not merely recover
part of the historical student deficit. The incumbent's actual public0.878 is
the current best; no KD hidden prediction has been produced.

If a fresh hard-label student is as good as or better than the teacher, BAN
makes useful transfer *possible*, but provides no assurance here. Retain the
same teacher and recipe for a controlled result; do not select a new teacher
after seeing control outcomes. If the full KD primary fails any prospective
gate, stop this candidate rather than sweeping temperatures or weakening
MixUp/Cutout. A teacher's mistakes or poor targets on mixed/occluded inputs
may transfer too. That is an unresolved risk, not a demonstrated explanation
for this project's public-score plateau.

## Teacher isolation, views and numerical contract

- Resolve only the frozen primary26 identity. Any seed27/repeat checkpoint,
  renamed copy, logits, target cache or descendant is forbidden as a teacher
  or input to hidden predictions. The future student27 is optimization
  replication conditional on the same fixed teacher, not a replication of
  teacher-training variance. Only the frozen qualified primary26 student may
  ultimately supply hidden predictions.
- Construct/load teacher under an isolated or restored RNG boundary, then
  reseed the fresh student exactly as its control. Teacher optimizer/EMA/state
  cannot initialize student weights. Hash equality of control/KD initial
  student state and replay tests must prove this rather than assume it.
- Teacher stays `eval()` and `requires_grad_(False)`, under `no_grad()`, with
  no optimizer, EMA update, dropout sampling or BN recalibration. Verify all
  teacher tensors/BN counters before and after the actual23-step preflight;
  protect them for the full run. Student BN advances once per training update.
- Apply parent augmentation and MixUp once to40k-only donors, then feed the
  **identical final tensor** to student and teacher. Reuse exact donor order,
  lambda and Cutout. Hard targets remain lambda-weighted source labels. The
  teacher distribution on a mixed input is generally not a mixture of teacher
  distributions on unmixed images; a clean-logit cache changes the experiment.
- Both forwards may use the admitted native BF16 context. Cast detached
  teacher and live student logits to FP32 and compute log-softmax, KL and CE
  outside autocast. Use `KL(teacher || student)`, class sum/actualB mean:
  `T*T*kl_div(log_ps, log_pt, reduction="batchmean", log_target=True)`.
  Combine with the original mixed hard CE as `0.5*hard + 0.5*soft`.
  [PyTorch2.12 KLDivLoss](https://docs.pytorch.org/docs/2.12/generated/torch.nn.KLDivLoss.html)
  specifies log-input/log-target semantics and `batchmean`. Keep gradients
  through the student only. Reject nonfinite values; do not clamp away faults.
- Save/recover the parent RNG and actual training state precisely. Teacher
  eval forwards must leave parent Python/NumPy/Torch RNG unchanged. Include
  new trainer/loss/benchmark/controller source and frozen parent dependencies,
  actual teacher lineage and raw configs in the complete source/evidence
  closure. Unsupported old method markers are not KD evidence.

At inference deploy one residual student, with total/trainable parameters
individually verified <=5M. The teacher plus student plus EMA training state
exceeds5M in aggregate; disclose its memory/compute rather than describing all
simultaneously resident training state as a <=5M model. No external examples,
pretrained weights, confirmation or hidden-image targets are allowed.

## Admission and stopping conditions

The original one-time confirmation remains untouched. First implement/review
the minimum separate typed path and run math/state tests in Linux CPU CI.
Do not run local Torch CPU work while another local GPU job owns the lease.
Check alpha0/control equivalence, FP64 KL/gradient references, T-squared factor,
batch reduction including B64/B1, extreme/nonfinite logits, input equality,
teacher immutability, RNG isolation, resume, source/split/hash refusal and
successful-step-only SGD/EMA. These checks have not run for KD.

Only after that, when the GPU is free, make one supervised <=180s official40k
preflight with3 warmup plus20 timed successful steps. Time the full student
step, teacher forward, augmentation, mixed targets, FP32 losses, gradient/state
proofs and synchronization; validate both models and discard preflight state.
No synthetic/profile/student-only timing admits this method.

For313 batches/epoch and160 epochs, the unchanged guard is:

```text
guard_seconds = 1.5 * measured_median_step_seconds * 313 * 160 + overhead_seconds
overhead_seconds >= 300
guard_seconds <= min(10800, actual_remaining_UTC_day_budget, time_to_hard_stop)
```

With exactly300s overhead, the3h cap permits median<=0.1397763578s. This is
arithmetic, not measured KD throughput. Use larger overhead if actual loading,
dev selection, checkpoint I/O and full benchmark need it, and tighten the step
bound accordingly. Native capability, VRAM, current lease/PID identity and
actual budget must all pass. Do not split jobs or reuse rejected300-core
timings to circumvent admission. Reserve each fresh control/KD/repeat as a
separate bounded job: their worst-case upper allocation totals9h before
preflights/pairing/delivery, so they cannot all occupy one6h UTC day. Actual
unused reservations are settled from receipts; this is not a claim of9h usage.

Every completed candidate needs all5000 independent dev images, the same
native-v3 context/pixels, clean/NLL, five stress cases, all three policies and
complete tensor/BN/source/config/checkpoint/update/time evidence. Retain the
old fixed-weighted baseline gate, the original KD plan's matched-alpha0 gate,
and the prospective current-incumbent requirements in
[incumbent-promotion-plan.md](incumbent-promotion-plan.md): fixed weighted mean
criterion **and no stress-case loss greater than0.01**. No mean improvement
may conceal noise regression. No changed policy can rescue a failure.

Before a repeat, collect actual paired correctness on the same ordered5000
original IDs against both matched control and incumbent;10000 bootstrap draws
keep the five views of each image together. Freeze primary weights/policy and
exact source/config/planned27 before fresh scratch27. Repeat the same benchmark
and gates. Conditional intervals are neither a new holdout nor seed variance;
synthetic stress is not proof of the hidden distribution. Only actual Kaggle
receipts establish a competition improvement, after separately implemented,
reviewed and tested KD lineage/delivery checks. This document authorizes none.
