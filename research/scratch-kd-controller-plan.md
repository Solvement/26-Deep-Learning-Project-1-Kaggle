# Explicit local scratch-KD control26 and primary26 supervision

Issue7 / PR8. This controller only adds local resource/authentication evidence
for the separately frozen KD learning/typed benchmark. No runtime, accuracy,
qualification or score is claimed by source/tests. Old sources and receipts
remain unchanged. There is no repeat27, resume, prediction, upload, auto-formal
or implicit retry. Root alone launches an explicitly reviewed command.

## Commands and stage contract

No arguments show help without numerical imports. Unique ignored directories:

```text
python research/run_scratch_kd.py --run UNIQUE_CONTROL_PREF --mode preflight --role control26
python research/run_scratch_kd.py --run UNIQUE_KD_PREF --mode preflight --role primary26
python research/run_scratch_kd.py --run UNIQUE_CONTROL --mode formal --role control26 --control-preflight DIR --kd-preflight DIR
python research/run_scratch_kd.py --run UNIQUE_PRIMARY --mode formal --role primary26 --control-preflight DIR --kd-preflight DIR --control-run DIR
```

Role/config is explicit: control26 is alpha0 seed2026 and engine role
`development_control`; primary26 is alpha.5 seed2026 and engine role`primary`.
Both retain T4/WRN16-5/160/4,289,754 student parameters and exact original
40k/5k split, nativeBF16/directSGD/no clipping/no scaler/no skipped update.
The three original KD configs, helper, trainer, typed benchmark and lineage
are immutable dependencies; source/config/evidence changes reject admission.

Both independent role preflights must numerically complete, have their own
timing guard<=3h, actual identical native runtime, student initialization hash
and ordered source/ID hashes, and verified native cleanup before formal
control. All pref state is discarded. No accuracy or weights are saved.
Formal control's complete typed benchmark is required; it is not a submission
candidate and its false candidate-screen flag does not block KD by itself.
KD formal consumes the actual completed control plus its own already qualified
KD preflight. A same-source/environment cohort remains valid; there is no
unrequested repetition after control or reuse of alpha0 speed for KD.
Changed runtime/source/guard/budget rejects rather than silently remeasuring.

The base engineering input binding contains only the5 official training-cache
paths, trusted baseline, canonical teacher30-file closure and complete source
closure. Formal adds both complete native preflight bundles; primary further
adds the real control result, full inputs, selected/last/state/benchmark files
and native closed-boundary evidence. Added stage inputs are expected; do not
force an engineering manifest to contain future control outcomes.

## Own authenticated worker and closed receipt

Host imports stdlib/psutil and pure metadata modules only. Its own manifest,
script/argv/source SHA, nonce, original PID/creation times, bounded deadline,
canonical local lease, native Windows Job/POSIX session membership and actual
bootstrap are checked before runtime/Torch. Reuse only generic unchanged
accounting/guard/OS verifier primitives, never an old script-specific validator.

Completed evidence binds the unique native `host_result.json` to the supervisor
proof, spec/ready/completion PID/creation times, nonce/source/argv/deadline/exit
and cleanup. Old owner/child identities must be inactive; PID reuse is harmless
and never killed. Any unknown cleanup retains the matching lease. No orphan
allocation is inferred absent or deleted by this controller.

## Actual preflight and formal execution

Use exactly shared `engine.setup`, `start_epoch` and `train_step`, also used by
formal. Snapshot actual initial student state to an aggregate SHA only.
Original40k uint8 staging/H2D and teacher snapshot/loading are charged by the
supervisor; the already resident pool is not restaged every step. Timed step
includes gather/augmentation/Mixup, all rollback clones/RNG capture, student
44gradient/210state/13BN/SGD/EMA/metrics, teacher191state/31BN/no-grad/eval/
immutable/full-parent-RNG proofs and positive-alpha teacher forward. Actual
student logits must be native BF16 and model/EMA masters FP32. Three warmup
plus20 timed successful B128 steps consume2944 unique original source rows.
Check real parameter/EMA changes and one student BN advance per step. Scalars
and opaque hashes only are persisted, never tensors, logits, labels, IDs or
weights. Formal retains its actual student artifacts privately for audit.

Pure canonical lineage fixes teacher primary26 WRN34-3/300: checkpoint53d8...,
freeze b631..., matched native benchmark e3c586..., same official40k. Preserve
its original T4/FP16 93864successful+36overflow skips and null training commit.
Teacher forward here is BF16; it never becomes student/EMA state or supplies
repeat weights. Only qualified future frozen primary student may predict
through a separately implemented typed chain.

Formal calls unchanged engine training for160 epochs/50080student updates and
full typed5000×18 benchmark. It does not require a control candidate gate. KD
records original baseline, incumbent mean+eachcase floor and matched-control
mean gates as registered, without adding a control floor. A failing primary
is still a real complete negative; it creates no repeat/delivery qualification.

## Fixed resource policy

One canonical local job; exactly180s per engineering attempt. Every formal
reservation is min(10800s, actual remaining21600s UTC-day allowance, seconds
to2026-11-05 05:00UTC). Admit only if its own measured guard
`1.5*median*313*160+300` fits all bounds. Safety factor and minimum300s overhead
stay fixed; setup/selection/checkpoint/benchmark remain in job wall time.
Each budget uses one captured current UTC day and timestamp as its start.
Actual receipts settle usage; no planned speed, old160/200/300 timing or paper
FLOPs is admission. No paid compute, split-resume or limit relaxation.

Only explicit root calls move phases. Fresh27/delivery remain unsupported by
this controller; actual primary qualification is required before implementing
them. Public score/private ranking and engineering proxies stay distinct.
