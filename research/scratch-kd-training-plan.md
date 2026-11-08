# Fixed scratch KD: independent learning and typed benchmark

Issue7 / PR8, prospective2026-11-03. Supplements the unchanged
[original proposal](scratch-distillation-plan.md) and
[follow-up](scratch-distillation-followup.md), SHA256
`b6c916bd778e1042cfcd6a0f0bacb85ec66fc014ef282ab6204be4a4727b3d8f`.
No KD timing, formal training, accuracy or delivery result is claimed here.
WRN16/200 and300 negatives stay closed. All old source and confirmation remain
frozen; no external examples, pretrained weights or hidden labels are allowed.

## Fixed experiment

Three raw configs use the original WRN16-5/160 parent plus `kd_temperature:4.0`
and `kd_alpha`. Control26 alpha0, KDprimary26 alpha.5, KDrepeat27 alpha.5.
The only differences are alpha and the predeclared repeat seed. Keep the exact
40k/5k/splitseed2026, original crop/flip/normalization/Cutout16/Mixup.2 p.5,
SGD/Nesterov LR.1/min.0001/warmup5/momentum.9/WD.0005, EMA.995 and original
online/EMA dev selection. Student has4,289,754 total/trainable parameters.
There is no gradient clipping, scaler, skipped-update branch, feature loss,
teacher TTA, extra augmented view, teacher ensemble or temperature search.

Teacher is only frozen primary26 WRN34-3/300, selected epoch298 online model,
4,166,298 parameters, same official40k. CheckpointSHA
`53d8d64df899ebab79ae77144ff36e912c45e65178c0b50bd9af381dc48187bc`,
freezeSHA `b6314d7f8b05f3f1978c682115d9cb4185646a32771db2826f95b675161045a3`,
native-v3 benchmarkSHA
`e3c586afc302d1422c80eb3831562f6b67f684d53c7b2b5390c9d4623001570b`.
Preserve actual teacher T4/FP16 training93900attempt/93864success/36overflow
skips and null recorded training Git commit. Its forward here uses native
BF16; do not relabel that old training as BF16/zero-skip or assign it the local
remeasurement commit. Real core/method/config/data/freeze hashes carry lineage.
No27/stability checkpoint, target, cache or descendant can be a teacher.

[Hinton et al., section2](https://arxiv.org/html/1503.02531v1) motivates soft
targets, hard-label loss and T-squared. [Born Again Networks](https://proceedings.mlr.press/v80/furlanello18a.html)
supports testing fresh same-capacity students; neither predicts this score or
establishes T4/alpha.5 as optimal. Only one student residual CNN is retained
at inference. Teacher/student/EMA resident training state exceeds5M in total
and its extra compute/memory is disclosed and measured.

## Shared interfaces and loss

Both new modules import stdlib only at top level. The authenticated controller
owns script/argv/nonce/lease/OSJob/budget checks before calling their numerical
functions. Numerical dependencies are lazy and import `runtime` before Torch.
No direct training CLI, resume, hidden loader, prediction or upload is added.

`setup(config,data_dir,device,teacher_reference)` is the sole formal/preflight
setup, returning student/EMA/SGD, resident40k arrays, dev arrays, indices,
teacher/snapshot, dtype/config/reference and source/signature. It seeds the
parent, loads and deeply audits the exact teacher inside full parent-RNG
isolation, then initializes the scratch student. Teacher is a separate object,
never a registered student child. Its191states/31BN are cloned once at setup;
it stays eval, requires_gradFalse and no_grad, with no optimizer/EMA updates.
`start_epoch(bundle,epoch)` sets student.train and produces the original GPU
randperm. No preflight state is saved or reused for formal initialization.

`train_step(model,ema,optimizer,train_x,train_y,order,batch,config,updates,
epoch=...,teacher=...,teacher_snapshot=...,dtype=...)` is the only step used by
formal and actual preflight. It performs original schedule/gather/augmentation/
Mixup once. Teacher receives the exact final mixed student tensor and labels
remain lambda-weighted original donor labels. One detached plain teacher
forward supplies targets; a clean-logit cache would change the experiment.

For positive alpha, CE, log-softmax and forward KL are FP32 outside autocast:
`loss=(1-alpha)*hard + alpha*T*T*kl_div(log_ps,log_pt,batchmean,log_target=True)`.
Student gradients remain live; teacher logits detach. The helper is
`scratch_kd.py`, independently reviewed. Alpha0 short-circuits all teacher
forward/KL work and returns the original hard-loss object with its original
autocast placement and out-of-place mixed CE addition; no extra0/1 scaling.
Production code does not dynamically rewrite or patch any old learning math.
CPUCI independently executes the original frozen batch AST to check exact
loss/logits/SGD/EMA/BN/Python/NumPy/TorchCPU/allCUDA RNG equivalence.

Each step snapshots complete student/EMA/SGD/prior gradients/parent RNG before
work. Nonfinite loss/gradient/state or a teacher proof failure restores the
entire pre-step state and raises; it never increments updates, skips a batch,
changes the EMA rule or leaves mutated BN/momentum/RNG. No failure clamp or
silent retry. Teacher forward uses no_grad rather than inference_mode, and its
full immutable/finite/no-grad/eval/RNG proofs are inside the same timed call.
Normal finite math remains direct backward→SGD→updates++→EMA; original epoch
end copies model BN buffers to EMA. Returned logits/permutation are in-memory
only. Controllers write allowlisted scalars, never tensors, IDs or predictions.

## Complete state and benchmark

All runs have160×313=50080 student updates, zero skips and6.4M source/student
inputs. KD has50080 teacher forwards/6.4M teacher inputs; alpha0 has0/0.
These use one shared augmented view per source, not two independently sampled
views. Count teacher inputs separately rather than hiding teacher cost in a
single-model training claim. Selected/last checkpoints embed student state
and exact teacher-reference hash, never teacher weights. Last includes actual
student/EMA/SGD/RNG/history; resume is explicitly unsupported.

`benchmark_scratch_kd.validate_training(...)` actually audits selected and last
tensors,44momentum buffers, all BN counters, full160 history/raw/effective
config/source/teacher lineage and parent RNG. It is a new KD schema, never an
old shadow/core marker. `run_benchmark(...)` then reuses frozen core scoring
for all18 complete native-v3 policy/case pairs,5000each, original BN, identical
pixel/context/split/source hashes. A KD benchmark additionally loads/audits
its actual new alpha0 control and its typed benchmark. The historical160
run cannot replace that control. Check original weighted baseline gate,
incumbent same-policy mean+eachstress floor and matched-control analogous mean
condition. Report all control per-case changes; do not invent a new mandatory
control floor. Full uncertainty/pairing and a separate fresh student27 follow
only a passing candidate, conditional on the same fixed teacher.

## Resource and authorization boundary

First run both fixed control and KD actual23step preflights before spending on
formal control. Require official40k residency/native Laptop GPU/BF16,3warmup+
20timed updates, all student rollback clones/finite/norm/BN/SGD/EMA/metrics and
all extra teacher forward/191state/31BN/no-RNG/immutable proofs in whole-step
timing. Setup teacher clones/loading/staging are charged by supervisor. Native
peak VRAM is measured. No source changes during either proof-bound run.

Exactly180s engineering cap; fixed formal guard is
`1.5*median_full_step_seconds*313*160+300`, within job10800/day21600 and
hard stop2026-11-05 05:00UTC. Minimum300s overhead cannot be reduced; increase
it prospectively if actual setup/dev/checkpoint/fullbenchmark cost requires.
The3h limit permits median<=.1397763578s. Old student timings and FLOPs cannot
admit KD. Each control/primary/repeat gets separate native admission and receipt;
their maximum9h aggregate plus all diagnostics spans UTC days. Both arms must
fit before any formal control starts. If the full required chain cannot finish
within the actual remaining calendar/day budget, skip this experiment/delivery.

Source review/CPU tests do not authorize GPU or upload. Root owns the new
guarded controller and pure teacher-lineage manifest. Old pair/freeze/repeat/
delivery/uploader paths reject the new KD marker. Only if the actual primary
passes may separate explicit typed adapters be implemented/tested for genuine
pairing, frozen primary, exact fresh27 and primary-only2000-ID CSV delivery.
No old allowlist, schema, source, receipt or teacher/repeat weight is relabeled.
Existing quota/hash/reference/intent/duplicate/unknown and private-score rules
remain mandatory; no score improvement or submission qualification is claimed.
