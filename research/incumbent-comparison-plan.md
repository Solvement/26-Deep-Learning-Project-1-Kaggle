# Prospective incumbent comparison, Issue 7 / PR 8

This independent entry adds an aggregate comparison against the best observed
official primary, WRN34-3/300, public 0.878 / reference 57854294. It does not
modify any old experiment, gate, benchmark, paired evidence or delivery helper.
The five-day campaign's deadline, resource limits and submission checks apply.

## Fixed reference and scope

- Reference: `results/research/wrn34_3_300_s2026_local_v3/benchmark.json`, raw
  SHA-256 `e3c586afc302d1422c80eb3831562f6b67f684d53c7b2b5390c9d4623001570b`.
- Frozen primary checkpoint SHA-256:
  `53d8d64df899ebab79ae77144ff36e912c45e65178c0b50bd9af381dc48187bc`.
  It is epoch298 online, trained scratch seed2026 on40k. Its matched local
  weighted result is clean96.54%, NLL0.11753104768097401, stress79.716%.
- Compare `center80_shift20` with that same policy, never incumbent `flip`.
  Require the complete native-v3 local BF16 / Torch2.12.0+cu130 context,
  fixed128 batch, identical official source/split and every case pixel hash.
  T4 FP16 benchmarks cannot be mixed with these local BF16 measurements.
- Both documents must contain all three policies and all six cases, each5000
  development examples,40k training, original split seed2026, primary seed2026,
  unchanged BN, parameter count<=5M, source and checkpoint/history/metrics hashes.
- Record clean accuracy/NLL, mean stress, every case's accuracy/NLL and signed
  deltas. Explicitly list noise and other stress regressions, rather than
  concealing them inside the mean. Effective parsed-config hashes are named as
  such; they are not presented as raw config-file checksums.

## Execution and interpretation

`incumbent_compare.py` uses only Python's standard library. It reads complete
aggregate benchmark JSON and checks their recorded consistency and frozen
source bytes. It does not load official arrays, labels, models, Torch, account
APIs or a Notebook kernel; it consumes no GPU allocation. Output is fresh-only,
with input/self/plan hashes and a final input/source stability check.

Example (a complete future candidate, output kept private until reviewed):

```text
python research/incumbent_compare.py --candidate-benchmark results/research/NEW_PRIMARY/benchmark.json --output results/research/NEW_PRIMARY/incumbent_comparison.json
```

The output is explicitly a recorded-scalar summary. It does not independently
audit checkpoint tensors, rerun inference, generate a confidence interval,
estimate seed variance, or establish a new confirmation or hidden-test score.
It emits no `passed` or promotion field. All prospective new methods should
include this same-policy incumbent comparison, alongside the unchanged old
baseline gate and full method-specific evidence; this summary is not itself
an authorization for repeat, prediction or submission.

Existing `paired_compare_v2.run` cannot be repointed: it requires the original
baseline checkpoint SHA and baseline.flip. Its pure original-image clustered
bootstrap and exact correctness collection may be reused by a future separately
reviewed GPU entry, but are not executed or imported here. An aggregate
comparison cannot recover paired uncertainty from marginal accuracies.

Old candidates can be diagnosed with this summary without changing their old
decisions. In particular, it does not rescue AugMix's failed clean gate or
automatically advance WRN16 merely because its old-baseline gate passed.
No thresholds are adjusted after seeing those outcomes. Any future promotion
protocol or paired GPU extension must be fixed before the new experiment.
