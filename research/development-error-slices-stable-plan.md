# Stable-context development diagnostic

The first diagnostic stopped on TA seed 2026's `noise_004`: all 5,000 regenerated transformed inputs matched the historical pixel hash, but the correct count was 4,253 instead of 4,254. The failed receipt and original pairing evidence remain intact. This observation does not identify which operation caused the drift.

The independent diagnostic uses native BF16 and the same official development images, ordered cases, weights, batch size, inference policy and statistics as the original error-slices plan. In the worker only, set cuDNN benchmarking off and cuDNN deterministic convolution on. Record both flags and the global deterministic-algorithms flag. This is a new measurement context, separate from the frozen benchmark context.

Require exact historical pixel hashes. Report both old and new correct counts and the difference for every case and model. Prior correctness hashes are references, and the new runs must establish their own correctness identity. Only interpret class slices after two separate supervised executions match all three full correctness hashes, all case diagnostics and all aggregate statistics exactly. A mismatch remains unresolved evidence and does not justify a training change. Neither run replaces a benchmark, qualifies a candidate, alters an existing frozen policy or authorizes a submission.
