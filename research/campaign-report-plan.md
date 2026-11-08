# Independent final-campaign JSON projection

Issue7 / draftPR8. New report-only adapter and tests. Original
`scripts/research_report_evidence.py` is pinned at
`888c8c509d5dc3034847bcdb9b8b8b213babe1e04e336345a94bdd920306d0ef`;
its ROWS/schema/functions and the original `build_report.py`, executed
notebooks, frozen training50/delivery sources and proof artifacts stay intact.
This adapter does not create a notebook/PDF, read tensors/images/prediction
rows, run a kernel, import Torch/NumPy/PIL, call an API or authorize delivery.

## Interface and explicit inputs

`load_campaign(manifest_path)` returns
`audited_campaign_report_projection_v1`. The standalone CLI prints that JSON
only with `--manifest`; no arguments show help. Root owns real input-manifest
generation and publication. All paths below remain private input metadata.

```json
{
  "schema": "campaign_report_input_manifest_v1",
  "recorded_at": "2026-11-03T01:00:00+00:00",
  "files": {
    "legacy_summary": {"path": "ABSOLUTE_OR_MANIFEST_RELATIVE_JSON", "sha256": "ACTUAL_RAW_SHA256"}
  },
  "scored_receipts": []
}
```

Every file record has only `path`, actual raw `sha256` and optional
`public_ref`. Relative paths resolve against the input manifest's directory.
A public ref must be a matching existing `docs/...` relative file, never an
absolute path or copied private filename. Paths/symlinks, duplicates and
nonfinite JSON are checked; no raw input object is dumped into the projection.

Required roles are `legacy_summary`, `legacy_results`, `baseline_benchmark`,
`incumbent_benchmark`; `ta_summary`, `ta_benchmark`, `ta_pair_summary`, `ta_pair`,
`ta_frozen`; and public-summary/full-raw-benchmark pairs named `se_*`,
`resizemix300_*`, `wrn16_*`, `augmix_*`. The WRN16 summary is the existing
reassessment document. Missing public metadata is verified from its explicitly
bound raw benchmark, never copied from a different model or invented.

Optional repeat roles are all-or-none: `ta_repeat_benchmark`,
`ta_repeat_result`, `ta_repeat_receipt`. Only actual complete formal seed27
evidence is accepted. Optional prediction roles are all-or-none and require
that repeat group: `ta_delivery`, `ta_prediction_receipt`,
`ta_prediction_inputs`, `ta_tensor_audit`. A partial/in-progress result cannot
overwrite pending status. Canonical complete scored receipts are explicit
records in `scored_receipts`, named by official ref under the existing canonical
receipt directories. Intents, pending references, predictions or benchmarks
cannot create a score. An actual scored TA receipt must bind the complete
repeat/delivery/freeze/checkpoint/CSV-hash chain. No CSV rows are read here.

## Audits and projection semantics

Reuse only the old adapter's pure parser/hash/numeric/gate/benchmark/legacy
pair/scored-receipt primitives. Do not route the new dual-pair schema through
an old validator under another name and do not monkey-patch old module globals.
The new module explicitly dispatches TA dual-pair/scored-receipt and each
negative-result schema. Full raw benchmarks must have40k disjoint-training
scope/5000dev, all18 native-v3 case/policy pairs, matching data/context/pixels,
original BN, complete counts, finite NLL, integer-grid accuracy, source/config/
checkpoint/history fingerprints and unchanged source bytes.

The actual origin baseline has a null `baseline_sha256`: it is audited directly
against its pinned raw/checkpoint identity, all18 cases, recorded flip summary,
context/pixels, source bytes and recomputed own-reference gates. It is not sent
through a candidate-against-baseline validator or assigned a synthetic self
reference. Candidate benchmarks must carry the actual origin fingerprint.
The SE constructor retains its explicit architecture/reduction fields and is
checked as the recorded SE-WRN22-4/r16 type; it is not reduced to an ordinary
four-field WRN constructor.

Keep original gate, incumbent mean criterion and per-case floor as separate
fields. SE's original dual gate passes but its prospective noise floor fails;
ResizeMix300 fails both incumbent mean and noise floor. WRN16's incumbent
comparison is retrospective and its status is not-prioritized, not a rewritten
preregistered failure. AugMix preserves its original clean-gate rejection and
50068success+12skip/three-view accounting. No missing WRN16 ledger is filled.

Join older TA completed and newer paired snapshots only on exact checkpoint,
benchmark and policy. The older `paired=false` cannot erase the actual later
pair/freeze. Preserve both CI objects: `clean_ci95/stress_ci95` and their point
deltas always refer to **original baseline flip**; `incumbent_*_ci95` and
incumbent deltas always refer to **WRN300 weighted**. Report intervals crossing
zero unchanged.10000 resamples cluster the five views by original image.

Output includes only allowlisted scalars, bounded configuration values, case
metrics, public source refs/hashes and actual canonical scored receipt fields.
No absolute input paths, source IDs, prediction rows, weights, full receipts,
arbitrary strings, private URLs or credentials enter it. Original primary,
original best, research best, overall best and unknown private score remain
separate. Reading JSON proof links does not constitute a new tensor/OS audit;
attribute those to their archived typed execution engines.

The separate new final wrapper consumes this new schema directly. It preserves
the original six notebook outputs, executes only two new research cells after
root permits CPU execution, and adds at most two research PDF pages to the
frozen primary report. This adapter does not modify old rendering text or
pretend the old fixed-ROWS CLI automatically supports new methods.
