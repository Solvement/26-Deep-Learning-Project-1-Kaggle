# Prospective decisions for fresh post-audit candidates

Issue7 / draftPR8, registered2026-11-02 before any fresh post-audit primary
training. Preserve every historical gate, failure, freeze and official score.
This adds a prospective research decision, not an uploader exception.

The fixed incumbent is primary2026 WRN34-3/300 checkpoint
`53d8d64df899ebab79ae77144ff36e912c45e65178c0b50bd9af381dc48187bc`,
public0.878/ref57854294, matched-local benchmark
`e3c586afc302d1422c80eb3831562f6b67f684d53c7b2b5390c9d4623001570b`.
Use native-v3 localBF16/Torch2.12.0+cu130, batch128, original5000dev and
`center80_shift20` on both models, unchanged BN. Incumbent clean0.9654,
NLL0.11753104768097401, mean synthetic stress0.79716.

For a fresh40k primary to warrant a seed repeat and possible exploratory
submission, require all the following before freezing it:

1. Complete original typed training/source/data/checkpoint/tensor audit and
   all18 benchmark cases. The unchanged original baseline's **fixed weighted**
   promotion flag must pass; an any-policy flag is insufficient.
2. Against the incumbent with the same weighted policy, either clean gain
   >=0.001 and mean-stress change>=-0.002, or mean-stress gain>=0.005 and clean
   change>=-0.002. These are predeclared engineering differences, not tests of
   statistical significance. NLL is recorded, not used to choose another policy.
3. No individual synthetic-stress accuracy may fall by more than0.01 against
   that incumbent. Report all five signed changes and noise separately; a large
   blur gain cannot conceal substantial losses elsewhere. This is a future
   conservative operating choice, not a claim about hidden-test composition.
4. Produce actual incumbent paired correctness on the identical ordered5000
   source IDs, clean plus five views, with10000 original-image cluster
   bootstrap samples. Keep views of each image together. Report the full
   conditional interval and discordance even when it crosses zero; it neither
   creates independent confirmation nor estimates training-seed variance.
5. Then freeze exact primary checkpoint/policy/config/source and planned27
   configuration before fresh scratch27. The repeat must pass the same old
   fixed-policy gate and the same scalar incumbent criteria in2-3. Record both
   outcomes; do not replace the frozen primary with repeat weights. Retain
   legacy paired-v2 evidence required by delivery_v2 as a separate comparison.

Only the already-frozen qualified primary may supply hidden predictions.
All existing repeat/freeze/timing/source/quota/official-ID/hash/duplicate and
unknown-upload checks remain mandatory. New-method lineage must be explicitly
supported and tested by an independent delivery path; neither this document
nor aggregate/paired diagnostics makes an unknown trainer acceptable.

This protocol does not reevaluate or promote WRN16, rejected AugMix, SWA or
any old candidate under a changed rule. A new BF16300 control can be trained
for a controlled subsequent method comparison without being a submission
candidate; failure to beat the incumbent stops its delivery work.

Already-qualified historical full50k refits keep their frozen parent protocol,
fixed final weights and no independent development claim. Do not run this dev
decision on a full50k checkpoint. Do not reuse the once-only confirmation or
use hidden images for local selection. Budgets and the Nov5 05:00UTC hard stop
remain unchanged.
