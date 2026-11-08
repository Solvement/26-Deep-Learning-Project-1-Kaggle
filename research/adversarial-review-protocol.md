# Adversarial review for new experiments

Before implementation or a GPU reservation, ask an independent agent to challenge each proposed architecture or method. Give the reviewer the strongest current reference, the proposed change, the comparison budget, and the competition constraints. Ask the reviewer to challenge the reference design as well as the candidate. A higher current leaderboard score makes a model a useful reference and does not establish that its architecture is optimal.

The review must identify the strongest alternative explanation, a failure mode the current evaluation could miss, and one observation that would falsify the proposal. Check architecture and parameter count, data and split provenance, preprocessing and inference policy, train and evaluation precision, update counts and checkpoint selection, and GPU and delivery feasibility. Record the review and disposition before launch. Resolve any issue that invalidates the comparison. If the proposed test cannot distinguish the relevant explanations within budget, defer it.

After a run, report both the matched-reference delta and the current-best delta, with the sign, percentage points, and scoring denominator. Show the clean and per-transformation outcomes alongside the actual public score when available. Separate direct observations from hypotheses and label any unmatched factors. Public aggregate scores establish net counts, while per-image differences require paired evidence. Preserve negative and inconclusive results.

Use the same review to decide the next single change. Keep the existing frozen qualification, resource, delivery, and submission rules. This protocol applies prospectively and leaves historical artifacts and decisions intact.
