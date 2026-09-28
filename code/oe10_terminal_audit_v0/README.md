# P2 terminal integrity audit v0

This helper reads the already terminated P2 experiment and frozen collector
output. It starts no training, inference or GPU work, sends no signal, and changes
no P2 source/input/result. It creates one new output directory only after all
mandatory integrity checks pass. Running/missing terminal status is rejected.

Input identities are pinned in identity_pins.json from the existing v3 model,
v2 collector, protocol and fairness evidence. The caller separately supplies the
actual terminal collector JSON SHA and this helper's frozen source-manifest SHA.
The collector must be run by the parent before invoking this helper; no fictional
collector freshness, prediction-decoding or independent-review fields are added.

On the server:

    env -u PYTHONPATH python -B /home/work/data/olmoearth/oe10_p2_v0/audit_snapshot/oe10_terminal_audit_v0/audit_terminal.py \
      --protocol PATH_TO_FROZEN_P2_EXECUTION_PROTOCOL_UNDER_P2_ROOT \
      --collection /home/work/data/olmoearth/oe10_p2_v0/terminal_collection_20260928_0843.json \
      --collection-sha256 c180fb97d24ed0d305c8b064bfbf4c10b0f811c3a2d3e84dab9d98565335629c \
      --helper-manifest-sha256 EXTERNALLY_FROZEN_HELPER_MANIFEST_SHA \
      --out /home/work/data/olmoearth/oe10_p2_v0/terminal_audit_20260928_v0

Checks include actual finished controller status, unchanged protected SHA/mtime
and source snapshot, inactive owned PIDs/commands, frozen protocol and collector
source, exact controller/collector completed-run set, five or six verified runs,
actual final receipt/checkpoint/log/order/final score/prediction-manifest hashes,
and the seven source-level fairness proofs. The partial collector does not reach
its full comparative fairness computation; this helper does not pretend it did.

Five verified runs with a terminal failed controller are reported incomplete.
No incomplete run enters the final 2304-step comparison. Its receipt update,
last logged update, available score steps and checkpoint file hashes are recorded
separately. A latest-checkpoint/receipt mismatch is retained as an interruption
mismatch, unverified; it cannot authorize resumption or score promotion. The helper
does not deserialize checkpoints. Prediction bytes were hashed by the frozen
collector; numerical array review and independent export review are separate.

Outputs:

- terminal_evidence.json: actual file/protected/process evidence, completed and
  unfinished run boundaries, original stability failures and reporting limits.
- terminal_attestation.json: schema oe10_p2_terminal_audit_attestation_v1,
  compatible with the frozen launcher; includes evidence-file SHA and an explicit
  requires_independent_review_before_use flag. audit_complete means only that
  this terminal integrity audit passed, not that P2 scientific gates passed.

The parent must independently review the output and externally pin its SHA before
using the attestation in any later launcher. This helper creates no reservation
and grants no automatic permission to start a GPU job.

CPU synthetic tests:

    python -B -m unittest discover -s code/oe10_terminal_audit_v0 -p test_audit_terminal.py -v
