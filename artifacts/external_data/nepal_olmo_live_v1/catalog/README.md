# Catalog snapshots

- `20260827T062409Z`: first official OData capture. It exposed 30 Sentinel-1
  products because each of 15 physical acquisitions had SAFE and COG
  representations. This snapshot is preserved but superseded for scene counts.
- `20260827T062506Z`: canonical acquisition-level snapshot. SAFE/COG replicas are
  retained as `alternate_representations` but counted once. Use this snapshot.

`LATEST` points to the current snapshot. Every snapshot is immutable and carries
its own `SHA256SUMS`.
