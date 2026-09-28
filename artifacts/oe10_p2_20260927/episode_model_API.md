# Shared episode module contract

```python
model = EpisodeModel(encoder, arm="B2", device="cuda:0",
                     qwen_path=local_pinned_qwen, gradient_checkpointing=True)
output = model(loader_result["model_input"], target=training_target,
               with_language=True)
output["loss"].backward()
```

Inputs are the existing loader's exact `model_input` dict: fixed generic prompt, query observations, and ordered support pairs. Query is original 128×128 at two acquired dates; each support contains all eight dates and its selected boolean object mask. K must be 1, 2, 4 or 8. The strict baseline rejects sentinel-missing pixels; it does not detect clouds. The provenance/acquisition loader remains responsible for raw/normalized consistency, exact query dates, input hashes and source-role enforcement.

Returns `logits` with shape `(128,128)`, `mask_loss`, scalar `language_loss`, and `loss = mask_loss + .05*language_loss`. With no target, `mask_loss` and `loss` are `None`. With `with_language=False`, Qwen is not called and language loss is scalar zero. A target-free language forward returns detached last-position logits in `vlm_last_logits`; this is one next-token distribution, not a generated explanation. Only a supplied supervised target produces teacher-forced coverage text and CE. Query gold does not enter the prompt, connector or segmentation head.

`head`, `connector`, `encoder`, and `qwen` are exposed. `trainable_parameters()` excludes Qwen and the B0 encoder. `trainable_state_dict()` contains head/connector and B2 encoder, excluding frozen Qwen. The root trainer owns optimizers, native replay, model identities, checkpoint storage, cold-process restoration and evaluation targets. It may save native full state instead of the module's duplicate encoder state.

The dense head is P1's LayerNorm→3D+2 features→256-unit MLP→bilinear mask. Every support produces an object prototype; the head uses the arithmetic mean prototype for each role. The 64-slot connector is P1's learned-query cross-attention with three role embeddings. It receives the query grid and all K positive plus all K counterexample prototypes. Tensor order is retained; there is no added positional encoding, and attention is invariant to permutation within a role. This is an extension of the same baseline to K supports, not a novel architecture.

B2 checkpoints each EO forward with `use_reentrant=False`, preserving encoder gradients even though raw input tensors do not require gradients. B0 encodes under no-grad. Encoder and frozen FP32 Qwen stay in eval mode, preserving P1's deterministic dropout behavior. A frozen Qwen forward itself must not be wrapped in no-grad during language training: connector slots are differentiable inputs. All real EO forward calls, including checkpoint recomputation and separate gradient diagnostics, increment the cumulative cost ledger.

`model.costs` records `EOencoder_calls`, `patch_date_encodes`, `vlm_forward_calls`, cache hits/misses, content-hash bytes, and checkpoint recomputation calls. Read these after backward to include recomputation. `output['metrics']['costs_cumulative']` is only the forward-time snapshot. Distinguish physical EO calls from the loader's declared unique observations and logical support exposures.

B0 defaults to an exact-content frozen-feature CPU cache. B2 training has no feature cache. For repeated evaluation:

```python
model.eval()
model.set_cache_mode("evaluation")
with torch.no_grad():
    output = model(public_input, target=None, with_language=False)
model.clear_cache()
model.set_cache_mode("frozen" if arm == "B0" else None)
model.train()
```

Cache keys include normalized acquired arrays, timestamps, validity, imputation flags, dtype and shape. The encoder's parameter/buffer versions invalidate the cache after normal optimizer or state-load mutations. The cache does not handle unsupported external `.data` mutations that bypass PyTorch version tracking; call `clear_cache()` after any external state manipulation. CPU cache capacity is 128 grids by default. Cache entries are detached and permitted only for a frozen encoder or no-grad evaluation. They never give a selector access to unacquired features.

Do not change a shared encoder's weights or stochastic mode between a checkpointed forward and its backward. The root worker currently completes native replay backward before the downstream episode, which satisfies this ordering. The module is single-episode and serial: concurrent forwards and generation loops sharing its embedding hook are unsupported. Call `close()` when done to remove the hook.

Local tests use small CPU encoder/language substitutes. Five checks passed: all K8 supports, B0 freeze/cache invalidation, checkpointed B2 gradient parity, CE gradient through frozen language layers into EO, and evaluation-cache boundaries. Actual checkpoint token shapes, Qwen forward behavior, K8 peak GPU memory and cold-process resume still require the root server run. No pretrained weights or server state were changed by these local tests.
