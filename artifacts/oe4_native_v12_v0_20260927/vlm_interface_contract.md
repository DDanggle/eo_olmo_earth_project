# OE4 Qwen3-VL / EO interface smoke contract

Status: source-reviewed and locally syntax-checked on 2026-09-27. **Actual Qwen/GPU execution has not been performed by the author of this script.** Local Python has no torch/transformers, so `py_compile` and `--help` are the completed checks. Root will run the real cached checkpoint.

## Connection

The existing Qwen3-VL RGB processor, native vision encoder, image placeholder positions, and DeepStack path remain in use. Insert 16 ordinary text-modality positions after the image and before the final user-message terminator. A hook on the input embedding replaces just those positions with projected EO features. It does not redefine Qwen image tokens, resize the vocabulary, call the language-only submodel directly, or supply an alternate DeepStack implementation.

The original native RGB prompt is the parity reference. A prompt with additional EO slots is a different sequence and is **not** expected to match original RGB logits, even if its EO values are zero. This distinction prevents a false parity claim.

The EO tokens keep their encoder information but receive Qwen ordinary text RoPE positions. They do not automatically acquire per-pixel spatial coordinates or sensor/band RoPE. Flattened-token adaptive average pooling is a bounded interface device, not a proposed spatial or conservation method. OlmoEarth v1.2's S2 output is not a set of per-band tokens.

## Exact pass checks

1. Original native RGB logits before installing the hook, with hook inactive, and after the EO-only optimizer step must be bitwise equal.
2. Native RGB projected image features and each DeepStack feature tensor must have identical byte hashes between original RGB and EO-slot prefill.
3. Qwen's actual `generate` must inject EO features once at prefill and bypass injection for its one-token cached continuation. The cache length is checked. Its second raw logits are compared to a full recomputation with the identical first generated token. Both allclose (declared BF16 tolerances) and top-1 agreement are required, with exact deltas reported.
4. One teacher-forced CE backward must yield finite, nonzero gradients in EO features and adapter; an optimizer step must change sampled adapter tensors. A supplied live encoder must also have finite nonzero parameter gradients and actual sampled parameter changes.
5. Frozen VLM parameters must have no gradients and no optimizer membership. Original RGB parity after the EO optimizer step is checked again.

The default sentence target is explicitly engineering input, not an annotated answer. The script does not assess semantic accuracy, training gains, EO benefit over RGB, or novelty. A precomputed NPY mode finishes with `ENCODER_NOT_TESTED`; it cannot demonstrate that any OlmoEarth weight was updated.

## Example invocation after root provides local assets

```sh
python vlm_interface_smoke.py \
  --model-dir /home/work/data/olmoearth/models/Qwen3-VL-8B-Instruct \
  --image /path/to/audited_rgb.png \
  --eo-provider /path/to/provider.py:make_provider \
  --eo-provider-config /path/to/provider_config.json \
  --out-dir /path/to/new_receipt_directory
```

Provider interface:

```python
def make_provider(config, device, dtype):
    # Load real local OlmoEarth, audited native inputs, and their normalization.
    # Keep EO parameters FP32 if desired; dtype denotes the VLM dtype only.
    return {"model": encoder,
            "encode": lambda: encode_native_sample_with_autograd(),  # [1,N,D]
            "metadata": {"checkpoint": "...", "input_sha256": "...",
                         "native_token_layout": "...", "bands": ["..."]}}
```

`model` must include the real encoder parameters used by `encode`; `encode` must not detach or enter no-grad. The smoke puts the encoder in eval mode for deterministic checks but keeps its trainable flags. All provider paths must be local. The loader owns the image/EO geographic and temporal correspondence; the interface script cannot infer that from separate PNG/NPY assets. Audit that correspondence before interpreting later scientific results.

## Version-specific source basis

- [Transformers v5.15 Qwen3-VL model source](https://raw.githubusercontent.com/huggingface/transformers/v5.15.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py): `get_placeholder_mask` uses input IDs; `forward` keeps masked-scatter image features and passes DeepStack features to the language model; `compute_3d_position_ids` requires `mm_token_type_ids` when image grids are supplied; native generation computes and carries M-RoPE positions; `logits_to_keep` allows bounded target logits. The script records the actual installed source file hash.
- [Transformers v5.15 processor source](https://raw.githubusercontent.com/huggingface/transformers/v5.15.0/src/transformers/models/qwen3_vl/processing_qwen3_vl.py): processor defaults return multimodal token types. Every inserted EO slot and target text token is explicitly type 0.

Multi-image batches, beam expansion, mixed sequence lengths, distributed training, checkpointing, and a reusable trained-model save/load/generation wrapper are not covered. Those need separate contracts before scale-up. The present file deliberately supports only a batch of one and one RGB image.
