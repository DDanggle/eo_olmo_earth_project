#!/usr/bin/env python3
"""One-step local Qwen3-VL + separate EO-slot interface contract.

No downloads, remote calls, or semantic-performance claims. Batch size 1, one
native RGB image, greedy two-token generation, and no beam search. The original
Qwen RGB/DeepStack path is retained. EO tokens occupy ordinary text positions;
they are NOT native image tokens or band-specific Qwen visual tokens.

Provider contract (optional, instead of --eo-token-file):
  --eo-provider /absolute/provider.py:make_provider --eo-provider-config cfg.json
  make_provider(config: dict, device: torch.device, dtype: torch.dtype) ->
    {"model": trainable torch.nn.Module,
     "encode": zero-argument callable returning Tensor[1,N,D],
     "metadata": JSON-compatible dict}
The provider must load only local assets, retain autograd, and return real native
EO features. This script checks its model's gradients and a sampled actual weight
update. A precomputed NPY instead proves ONLY adapter and feature-input gradient.

Implementation reference, verified 2026-09-27:
https://raw.githubusercontent.com/huggingface/transformers/v5.15.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
The 5.15 processor's mm_token_type_ids are retained; EO slots have type 0.
Embedding substitution preserves input_ids so image placeholder detection,
DeepStack masks, and generation's native M-RoPE/cache logic remain active.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def check_fresh_visual_capture(captures, start_index, expected_mask_shape,
                               expected_positions, expected_depth, hidden_size, phase):
    """Pure contract check: stale/missing, duplicated, or wrong-position capture fails."""
    fresh = captures[start_index:]
    if len(fresh) != 1:
        raise RuntimeError(f"{phase}: expected exactly one new visual prefill capture, got {len(fresh)}")
    capture = fresh[0]
    if capture["mask_shape"] != expected_mask_shape or capture["visual_positions"] != expected_positions:
        raise RuntimeError(f"{phase}: native visual mask does not match the current image placeholder positions")
    count = len(expected_positions)
    if count < 1 or capture["visual_token_count"] != count:
        raise RuntimeError(f"{phase}: unexpected native visual token count")
    if expected_depth < 1 or len(capture["deepstack"]) != expected_depth:
        raise RuntimeError(f"{phase}: native DeepStack depth mismatch")
    expected_shape = [count, hidden_size]
    if capture["image_embeddings"]["shape"] != expected_shape:
        raise RuntimeError(f"{phase}: native projected image feature shape mismatch")
    if any(feature["shape"] != expected_shape for feature in capture["deepstack"]):
        raise RuntimeError(f"{phase}: native DeepStack feature shape mismatch")
    return capture


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=Path("/home/work/data/olmoearth/models/Qwen3-VL-8B-Instruct"))
    p.add_argument("--image", type=Path, required=True, help="Local native RGB PNG/JPEG; no automatic EO-to-RGB rendering")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--eo-token-file", type=Path, help="NPY [N,D] or [1,N,D]; no encoder-gradient claim")
    source.add_argument("--eo-provider", help="Local module:factory or /absolute/file.py:factory; trusted local code")
    p.add_argument("--eo-provider-config", type=Path)
    p.add_argument("--deps-root", type=Path)
    p.add_argument("--out-dir", type=Path, required=True, help="Must not exist")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--eo-slots", type=int, default=16)
    p.add_argument("--max-image-pixels", type=int, default=256 * 32 * 32)
    p.add_argument("--max-prompt-tokens", type=int, default=2048)
    p.add_argument("--question", default="Describe the visible land cover in one short sentence.")
    p.add_argument("--target", default="The image shows land cover.", help="Engineering CE target only; not an expert label")
    p.add_argument("--adapter-lr", type=float, default=1e-4)
    p.add_argument("--encoder-lr", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=270927)
    p.add_argument("--cache-atol", type=float, default=0.15)
    p.add_argument("--cache-rtol", type=float, default=0.01)
    a = p.parse_args()
    if not (1 <= a.eo_slots <= 128 and 64 <= a.max_prompt_tokens <= 4096):
        p.error("Bounded smoke requires 1..128 EO slots and 64..4096 prompt tokens")
    if not (32 * 32 <= a.max_image_pixels <= 512 * 32 * 32):
        p.error("max-image-pixels must be 1024..524288")
    if a.eo_provider_config and not a.eo_provider:
        p.error("provider config requires a provider")
    return a


def main():
    a = parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=False)
    receipt_path = a.out_dir / "receipt.json"
    start = time.monotonic()
    receipt = {
        "status": "initializing", "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "One-step EO/VLM engineering interface only; no research or semantic accuracy result",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        "script_sha256": sha256(__file__),
        "source_reference": "https://raw.githubusercontent.com/huggingface/transformers/v5.15.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py",
        "limitations": ["Single image and batch 1 only", "No beam or multi-image contract", "EO slots receive text RoPE positions; no new spatial RoPE claim", "A supplied generic CE target is not evaluation gold"],
    }
    write_json(receipt_path, receipt)
    handles = []
    try:
        if a.deps_root:
            sys.path.insert(0, str(a.deps_root.resolve()))
        import numpy as np
        import torch
        import torch.nn.functional as F
        import transformers
        from PIL import Image
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        torch.manual_seed(a.seed)
        random.seed(a.seed)
        np.random.seed(a.seed)
        device = torch.device(a.device)
        dtype = getattr(torch, a.dtype)
        if dtype == torch.float32:
            # Separate FP32 diagnostic; preserve the failed BF16 result and gate.
            torch.set_float32_matmul_precision("highest")
            torch.backends.cudnn.allow_tf32 = False
        receipt["precision_contract"] = {
            "requested_dtype": a.dtype,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "interpretation": "A separate precision run; an FP32 pass does not establish BF16 cache parity",
        }
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            torch.cuda.init()
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        if not a.model_dir.is_dir() or not a.image.is_file():
            raise ValueError("The local model directory and local RGB image must exist")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(a.model_dir), local_files_only=True, dtype=dtype, attn_implementation="sdpa"
        ).to(device)
        model.requires_grad_(False)
        model.eval()
        processor = AutoProcessor.from_pretrained(str(a.model_dir), local_files_only=True)
        receipt["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__}
        source_file = Path(inspect.getfile(type(model)))
        receipt["runtime_model_source"] = {"path": str(source_file), "sha256": sha256(source_file)}
        receipt["model_config_sha256"] = sha256(a.model_dir / "config.json")
        receipt["image_sha256"] = sha256(a.image)
        receipt["model_parameter_count"] = sum(p.numel() for p in model.parameters())
        receipt["frozen_vlm"] = not any(p.requires_grad for p in model.parameters())
        receipt["checkpoint_commit_hint"] = getattr(model.config, "_commit_hash", None)

        image = Image.open(a.image)
        if image.mode != "RGB":
            raise ValueError(f"Require an explicitly rendered RGB image, got {image.mode}")
        receipt["rgb_image_size"] = list(image.size)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": a.question}]}]
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        native = dict(processor(text=[prompt_text], images=[image], return_tensors="pt", max_pixels=a.max_image_pixels))
        native = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in native.items()}
        required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"}
        if not required.issubset(native):
            raise RuntimeError(f"5.15 processor contract missing {sorted(required - set(native))}")
        if native["input_ids"].shape[0] != 1 or native["image_grid_thw"].shape[0] != 1:
            raise RuntimeError("Smoke supports batch 1, one RGB image")
        if not bool(native["attention_mask"].all()):
            raise RuntimeError("Unpadded single prompt required")
        native_len = native["input_ids"].shape[1]
        allowed = required | {"token_type_ids"}
        if set(native) - allowed:
            raise RuntimeError(f"Unreviewed processor fields: {sorted(set(native) - allowed)}")
        image_mask = native["input_ids"] == model.config.image_token_id
        if not torch.equal(native["mm_token_type_ids"] == 1, image_mask):
            raise RuntimeError("Native processor image token and multimodal type masks differ")
        end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if end_id is None or end_id == processor.tokenizer.unk_token_id:
            raise RuntimeError("Could not identify user message terminator")
        ends = (native["input_ids"][0] == end_id).nonzero().flatten()
        if not len(ends):
            raise RuntimeError("No user message terminator in rendered prompt")
        insertion = int(ends[-1])
        if insertion <= int(image_mask[0].nonzero().max()):
            raise RuntimeError("EO slots must follow all native image placeholders")
        # An ordinary existing vocabulary token is replaced only at explicit positions.
        placeholder_id = processor.tokenizer.encode("x", add_special_tokens=False)[0]
        if placeholder_id in processor.tokenizer.all_special_ids:
            raise RuntimeError("Expected an ordinary non-special placeholder token")
        prefix = dict(native)
        for key in ["input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"]:
            if key not in native:
                continue
            value = placeholder_id if key == "input_ids" else (1 if key == "attention_mask" else 0)
            extra = torch.full((1, a.eo_slots), value, dtype=native[key].dtype, device=device)
            prefix[key] = torch.cat([native[key][:, :insertion], extra, native[key][:, insertion:]], dim=1)
        prefix_len = prefix["input_ids"].shape[1]
        positions = torch.arange(insertion, insertion + a.eo_slots, device=device)
        if prefix_len > a.max_prompt_tokens:
            raise RuntimeError(f"Prompt {prefix_len} tokens exceeds bounded maximum {a.max_prompt_tokens}")
        if bool((prefix["mm_token_type_ids"][0, positions] != 0).any()):
            raise RuntimeError("EO slots must be ordinary text modality")
        if int((prefix["input_ids"] == model.config.image_token_id).sum()) != int(image_mask.sum()):
            raise RuntimeError("EO insertion changed native image token count")
        receipt["prompt_contract"] = {"native_length": native_len, "with_eo_length": prefix_len,
            "eo_slot_positions": positions.tolist(), "placeholder_token_id": placeholder_id,
            "native_image_token_count": int(image_mask.sum()), "image_grid_thw": native["image_grid_thw"].tolist(),
            "eo_modality_type": 0, "slot_location": "Before final user im_end, after native RGB tokens"}

        encoder = None
        if a.eo_token_file:
            raw = np.load(a.eo_token_file, allow_pickle=False)
            feature_input = torch.as_tensor(raw, dtype=torch.float32, device=device)
            if feature_input.ndim == 2:
                feature_input = feature_input.unsqueeze(0)
            feature_input = feature_input.detach().requires_grad_(True)
            encode = lambda: feature_input
            receipt["eo_source"] = {"kind": "precomputed_features", "file": str(a.eo_token_file), "sha256": sha256(a.eo_token_file),
                                    "encoder_gradient_claim": False}
        else:
            mod_path, function_name = a.eo_provider.rsplit(":", 1)
            if mod_path.endswith(".py"):
                spec = importlib.util.spec_from_file_location("oe4_local_provider", mod_path)
                if spec is None or spec.loader is None:
                    raise RuntimeError("Could not load local provider")
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                provider_file = Path(mod_path)
            else:
                module = importlib.import_module(mod_path)
                provider_file = Path(inspect.getfile(module))
            cfg = json.loads(a.eo_provider_config.read_text()) if a.eo_provider_config else {}
            provider = getattr(module, function_name)(config=cfg, device=device, dtype=dtype)
            encoder, encode = provider["model"], provider["encode"]
            if not isinstance(encoder, torch.nn.Module) or not callable(encode):
                raise TypeError("Provider must return model nn.Module and callable encode")
            if not any(p.requires_grad for p in encoder.parameters()):
                raise RuntimeError("Provider encoder has no trainable parameters")
            encoder.eval()  # Deterministic one-step smoke; evaluation mode does not disable autograd.
            receipt["eo_source"] = {"kind": "live_encoder_provider", "provider_file_sha256": sha256(provider_file),
                                    "config": cfg, "metadata": provider.get("metadata", {}), "encoder_gradient_claim": True}

        def validate_features(x):
            if x.ndim != 3 or x.shape[0] != 1 or not (1 <= x.shape[1] <= 65536 and 1 <= x.shape[2] <= 8192):
                raise RuntimeError(f"Expected finite [1,N,D] EO features, got {tuple(x.shape)}")
            if not bool(torch.isfinite(x).all()):
                raise RuntimeError("Nonfinite EO features")
            return x

        # No-grad is permitted here only for pre-training shape/inference diagnostics.
        with torch.no_grad():
            features0 = validate_features(encode())
        input_dim = features0.shape[-1]
        adapter = torch.nn.Sequential(torch.nn.LayerNorm(input_dim), torch.nn.Linear(input_dim, model.config.text_config.hidden_size)).to(device)
        torch.nn.init.normal_(adapter[1].weight, mean=0.0, std=0.02 / math.sqrt(input_dim))
        torch.nn.init.zeros_(adapter[1].bias)

        def adapt(features):
            features = validate_features(features)
            if features.shape[-1] != input_dim:
                raise RuntimeError("EO feature width changed")
            # Contiguous token pooling is solely a bounded interface choice, not a spatial method.
            pooled = F.adaptive_avg_pool1d(features.float().transpose(1, 2), a.eo_slots).transpose(1, 2)
            return adapter(pooled).to(dtype)

        receipt["eo_shape"] = list(features0.shape)
        receipt["adapter"] = {"kind": "FP32 LayerNorm + Linear, cast to VLM dtype", "parameters": sum(p.numel() for p in adapter.parameters()),
                              "pooling": "adaptive average pooling over native flattened token order", "slots": a.eo_slots}
        state = {"active": False, "embeddings": None, "injections": 0, "decode_bypasses": 0}

        def embedding_hook(module, args, output):
            if not state["active"]:
                return output
            ids = args[0]
            if ids.ndim != 2 or ids.shape[0] != 1:
                raise RuntimeError("Unexpected batch or embedding call during active EO injection")
            if ids.shape[1] < prefix_len:
                if ids.shape[1] != 1:
                    raise RuntimeError("Only one-token cached decoding is supported")
                state["decode_bypasses"] += 1
                return output
            if not torch.equal(ids[:, :prefix_len], prefix["input_ids"]):
                raise RuntimeError("Active EO injection received a different prompt")
            eo = state["embeddings"]
            if eo is None or tuple(eo.shape) != (1, a.eo_slots, output.shape[-1]):
                raise RuntimeError("EO embedding shape mismatch")
            result = output.clone()
            result[:, positions, :] = eo.to(device=output.device, dtype=output.dtype)
            state["injections"] += 1
            return result

        visual_captures = []

        def tensor_digest(x):
            x = x.detach().contiguous().cpu()
            return {"shape": list(x.shape), "dtype": str(x.dtype), "sha256": hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()}

        def capture_native_visual(module, args, kwargs):
            mask, deep = kwargs.get("visual_pos_masks"), kwargs.get("deepstack_visual_embeds")
            if mask is None:
                return
            if mask.dtype != torch.bool or tuple(mask.shape) != tuple(kwargs["inputs_embeds"].shape[:2]):
                raise RuntimeError("Native visual capture requires a boolean mask matching the current sequence")
            if deep is None:
                raise RuntimeError("Native visual capture is missing DeepStack features")
            visual_captures.append({"image_embeddings": tensor_digest(kwargs["inputs_embeds"][mask]),
                                    "deepstack": [tensor_digest(x) for x in deep],
                                    "visual_token_count": int(mask.sum()), "mask_shape": list(mask.shape),
                                    "visual_positions": mask.nonzero().tolist()})

        handles.append(model.model.language_model.register_forward_pre_hook(capture_native_visual, with_kwargs=True))
        expected_deepstack = len(model.config.vision_config.deepstack_visual_indexes)
        baseline_capture_start = len(visual_captures)
        with torch.no_grad():
            baseline_logits = model(**native, use_cache=False, logits_to_keep=1).logits.detach().float().cpu()
        baseline_visual = check_fresh_visual_capture(
            visual_captures, baseline_capture_start, list(native["input_ids"].shape),
            image_mask.nonzero().tolist(), expected_deepstack, model.config.text_config.hidden_size, "native baseline")
        handles.append(model.get_input_embeddings().register_forward_hook(embedding_hook))
        with torch.no_grad():
            inactive_logits = model(**native, use_cache=False, logits_to_keep=1).logits.detach().float().cpu()
        receipt["native_rgb_parity"] = {"bitwise_equal": bool(torch.equal(baseline_logits, inactive_logits)),
            "max_abs_logit_delta": float((baseline_logits - inactive_logits).abs().max()), "eo_slots_present": False,
            "interpretation": "Installing an inactive hook must leave the original RGB prompt exactly unchanged"}
        if not receipt["native_rgb_parity"]["bitwise_equal"]:
            raise RuntimeError("Inactive wrapper changed native RGB logits")

        def append_tokens(batch, token_ids):
            result = dict(batch)
            for key in ["input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"]:
                if key not in result:
                    continue
                extra = token_ids if key == "input_ids" else torch.full_like(token_ids, 1 if key == "attention_mask" else 0)
                result[key] = torch.cat([result[key], extra.to(result[key].dtype)], dim=1)
            return result

        # Use Qwen's actual generation path, then compare its second raw logits to
        # a full reference pass using the SAME first generated token.
        state["active"] = True
        with torch.no_grad():
            state["embeddings"] = adapt(features0)
            generation_config = copy.deepcopy(model.generation_config)
            generation_config.do_sample = False
            generation_config.num_beams = 1
            generation_config.max_new_tokens = 2
            generation_config.min_new_tokens = 2
            generation_config.return_dict_in_generate = True
            generation_config.output_logits = True
            generation_config.use_cache = True
            generation_config.temperature = 1.0
            generation_config.top_p = 1.0
            generation_config.top_k = 50
            # Establish the same cached/full gate on the original RGB prompt with
            # EO injection inactive, before evaluating the EO-augmented prompt.
            state["active"] = False
            native_cache_capture_start = len(visual_captures)
            native_generation = model.generate(**native, generation_config=copy.deepcopy(generation_config))
            if len(native_generation.logits) != 2 or native_generation.past_key_values is None:
                raise RuntimeError("Native RGB baseline did not return two logits and a cache")
            native_cache_visual = check_fresh_visual_capture(
                visual_captures, native_cache_capture_start, list(native["input_ids"].shape),
                image_mask.nonzero().tolist(), expected_deepstack,
                model.config.text_config.hidden_size, "native RGB cached baseline")
            if native_cache_visual != baseline_visual:
                raise RuntimeError("Native RGB cached baseline changed original visual features")
            native_cache_length = int(native_generation.past_key_values.get_seq_length())
            if native_cache_length != native_len + 1:
                raise RuntimeError(f"Unexpected native RGB cached sequence length {native_cache_length}")
            native_first_token = native_generation.sequences[:, native_len:native_len + 1]
            native_reference = model(**append_tokens(native, native_first_token), use_cache=False,
                                     logits_to_keep=1).logits[:, -1].float().cpu()
            native_cached = native_generation.logits[1].float().cpu()
            native_cache_close = bool(torch.allclose(native_cached, native_reference,
                                                    atol=a.cache_atol, rtol=a.cache_rtol))
            native_same_top1 = bool(torch.equal(native_cached.argmax(-1), native_reference.argmax(-1)))
            receipt["native_cache_contract"] = {
                "allclose": native_cache_close, "top1_equal": native_same_top1,
                "max_abs_logit_delta": float((native_cached - native_reference).abs().max()),
                "mean_abs_logit_delta": float((native_cached - native_reference).abs().mean()),
                "atol": a.cache_atol, "rtol": a.cache_rtol,
                "cached_sequence_length": native_cache_length,
                "generated_token_ids": native_generation.sequences[:, native_len:].tolist(),
                "generated_text": processor.tokenizer.decode(
                    native_generation.sequences[0, native_len:], skip_special_tokens=True),
                "eo_injection_active": False,
                "comparison": "Original RGB cached raw second logits vs same-prefix full recomputation",
            }
            write_json(receipt_path, receipt)
            if not native_cache_close or not native_same_top1:
                raise RuntimeError("Original native RGB cached generation disagrees with full recomputation")
            if state["injections"] != 0 or state["decode_bypasses"] != 0:
                raise RuntimeError("Inactive EO hook altered native RGB cache baseline counters")
            del native_generation, native_reference, native_cached
            state["active"] = True
            generation_capture_start = len(visual_captures)
            generation = model.generate(**prefix, generation_config=generation_config)
            if len(generation.logits) != 2:
                raise RuntimeError("Expected two native cached generation steps")
            injection_count = state["injections"]
            bypass_count = state["decode_bypasses"]
            if injection_count != 1 or bypass_count != 1:
                raise RuntimeError(f"Expected 1 EO prefill and 1 cached bypass, got {injection_count}/{bypass_count}")
            generated_native_visual = check_fresh_visual_capture(
                visual_captures, generation_capture_start, list(prefix["input_ids"].shape),
                (prefix["input_ids"] == model.config.image_token_id).nonzero().tolist(),
                expected_deepstack, model.config.text_config.hidden_size, "EO cached generation")
            generation_fresh_capture_count = len(visual_captures) - generation_capture_start
            # Mask shape changes with extra EO text slots. Compare image/DeepStack
            # feature bytes only after the current mask and image positions pass.
            feature_keys = ["image_embeddings", "deepstack", "visual_token_count"]
            if any(generated_native_visual[key] != baseline_visual[key] for key in feature_keys):
                raise RuntimeError("EO insertion changed RGB embedding or DeepStack feature bytes")
            if generation.past_key_values is None:
                raise RuntimeError("Native generation did not return a cache")
            cache_length = int(generation.past_key_values.get_seq_length())
            if cache_length != prefix_len + 1:
                raise RuntimeError(f"Unexpected cached sequence length {cache_length}")
            first_token = generation.sequences[:, prefix_len:prefix_len + 1]
            reference = model(**append_tokens(prefix, first_token), use_cache=False, logits_to_keep=1).logits[:, -1].float().cpu()
            cached = generation.logits[1].float().cpu()
            cache_close = bool(torch.allclose(cached, reference, atol=a.cache_atol, rtol=a.cache_rtol))
            same_top1 = bool(torch.equal(cached.argmax(-1), reference.argmax(-1)))
            receipt["cache_contract"] = {"allclose": cache_close, "top1_equal": same_top1,
                "max_abs_logit_delta": float((cached - reference).abs().max()), "mean_abs_logit_delta": float((cached - reference).abs().mean()),
                "atol": a.cache_atol, "rtol": a.cache_rtol, "prefill_injections": injection_count, "cached_decode_bypasses": bypass_count,
                "cached_sequence_length": cache_length, "generated_token_ids": generation.sequences[:, prefix_len:].tolist(),
                "generated_text": processor.tokenizer.decode(generation.sequences[0, prefix_len:], skip_special_tokens=True),
                "comparison": "Native generate cached raw second logits vs same-prefix full recomputation"}
            if not cache_close or not same_top1:
                raise RuntimeError("EO cached generation disagrees with full recomputation")
        receipt["native_visual_preservation"] = {"exact_feature_hash_parity": True,
            "generation_fresh_prefill_capture_count": generation_fresh_capture_count,
            "native_features": baseline_visual, "generation_features": generated_native_visual}
        del generation, features0
        receipt["status"] = "native_rgb_deepstack_and_cache_contract_passed"
        write_json(receipt_path, receipt)
        print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path)}), flush=True)

        # This block MUST retain autograd through the frozen VLM to EO features.
        target_ids = processor.tokenizer.encode(a.target, add_special_tokens=False)
        if not (1 <= len(target_ids) <= 64):
            raise RuntimeError("Engineering target must contain 1..64 tokens")
        target_tensor = torch.tensor([target_ids], dtype=torch.long, device=device)
        training_inputs = append_tokens(prefix, target_tensor)
        param_groups = [{"params": list(adapter.parameters()), "lr": a.adapter_lr}]
        if encoder is not None:
            param_groups.append({"params": [p for p in encoder.parameters() if p.requires_grad], "lr": a.encoder_lr})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            features = validate_features(encode())
            if not features.requires_grad:
                raise RuntimeError("EO provider detached its output; no gradient can reach encoder")
            features.retain_grad()
            state["embeddings"] = adapt(features)
            state["embeddings"].retain_grad()
            train_output = model(**training_inputs, use_cache=False, logits_to_keep=len(target_ids) + 1)
            loss = F.cross_entropy(train_output.logits[:, :-1].float().reshape(-1, train_output.logits.shape[-1]), target_tensor.reshape(-1))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite engineering CE loss")
            loss.backward()

        def gradient_stats(named_parameters):
            rows = []
            for name, p in named_parameters:
                if p.grad is not None:
                    if not bool(torch.isfinite(p.grad).all()):
                        raise RuntimeError(f"Nonfinite gradient: {name}")
                    norm = float(p.grad.float().norm())
                    rows.append({"name": name, "l2_norm": norm, "nonzero": norm > 0})
            return {"tensors_with_grad": len(rows), "tensors_with_nonzero_grad": sum(r["nonzero"] for r in rows), "rows": rows}

        for name, tensor in [("EO features", features), ("EO embeddings", state["embeddings"])]:
            if tensor.grad is None or not bool(torch.isfinite(tensor.grad).all()) or not bool(torch.count_nonzero(tensor.grad)):
                raise RuntimeError(f"No finite nonzero gradient at {name}")
        adapter_stats = gradient_stats(adapter.named_parameters())
        if not adapter_stats["tensors_with_nonzero_grad"]:
            raise RuntimeError("No adapter gradient")
        encoder_stats = gradient_stats(encoder.named_parameters()) if encoder is not None else None
        if encoder_stats is not None and not encoder_stats["tensors_with_nonzero_grad"]:
            raise RuntimeError("No actual encoder gradient")
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("A frozen VLM parameter received a gradient")
        tracked = []
        for group_name, module in [("adapter", adapter), ("encoder", encoder)]:
            if module is None:
                continue
            candidates = [(n, p) for n, p in module.named_parameters() if p.grad is not None and bool(torch.count_nonzero(p.grad))]
            for name, p in candidates[:4]:
                tracked.append((group_name, name, p, p.detach().clone()))
        all_trainable = [p for group in param_groups for p in group["params"]]
        grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        updates = [{"module": group, "name": name, "max_abs_delta": float((p.detach() - before).abs().max())}
                   for group, name, p, before in tracked]
        for group in ["adapter"] + (["encoder"] if encoder is not None else []):
            if not any(row["module"] == group and row["max_abs_delta"] > 0 for row in updates):
                raise RuntimeError(f"No sampled actual {group} parameter change")
        receipt["one_step"] = {"ce_loss": float(loss.detach()), "ce_target": a.target, "ce_target_is_evaluation_gold": False,
            "grad_norm_before_clip": float(grad_norm), "eo_feature_gradient_norm": float(features.grad.float().norm()),
            "eo_embedding_gradient_norm": float(state["embeddings"].grad.float().norm()),
            "adapter_gradients": adapter_stats, "encoder_gradients": encoder_stats, "sampled_parameter_updates": updates,
            "frozen_vlm_parameter_gradients": 0, "vlm_parameters_in_optimizer": 0,
            "encoder_update_verified": encoder is not None, "optimizer_steps": 1}
        state["active"] = False
        state["embeddings"] = None
        # Verify original RGB behavior survives the EO-side optimizer step too.
        with torch.no_grad():
            after_logits = model(**native, use_cache=False, logits_to_keep=1).logits.detach().float().cpu()
        receipt["native_rgb_parity"]["after_eo_step_bitwise_equal"] = bool(torch.equal(baseline_logits, after_logits))
        if not receipt["native_rgb_parity"]["after_eo_step_bitwise_equal"]:
            raise RuntimeError("Original native RGB logits changed after EO-only optimization")
        receipt["status"] = "PASS_live_encoder_adapter_vlm_gradient_contract" if encoder is not None else "PASS_adapter_feature_input_contract_ENCODER_NOT_TESTED"
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            receipt["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        receipt["wall_seconds"] = time.monotonic() - start
        receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(receipt_path, receipt)
        print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path)}), flush=True)
    except Exception as e:
        receipt.update({"status": "FAILED", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(), "wall_seconds": time.monotonic() - start})
        write_json(receipt_path, receipt)
        raise
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
