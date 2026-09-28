#!/usr/bin/env python3
"""OE1 bounded feasibility pilot; three equal-budget arms, then one final evaluation.

The frozen LLM still participates in autograd with respect to its input embeddings.
This is single-image BEN question alignment, not temporal pretraining or a CVPR claim.
No resume, checkpoint selection, data downloads, or automatic GPU selection.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from eo_model import EOImageEncoder, S2_BANDS, acquisition_day, sha256_file

ARMS = ("frozen", "joint", "blind")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".writing")
    with tmp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode()
        digest.update(len(header).to_bytes(8, "big")); digest.update(header)
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big")); digest.update(raw)
    return digest.hexdigest()


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def validate_data(data_dir, expected_patches=None):
    """No exclusions: reject missing/invalid data rather than change the population."""
    data_dir = Path(data_dir).resolve(strict=True)
    expected_patches = expected_patches or {"train": 512, "dev": 128}
    manifest_path = data_dir / "manifest.json"
    manifest_hash = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("status") == "complete" and manifest.get("seed") == 20260926, "Prepared data manifest not complete or wrong data-selection seed")
    require(manifest.get("original_question_wording_preserved") is True and manifest.get("text_test_or_bench_rows_materialized") is False, "Unexpected question provenance")
    require(manifest.get("questions_per_patch") == 2 and manifest.get("image_array_hashes_unique") is True, "Prepared image/QA contract differs")
    require(manifest.get("cross_split_minimum_center_km", -1) >= 2, "Prepared spatial separation not satisfied")
    require(manifest.get("bands") == list(S2_BANDS), "Prepared band order changed")
    prepared = {p["patch_id"]: p for p in manifest["patches"]}
    require(len(prepared) == len(manifest["patches"]) == sum(expected_patches.values()), "Prepared patch inventory differs")
    rows, images, image_keys, source_hashes, partitions = [], [], {}, {}, {}
    source_hashes[str(manifest_path)] = manifest_hash
    patch_meta, ids, pixel_hashes = {}, set(), set()
    for split in ("train", "dev"):
        path = data_dir / (split + ".jsonl")
        source_hashes[str(path)] = sha256_file(path)
        split_meta = manifest["splits"][split]
        require(source_hashes[str(path)] == split_meta["question_file_sha256"], "Question file differs from prepared manifest")
        require(split_meta["patches"] == expected_patches[split] and split_meta["questions"] == 2 * expected_patches[split], "Prepared split budget differs")
        current = read_rows(path)
        require(len(current) == 2 * expected_patches[split], "Question coverage differs: " + split)
        partitions[split] = []
        for row in current:
            require(isinstance(row.get("id"), str) and row["id"] and row["id"] not in ids, "Missing/duplicate question ID")
            require(isinstance(row.get("patch_id"), str) and row["patch_id"], "Missing patch ID")
            require(isinstance(row.get("input"), str) and row["input"].strip() and "<EO>" not in row["input"], "Invalid original question")
            require(row.get("output") in ("yes", "no"), "Source output must be exact yes/no")
            require(isinstance(row.get("bands"), list) and len(row["bands"]) == 12 and set(row["bands"]) == set(S2_BANDS), "Incomplete/duplicate native S2 bands")
            acquisition_day(row.get("date"))
            require(row["patch_id"] in prepared, "Question patch absent from prepared manifest")
            meta = prepared[row["patch_id"]]
            require(meta["split"] == split == row.get("split") and meta["timestamp_utc"] == row["date"] == row.get("timestamp_utc"), "Prepared observation/date/split mismatch")
            require(meta["mgrs"] == row.get("mgrs") and isinstance(row["mgrs"], str) and row["mgrs"], "Prepared MGRS mismatch")
            require(meta["image_path"] == row["image_path"] and meta["bands"] == row["bands"], "Prepared image/band metadata mismatch")
            relative = Path(row["image_path"])
            require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe image path")
            image_path = (data_dir / relative).resolve(strict=True)
            require(image_path.is_relative_to(data_dir) and image_path.suffix == ".npy", "Images must be local .npy arrays")
            signature = (str(image_path), row["date"], tuple(row["bands"]))
            require(row["patch_id"] not in patch_meta or patch_meta[row["patch_id"]] == signature, "Inconsistent patch metadata")
            patch_meta[row["patch_id"]] = signature
            if str(image_path) not in image_keys:
                first_hash = sha256_file(image_path)
                array = np.load(image_path, allow_pickle=False)
                require(array.shape == (12, 120, 120) and np.issubdtype(array.dtype, np.number), "Expected native aligned S2 [12,120,120]")
                require(np.isfinite(array).all() and array.min() >= 0 and array.max() <= 65535, "Invalid raw DN")
                pixel_hash = hashlib.sha256(array.tobytes()).hexdigest()
                require(pixel_hash == meta["image_array_sha256"], "Prepared pixel-array hash mismatch")
                require(pixel_hash not in pixel_hashes, "Duplicate raw pixels under different image paths")
                pixel_hashes.add(pixel_hash)
                require(list(array.shape) == meta["shape"] and str(array.dtype) == meta["dtype"], "Prepared pixel dtype/shape mismatch")
                require(first_hash == sha256_file(image_path), "Image changed during read")
                image_keys[str(image_path)] = len(images)
                source_hashes[str(image_path)] = first_hash
                images.append(torch.from_numpy(array.astype(np.float32, copy=True)))
            index = len(rows)
            rows.append(dict(row, partition=split, image_index=image_keys[str(image_path)]))
            partitions[split].append(index); ids.add(row["id"])
    require({rows[i]["patch_id"] for i in partitions["train"]}.isdisjoint(rows[i]["patch_id"] for i in partitions["dev"]), "Train/dev patch overlap")
    require({rows[i]["image_index"] for i in partitions["train"]}.isdisjoint(rows[i]["image_index"] for i in partitions["dev"]), "Train/dev raw image overlap")
    require({rows[i]["mgrs"] for i in partitions["train"]}.isdisjoint(rows[i]["mgrs"] for i in partitions["dev"]), "Train/dev MGRS overlap")
    require(set(patch_meta) == set(prepared), "Prepared patches missing or added")
    require(len(images) == sum(expected_patches.values()), "Prepared patches do not have distinct raw image files")
    for split, indices in partitions.items():
        labels = collections.Counter(rows[i]["output"] for i in indices)
        require(labels == {"yes": expected_patches[split], "no": expected_patches[split]} == manifest["splits"][split]["answers"], "Presence-label balance differs")
        patch_labels = collections.defaultdict(list)
        for i in indices:
            patch_labels[rows[i]["patch_id"]].append(rows[i]["output"])
        require(len(patch_labels) == expected_patches[split] and all(sorted(labels) == ["no", "yes"] for labels in patch_labels.values()), "Expected one yes and one no per prepared patch")
    require(sha256_file(manifest_path) == manifest_hash, "Prepared manifest changed during validation")
    return rows, torch.stack(images), partitions, source_hashes


def batches_for(train_indices, steps, seed, batch_size=8):
    require(len(train_indices) >= batch_size, "At least eight training questions required")
    rng, batches = np.random.default_rng(seed), []
    while len(batches) < steps:
        order = rng.permutation(train_indices).tolist()
        for start in range(0, len(order) - batch_size + 1, batch_size):
            batches.append(order[start:start + batch_size])
            if len(batches) == steps:
                break
    return batches


def donor_map(rows, dev_indices):
    groups = collections.defaultdict(list)
    for index in dev_indices:
        groups[(rows[index]["input"], rows[index]["output"])].append(index)
    result = {}
    for index in dev_indices:
        item = rows[index]
        candidates = groups[(item["input"], "no" if item["output"] == "yes" else "yes")]
        candidates = [j for j in candidates if rows[j]["patch_id"] != item["patch_id"]]
        if candidates:
            result[index] = min(candidates, key=lambda j: rows[j]["id"])
    return result


class Projector(nn.Module):
    def __init__(self, input_dim, hidden_size, embedding_rms):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))
        self.output_norm = nn.LayerNorm(hidden_size)
        self.gain = nn.Parameter(torch.ones(()))
        self.register_buffer("embedding_rms", torch.tensor(float(embedding_rms)))

    def forward(self, value):
        return self.output_norm(self.mlp(self.norm(value.float()))) * self.embedding_rms * self.gain


def tokenize_prompts(tokenizer, rows, max_tokens=512):
    prompts, answer_ids = [], {}
    for word in ("yes", "no"):
        tokens = tokenizer.encode(word, add_special_tokens=False)
        require(len(tokens) == 1, "Answer is not a single tokenizer token: " + word)
        answer_ids[word] = tokens[0]
    require(answer_ids["yes"] != answer_ids["no"], "Answer token collision")
    for row in rows:
        user = "Sentinel-2 satellite observation: <EO>\n" + row["input"] + "\nAnswer with yes or no."
        chat = tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        require(chat.count("<EO>") == 1, "EO marker must occur exactly once")
        prefix, suffix = chat.split("<EO>")
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        require(0 < len(prefix_ids) + 16 + len(suffix_ids) <= max_tokens and suffix_ids, "Prompt exceeds fixed untruncated context")
        prompts.append({"id": row["id"], "original_input": row["input"], "user_text": user,
                        "chat_text": chat, "prefix_ids": prefix_ids, "suffix_ids": suffix_ids})
    return prompts, answer_ids


def last_token_logits(llm, projector, prompts, indices, features, device):
    """Right-padding with gather at each last actual token; no answer fed as input."""
    projected = projector(features).to(dtype=llm.get_input_embeddings().weight.dtype)
    require(projected.shape[:2] == (len(indices), 16), "Expected sixteen image tokens")
    sequences = []
    embedding = llm.get_input_embeddings()
    for j, index in enumerate(indices):
        prompt = prompts[index]
        pre = embedding(torch.tensor(prompt["prefix_ids"], dtype=torch.long, device=device))
        post = embedding(torch.tensor(prompt["suffix_ids"], dtype=torch.long, device=device))
        sequences.append(torch.cat((pre, projected[j], post), dim=0))
    length = max(map(len, sequences))
    embeds = torch.stack([F.pad(value, (0, 0, 0, length - len(value))) for value in sequences])
    lengths = torch.tensor([len(value) for value in sequences], device=device)
    attention = torch.arange(length, device=device)[None] < lengths[:, None]
    output = llm(inputs_embeds=embeds, attention_mask=attention.long(), use_cache=False)
    logits = output.logits[torch.arange(len(indices), device=device), lengths - 1].float()
    require(bool(torch.isfinite(logits).all()), "Nonfinite first-answer-token logits")
    return logits


def metrics(records):
    result = {"n": len(records), "support": dict(collections.Counter(r["eval_gold"] for r in records))}
    for key in ("parsed", "constrained_prediction"):
        recalls = {gold: (sum(r[key] == gold for r in records if r["eval_gold"] == gold) / result["support"][gold])
                   for gold in ("yes", "no") if result["support"].get(gold)}
        result[key] = {"accuracy": sum(r[key] == r["eval_gold"] for r in records) / len(records) if records else None,
                       "balanced_accuracy": sum(recalls.values()) / 2 if len(recalls) == 2 else None,
                       "recall_by_label": recalls,
                       "invalid_count": sum(r[key] is None for r in records)}
    return result


def validate_checkpoint_probe(reference, question_ids, logits):
    require(reference["question_ids"] == question_ids, "Checkpoint probe questions differ from pre-save training probe")
    actual = logits.detach().cpu()
    require(torch.equal(reference["logits"], actual), "Reloaded checkpoint does not reproduce original trained-state logits exactly")
    return {"split": "train", "n_questions": len(question_ids), "logits_bit_exact": True,
            "logits_sha256": state_hash({"logits": actual})}


def run(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def timeout(_signal, _frame):
        raise TimeoutError("OE1 fixed wall-time bound exceeded; incomplete run is invalid")
    prior_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_wall_seconds)
    try:
        write(out / "status.json", {"status": "preparing", "valid": False, "started_at": now()})
        require(1 <= args.steps <= 256 and args.seed == 17, "Pilot steps must be 1..256 and seed fixed at 17")
        require(1 <= args.max_wall_seconds <= 3600, "Maximum wall time is 3600s")
        require(torch.cuda.is_available(), "CUDA required; use separate CPU synthetic tests")
        torch.set_num_threads(2); torch.cuda.set_device(0)
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda:0")
        rows, images, partitions, data_hashes = validate_data(args.data_dir)
        schedule = batches_for(partitions["train"], args.steps, args.seed)
        donors = donor_map(rows, partitions["dev"])
        config = {"schema": "oe1-bentxt-feasibility-v0", "arms": list(ARMS), "seed": args.seed, "training_seed": args.seed, "data_selection_seed": 20260926,
                  "steps_per_arm": args.steps, "batch_size": 8, "exposures_per_arm": args.steps * 8,
                  "projector_lr": .001, "joint_encoder_lr": .00001, "weight_decay": .01, "clip_norm": 1.,
                  "encoder_eval_mode_in_all_arms": True, "encoder_model": "OlmoEarth-v1-Tiny",
                  "training_target": "full-vocabulary next-token CE on source yes/no token",
                  "evaluation": "all arms trained before final dev evaluation; free-vocabulary argmax primary",
                  "data_dir": str(Path(args.data_dir).resolve()), "eo_checkpoint": str(Path(args.model_dir).resolve()),
                  "data_manifest_sha256": data_hashes[str(Path(args.data_dir).resolve() / "manifest.json")],
                  "llm_dir": str(Path(args.llm_dir).resolve()), "max_wall_seconds": args.max_wall_seconds,
                  "n_train": len(partitions["train"]), "n_dev": len(partitions["dev"]),
                  "data_files_sha256": data_hashes, "code_sha256": {p.name: sha256_file(p) for p in [Path(__file__), Path(__file__).with_name("eo_model.py")]}}
        write(out / "run_config.json", config)
        write(out / "batches.json", [[rows[i]["id"] for i in batch] for batch in schedule])
        write(out / "donor_plan.json", {"mapping": {rows[i]["id"]: rows[j]["id"] for i, j in donors.items()},
               "eligible": len(donors), "total_dev": len(partitions["dev"]),
               "rule": "Exact original question text, opposite source answer, different patch; smallest donor ID; swap image AND its actual date (observation swap, not pixel-only)"})
        write(out / "items.json", rows)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        write(out / "runtime.json", {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
              "transformers": importlib.metadata.version("transformers"),
              "olmoearth_pretrain_minimal": importlib.metadata.version("olmoearth-pretrain-minimal"),
              "cuda_runtime": torch.version.cuda, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "logical_gpu": 0, "gpu_name": torch.cuda.get_device_name(0)})
        tokenizer = AutoTokenizer.from_pretrained(args.llm_dir, local_files_only=True)
        llm = AutoModelForCausalLM.from_pretrained(args.llm_dir, dtype=torch.bfloat16, local_files_only=True).to(device).eval()
        llm.requires_grad_(False)
        llm_initial_hash = state_hash(llm.state_dict())
        llm_versions = {name: param._version for name, param in llm.named_parameters()}
        llm_buffer_hash = state_hash(dict(llm.named_buffers()))
        prompts, answer_ids = tokenize_prompts(tokenizer, rows)
        write(out / "prompts.json", {"answer_ids": answer_ids, "prompts": prompts})
        eo = EOImageEncoder.from_pretrained(args.model_dir, trainable=False).to(device)
        require(eo.output_dim == 192, "This pilot is fixed to official v1 Tiny (192 dimensional encoder)")
        emb_rms = float(llm.get_input_embeddings().weight.detach().float().square().mean().sqrt())
        torch.manual_seed(args.seed)
        projector = Projector(eo.output_dim, int(llm.config.hidden_size), emb_rms).to(device)
        initial_eo, initial_projector = cpu_state(eo), cpu_state(projector)
        initial_encoder_hash, initial_projector_hash = state_hash(initial_eo), state_hash(initial_projector)
        torch.save({"encoder": initial_eo, "projector": initial_projector, "seed": args.seed}, out / "initial_state.pt")
        write(out / "model_sources.json", {"eo": eo.provenance, "initial_eo_sha256": initial_encoder_hash,
              "initial_projector_sha256": initial_projector_hash, "llm_tensor_sha256": llm_initial_hash,
              "llm_buffers_sha256": llm_buffer_hash, "llm_frozen_parameter_count": sum(p.numel() for p in llm.parameters())})
        # The frozen cache is derived only from raw imagery + actual date/bands, never labels.
        representatives = {}
        for i, row in enumerate(rows):
            representatives.setdefault(row["image_index"], i)
        def raw_features(indices):
            selected = images[[rows[i]["image_index"] for i in indices]].to(device)
            require(len({tuple(rows[i]["bands"]) for i in indices}) == 1, "Batch band order differs")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return eo(selected, [rows[i]["date"] for i in indices], band_names=rows[indices[0]]["bands"])
        cached = {}
        for start in range(0, len(representatives), 8):
            indices = list(representatives.values())[start:start + 8]
            feats = raw_features(indices).detach().cpu()
            for i, feature in zip(indices, feats):
                cached[rows[i]["image_index"]] = feature
        def features(indices, arm, control="real"):
            if arm == "blind" or control == "zero":
                return torch.zeros((len(indices), 16, eo.output_dim), device=device)
            if arm == "joint":
                return raw_features(indices)
            return torch.stack([cached[rows[i]["image_index"]] for i in indices]).to(device)
        summaries = {}
        write(out / "status.json", {"status": "training", "valid": False, "started_at": now()})
        log = (out / "training_log.jsonl").open("x")
        try:
            for arm in ARMS:
                torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
                eo.load_state_dict(initial_eo, strict=True); projector.load_state_dict(initial_projector, strict=True)
                require(state_hash(eo.state_dict()) == initial_encoder_hash and state_hash(projector.state_dict()) == initial_projector_hash, "Arm initialization differs")
                eo.set_trainable(arm == "joint"); eo.train(); projector.train()
                eo.audit_counters = {"optimizer_steps_observed": 0, "encoder_update_steps": 0}
                groups = [{"params": list(projector.parameters()), "lr": .001}]
                if arm == "joint":
                    groups.append({"params": list(eo.encoder.parameters()), "lr": .00001})
                optimizer = torch.optim.AdamW(groups, weight_decay=.01)
                before_arm = eo.capture_parameter_state(); before_projector = state_hash(projector.state_dict())
                torch.cuda.reset_peak_memory_stats(); arm_start = time.monotonic(); losses = []
                for step, indices in enumerate(schedule, 1):
                    optimizer.zero_grad(set_to_none=True)
                    logits = last_token_logits(llm, projector, prompts, indices, features(indices, arm), device)
                    labels = torch.tensor([answer_ids[rows[i]["output"]] for i in indices], device=device)
                    loss = F.cross_entropy(logits, labels)
                    require(bool(torch.isfinite(loss)), "Nonfinite train loss")
                    loss.backward()
                    grad = eo.gradient_report()
                    require(grad["finite"] and (grad["nonzero_grad_tensors"] > 0 if arm == "joint" else grad["with_grad"] == 0), "Unexpected encoder gradient flow")
                    params = [p for group in optimizer.param_groups for p in group["params"]]
                    require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params), "Nonfinite trainable gradient")
                    require(any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in projector.parameters()), "No projector gradient")
                    total_norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                    optimizer.step()
                    require(all(bool(torch.isfinite(p).all()) for p in params), "Nonfinite updated parameter")
                    value = float(loss.detach()); losses.append(value)
                    record = {"arm": arm, "step": step, "loss": value, "grad_norm_pre_clip": float(total_norm),
                              "encoder_gradient": grad, "elapsed_seconds": time.monotonic() - started}
                    log.write(json.dumps(record, allow_nan=False) + "\n")
                    if step % 16 == 0 or step == args.steps:
                        log.flush(); os.fsync(log.fileno())
                        progress = dict(record, peak_gpu_bytes=torch.cuda.max_memory_allocated())
                        write(out / "progress.json", progress); print(json.dumps(progress), flush=True)
                    del logits, loss
                audit = eo.audit_after_step(before_arm)
                audit["measurement_scope"] = "one whole-arm before/after observation; not one observation per optimizer step"
                require((audit["changed_parameter_tensors"] > 0) == (arm == "joint"), "Only joint encoder must update")
                require(state_hash(projector.state_dict()) != before_projector, "Projector did not update")
                require(all(param._version == llm_versions[name] and param.grad is None for name, param in llm.named_parameters()), "Frozen LLM parameter mutated")
                require(state_hash(dict(llm.named_buffers())) == llm_buffer_hash, "Frozen LLM buffer mutated")
                eo.eval(); projector.eval()
                train_probe_ids = partitions["train"][:8]
                with torch.no_grad():
                    train_probe_logits = last_token_logits(llm, projector, prompts, train_probe_ids, features(train_probe_ids, arm), device).detach().cpu().clone()
                train_probe = {"question_ids": [rows[i]["id"] for i in train_probe_ids], "logits": train_probe_logits}
                state = {"encoder": cpu_state(eo), "projector": cpu_state(projector), "arm": arm, "steps": args.steps,
                         "pre_save_train_probe": train_probe}
                torch.save(state, out / (arm + "_final.pt"))
                summaries[arm] = {"steps": args.steps, "exposures": args.steps * 8, "loss_first": losses[0], "loss_last": losses[-1],
                    "encoder_update_audit": audit, "elapsed_seconds": time.monotonic() - arm_start,
                    "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "final_file_sha256": sha256_file(out / (arm + "_final.pt")),
                    "final_encoder_tensor_sha256": state_hash(state["encoder"]), "final_projector_tensor_sha256": state_hash(state["projector"])}
                optimizer.zero_grad(set_to_none=True); del optimizer, state
                write(out / "training_summary.json", summaries)
        finally:
            log.close()
        require(set(summaries) == set(ARMS), "All three arms must finish before evaluating")
        write(out / "status.json", {"status": "evaluating", "valid": False, "trained_arms": list(ARMS)})
        predictions, evaluations, reproduction = [], {}, {}
        with torch.no_grad():
            for arm in ARMS:
                state = torch.load(out / (arm + "_final.pt"), map_location="cpu", weights_only=True)
                require(state_hash(state["encoder"]) == summaries[arm]["final_encoder_tensor_sha256"] and state_hash(state["projector"]) == summaries[arm]["final_projector_tensor_sha256"], "Final checkpoint changed")
                eo.load_state_dict(state["encoder"]); projector.load_state_dict(state["projector"])
                eo.set_trainable(False); eo.eval(); projector.eval()
                train_probe_ids = partitions["train"][:8]
                loaded_train_logits = last_token_logits(llm, projector, prompts, train_probe_ids, features(train_probe_ids, arm), device)
                roundtrip = validate_checkpoint_probe(state["pre_save_train_probe"], [rows[i]["id"] for i in train_probe_ids], loaded_train_logits)
                probe_ids = partitions["dev"][:8]
                probe = last_token_logits(llm, projector, prompts, probe_ids, features(probe_ids, arm), device).cpu()
                eo.load_state_dict(state["encoder"]); projector.load_state_dict(state["projector"])
                probe_repeat = last_token_logits(llm, projector, prompts, probe_ids, features(probe_ids, arm), device).cpu()
                require(torch.equal(probe, probe_repeat), "Repeated saved-state load did not reproduce fixed dev logits exactly")
                reproduction[arm] = {"trained_state_roundtrip": roundtrip,
                    "same_checkpoint_repeatability": {"split": "dev", "n_questions": len(probe_ids), "logits_bit_exact": True, "logits_sha256": state_hash({"logits": probe})}}
                for control in ("real", "zero", "observation_swap"):
                    selected = [i for i in partitions["dev"] if control != "observation_swap" or i in donors]
                    group = []
                    for start in range(0, len(selected), 8):
                        indices = selected[start:start + 8]
                        presented = [donors[i] if control == "observation_swap" else i for i in indices]
                        logits = last_token_logits(llm, projector, prompts, indices, features(presented, arm, control), device)
                        winners = logits.argmax(dim=-1).tolist()
                        yn = logits[:, [answer_ids["yes"], answer_ids["no"]]]
                        probabilities = yn.softmax(dim=-1)[:, 0].tolist()
                        for j, (i, shown, winner, p_yes) in enumerate(zip(indices, presented, winners, probabilities)):
                            decoded = tokenizer.decode([winner], skip_special_tokens=False)
                            normalized = decoded.strip().lower()
                            record = {"arm": arm, "control": control, "id": rows[i]["id"], "patch_id": rows[i]["patch_id"],
                                "presented_id": rows[shown]["id"], "presented_patch_id": rows[shown]["patch_id"], "presented_date": rows[shown]["date"],
                                "source_gold": rows[i]["output"], "eval_gold": rows[shown]["output"], "argmax_token_id": winner, "raw_token": decoded,
                                "gold_basis": "same_question_donor_source_answer" if control == "observation_swap" else ("unchanged_source_label_control" if control == "zero" or arm == "blind" else "source_answer"),
                                "parsed": normalized if normalized in ("yes", "no") else None,
                                "constrained_prediction": "yes" if p_yes >= .5 else "no", "constrained_yes_probability": p_yes,
                                "yes_logit": float(yn[j, 0]), "no_logit": float(yn[j, 1]), "image_tokens_used": arm != "blind" and control != "zero"}
                            group.append(record); predictions.append(record)
                    evaluations[arm + "/" + control] = metrics(group)
        with (out / "predictions.jsonl").open("x") as stream:
            for record in predictions:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        require(state_hash(llm.state_dict()) == llm_initial_hash, "LLM tensor hash changed")
        require(all(sha256_file(Path(path)) == digest for path, digest in data_hashes.items()), "Source data changed during pilot")
        require(all(sha256_file(Path(args.model_dir) / name) == digest for name, digest in eo.provenance["checkpoint_sha256"].items()), "EO checkpoint files changed during pilot")
        require(all(sha256_file(Path(__file__).with_name(name)) == digest for name, digest in config["code_sha256"].items()), "Pilot source changed during execution")
        summary = {"schema": "oe1-bentxt-feasibility-results-v0", "valid": True, "completed_at": now(),
                   "arms": evaluations, "training": summaries, "checkpoint_reload": reproduction,
                   "llm_frozen_tensor_hash_unchanged": True, "donor_coverage": {"n": len(donors), "of": len(partitions["dev"])},
                   "prediction_rows": len(predictions), "elapsed_seconds": time.monotonic() - started,
                   "limitations": ["Single fixed-seed feasibility pilot; not a held-out benchmark or temporal-change claim.",
                       "Free-vocabulary first-token invalid answers count as incorrect; yes/no-constrained scores are secondary.",
                       "Zero-feature and blind scores measure source-label agreement, not accuracy for a new physical scene.",
                       "Native S2 and QA provenance are inherited from the frozen data builder; source QA correctness is not independently reannotated.",
                       "Joint encoder adaptation uses QA supervision, not a new self-supervised pretraining objective."]}
        write(out / "summary.json", summary)
        files = {p.name: sha256_file(p) for p in out.iterdir() if p.is_file() and p.name not in ("status.json", "manifest.json")}
        write(out / "manifest.json", {"schema": "oe1-bentxt-pilot-manifest-v0", "valid": True, "files_sha256": files, "source_files_sha256": data_hashes})
        write(out / "status.json", {"status": "completed", "valid": True, "completed_at": summary["completed_at"], "manifest_sha256": sha256_file(out / "manifest.json")})
    except BaseException as exc:
        write(out / "failure.json", {"type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "at": now()})
        write(out / "status.json", {"status": "failed", "valid": False, "at": now()})
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, prior_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-dir", required=True, help="Local official OlmoEarth-v1-Tiny config.json/weights.pth")
    parser.add_argument("--llm-dir", default="/home/work/data/olmoearth/olmo_llm/Olmo-3-7B-Instruct")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-wall-seconds", type=int, default=3600)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
