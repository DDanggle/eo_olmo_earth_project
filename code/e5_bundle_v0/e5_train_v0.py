#!/usr/bin/env python3
"""E5 equal-budget training/inference, executed only from a frozen prepared bundle.

No resume, epoch selection, historical-E2 replay claim, or test-driven stopping.
The launcher owns inherited cooperative locks. This process never releases them.
"""
import argparse
import collections
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np

ARMS = ('full', 'pair', 'later', 'delta')
SEEDS = (1, 2, 3)
PHEN = {'landslide': ('Sentinel-2', 'landslide'), 'flood': ('Sentinel-1', 'flood')}
LOCK_NAMES = {'.eo_e3_pair_dependence.lock', '.eo_reader_gpu0.lock',
              '.eo_reader_gpu1.lock', '.eo_e5_equal_budget.lock'}
REQUIRED_FILES = {'items.jsonl', 'pairs.npy', 'ordered_ids.json', 'batches.json',
                  'eval_sets.json', 'prereg.json', 'input_audit.json'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    with temporary.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def relative_file(root, name):
    p = Path(name)
    require(not p.is_absolute() and '..' not in p.parts, 'Unsafe manifest path')
    result = (root / p).resolve()
    require(result.is_relative_to(root.resolve()) and result.is_file(), 'Missing/outside manifest file: ' + name)
    return result


def tensor_state_hash(state):
    """Tensor identity independent of torch.save filename/zip serialization."""
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        header = json.dumps([key, str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
        h.update(len(header).to_bytes(8, 'big'))
        h.update(header)
        raw = value.reshape(-1).view(__import__('torch').uint8).numpy().tobytes()
        h.update(len(raw).to_bytes(8, 'big'))
        h.update(raw)
    return h.hexdigest()


def expected_batches(train_ids, seed):
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(3):
        order = rng.permutation(len(train_ids))
        result.append([[train_ids[j] for j in order[i:i + 8]] for i in range(0, len(order), 8)])
    return result


def validate_training_config(cfg):
    expected = {'model_arms': list(ARMS), 'seeds': list(SEEDS), 'epochs': 3,
                'batch_size': 8, 'optimizer': 'AdamW', 'lr': 1e-4, 'weight_decay': .01,
                'expected_updates_per_model': 1590, 'expected_exposures_per_model': 12702,
                'n_train': 4234}
    require(all(cfg['training'].get(k) == v for k, v in expected.items()), 'Training contract changed')
    require(cfg['compute']['gpu_index'] == 0 and cfg['compute']['max_runtime_minutes'] == 240
            and cfg['compute']['max_model_minutes'] == 45, 'Runtime contract changed')
    require(cfg['evaluation']['n_test'] == 1755 and cfg['evaluation']['n_answers'] == 26325,
            'Evaluation contract changed')
    require(cfg['validity']['max_parse_fail'] == .01, 'Parse validity threshold changed')


def validate_population(items, ordered, batches, pairs):
    require(len(items) == 5989 and len({i['id'] for i in items}) == 5989, 'Item coverage/duplicates')
    require(set(ordered) == {'train', 'test'}, 'Ordered ID partitions')
    lookup = {item['id']: index for index, item in enumerate(items)}
    for split, count in [('train', 4234), ('test', 1755)]:
        ids = ordered[split]
        require(len(ids) == len(set(ids)) == count, 'Ordered ID count/duplicates: ' + split)
        require(ids == [i['id'] for i in items if i['partition'] == split], 'Ordered sequence differs from frozen C0 rows')
    require(set(ordered['train']).isdisjoint(ordered['test']), 'Split ID overlap')
    counts = collections.Counter((i['partition'], i['phen'], i['kind']) for i in items)
    require(counts == {('train', 'flood', 'pos'): 1066, ('train', 'flood', 'neg'): 1066,
                      ('train', 'flood', 'hard_neg'): 1066, ('train', 'landslide', 'pos'): 518,
                      ('train', 'landslide', 'neg'): 518, ('test', 'flood', 'pos'): 457,
                      ('test', 'flood', 'neg'): 457, ('test', 'flood', 'hard_neg'): 457,
                      ('test', 'landslide', 'pos'): 192, ('test', 'landslide', 'neg'): 192}, 'Population counts changed')
    for item in items:
        require(item['answer'] == ('yes' if item['kind'] == 'pos' else 'no'), 'Source label/kind mismatch')
        require(len(item['dates']) == 2 and all(isinstance(x, str) for x in item['dates']), 'Two date strings required')
    for phen in PHEN:
        a = [i for i in items if i['partition'] == 'train' and i['phen'] == phen]
        b = [i for i in items if i['partition'] == 'test' and i['phen'] == phen]
        require({i['tile'] for i in a}.isdisjoint(i['tile'] for i in b), 'Train/test tile overlap')
        require({str(i['cluster']) for i in a}.isdisjoint(str(i['cluster']) for i in b), 'Train/test cluster overlap')
        require((len({str(i['cluster']) for i in a}), len({str(i['cluster']) for i in b}))
                == ((27, 10) if phen == 'flood' else (7, 2)), 'Cluster support differs')
    require(set(batches) == {'1', '2', '3'}, 'Batch seed coverage')
    for seed in SEEDS:
        require(batches[str(seed)] == expected_batches(ordered['train'], seed), 'Batch order differs: ' + str(seed))
        require(sum(len(ep) for ep in batches[str(seed)]) == 1590, 'Update budget differs')
        require(sum(len(batch) for ep in batches[str(seed)] for batch in ep) == 12702, 'Exposure budget differs')
    require(pairs.shape == (5989, 2, 64, 768) and pairs.dtype == np.float32, 'Pair shape/dtype')
    for index in range(len(items)):
        require(np.isfinite(pairs[index]).all(), 'Nonfinite input pair: ' + items[index]['id'])
        with np.errstate(over='ignore', invalid='ignore'):
            difference = pairs[index, 1] - pairs[index, 0]
        require(np.isfinite(difference).all(), 'Nonfinite float32 difference')
    return lookup


def make_projector(torch, hidden, emb_rms):
    nn = torch.nn

    class Projector(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(768, 2048), nn.GELU(), nn.Linear(2048, hidden))
            self.ttype = nn.Embedding(4, hidden)
            self.norm = nn.LayerNorm(768)
            self.out = nn.LayerNorm(hidden)
            self.gain = nn.Parameter(torch.tensor(1.0))
            nn.init.normal_(self.ttype.weight, std=0.02)

        def forward(self, tokens, types):
            return self.out(self.mlp(self.norm(tokens))) * (emb_rms * self.gain) + self.ttype(types) * emb_rms

    return Projector()


def source_prompt(item):
    sensor, word = PHEN[item['phen']]
    return (f'These are 2 {sensor} observations of the same area in chronological order, taken on '
            f"{', '.join(item['dates'])}: <EO> Did a {word} occur between the two observations? Answer with yes or no.")


def assemble_sequence(torch, embedding, prompt_ids, projected, with_answer, device):
    """Never embed answer IDs for evaluation; answer/EOS are teacher-forcing only."""
    require(len(projected) == 192, 'EO token count changed')
    prefix, suffix, answer = prompt_ids
    prefix, suffix = prefix.to(device), suffix.to(device)
    pieces = [embedding(prefix), projected, embedding(suffix)]
    labels = [torch.full((len(prefix) + 192 + len(suffix),), -100, device=device, dtype=torch.long)]
    if with_answer:
        answer = answer.to(device)
        pieces.append(embedding(answer))
        labels.append(answer)
    return torch.cat(pieces), torch.cat(labels)


def output_row(seed, arm, eval_arm, item, index, raw):
    require(type(seed) is int and seed in SEEDS and arm in ARMS, 'Invalid model role')
    require(eval_arm == 'native' or (arm == 'full' and eval_arm == 'full_no_delta'), 'Invalid evaluation role')
    match = re.search(r'\b(yes|no)\b', raw.strip().lower())
    return {'seed': seed, 'model_arm': arm, 'eval_arm': eval_arm, 'id': item['id'],
            'tile': item['tile'], 'cluster': item['cluster'], 'phen': item['phen'], 'kind': item['kind'],
            'source_gold': item['answer'], 'transformed_gold': None, 'answer_raw': raw,
            'parsed': match.group(1) if match else None, 'pair_index': index}


def check_locks(root):
    fdmap = json.loads(os.environ.get('E5_LOCK_FDS', '{}'))
    require(set(fdmap) == LOCK_NAMES, 'Missing inherited lock descriptors')
    for name, fd in fdmap.items():
        require(type(fd) is int and fd >= 3, 'Invalid inherited lock descriptor')
        inherited, actual = os.fstat(fd), (root / name).stat()
        require((inherited.st_dev, inherited.st_ino) == (actual.st_dev, actual.st_ino), 'Inherited lock inode differs')
        # Opening separately must fail to acquire the lock already held by the
        # inherited open-file description. Never unlock/close inherited FDs.
        separate = os.open(root / name, os.O_RDWR)
        try:
            try:
                fcntl.flock(separate, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise ValueError('Inherited descriptor does not protect lock: ' + name)
        finally:
            os.close(separate)
    return fdmap


def check_gpu_idle(root):
    check_locks(root)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and os.environ.get('E5_GPU_INDEX') == '0', 'GPU0 mapping required')
    uuid = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=20).strip()
    require(uuid and uuid == os.environ.get('E5_GPU_UUID'), 'GPU UUID differs from launcher')
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True, timeout=20)
    require(uuid not in {s.strip() for s in active.splitlines()}, 'GPU0 occupied; no allocation allowed')
    return uuid


def verify_files(out, manifest, llm_override=None):
    require(manifest['schema'] == 'e5-equal-budget-prepared-v0', 'Unexpected manifest schema')
    require((manifest['n_items'], manifest['n_train'], manifest['n_test']) == (5989, 4234, 1755), 'Manifest counts')
    require(REQUIRED_FILES <= set(manifest['files_sha256']), 'Missing required file pins')
    for name, expected in manifest['files_sha256'].items():
        require(sha(relative_file(out, name)) == expected, 'Frozen input changed: ' + name)
    require(manifest['parent_snapshot_sha256'], 'Missing parent snapshot pins')
    for name, expected in manifest['parent_snapshot_sha256'].items():
        require(sha(relative_file(out / 'parent_snapshot', name)) == expected, 'Frozen parent snapshot changed: ' + name)
    snap = (out / 'code_snapshot').resolve()
    require(Path(__file__).resolve().parent == snap, 'Execute actual frozen code_snapshot, not live code')
    require(Path(__file__).name in manifest['code_snapshot_sha256'], 'Runner not source-pinned')
    require('e5_scoring_v0.py' in manifest['code_snapshot_sha256'], 'Scoring/transform code not source-pinned')
    for name, expected in manifest['code_snapshot_sha256'].items():
        require(sha(relative_file(snap, name)) == expected, 'Frozen executable changed: ' + name)
    llm_dir = Path(manifest['llm_dir']).resolve()
    require(llm_override is None or Path(llm_override).resolve() == llm_dir, 'LLM override differs')
    require(manifest['llm_files_sha256'], 'No LLM/tokenizer hashes')
    for name, expected in manifest['llm_files_sha256'].items():
        path = Path(name).resolve()
        require(path.is_relative_to(llm_dir) and path.is_file(), 'LLM path outside declared model')
        require(sha(path) == expected, 'LLM/tokenizer file changed: ' + str(path))
    # Do not allow an unpinned extra model/tokenizer/config file to be loaded.
    load_extensions = {'.json', '.safetensors', '.bin', '.pt', '.pth', '.model', '.txt', '.jinja', '.py'}
    actual = {str(p.resolve()) for p in llm_dir.iterdir() if p.is_file() and p.suffix in load_extensions}
    pinned = {str(Path(p).resolve()) for p in manifest['llm_files_sha256']}
    require(actual == pinned, 'LLM directory file membership differs from frozen pins')
    return llm_dir


class Budget:
    def __init__(self):
        self.started = time.monotonic()
        self.model_started = None

    def check(self):
        current = time.monotonic()
        require(current - self.started <= 240 * 60, 'Overall 240-minute budget exhausted')
        if self.model_started is not None:
            require(current - self.model_started <= 45 * 60, 'Model 45-minute budget exhausted')


def stop_signal(signum, frame):
    raise RuntimeError('Runner interrupted by signal ' + str(signum))


def run(args):
    out = Path(args.out).resolve()
    require(read(out / 'status.json')['status'] == 'prepared', 'No resume: output is not prepared')
    require(sha(out / 'manifest.json') == os.environ.get('E5_MANIFEST_SHA256'), 'Launcher manifest identity differs')
    # Exclusive claim makes attempted concurrent runners fail before any model.
    with (out / 'run_claim.json').open('x') as claim:
        json.dump({'pid': os.getpid(), 'started_at': now(), 'manifest_sha256': sha(out / 'manifest.json')}, claim)
    budget = Budget()
    phase = 'preflight'
    state = {'seed': None, 'model_arm': None}
    previous_handlers = {s: signal.signal(s, stop_signal) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT, signal.SIGALRM)}
    signal.setitimer(signal.ITIMER_REAL, 240 * 60)

    def status(stage, **extra):
        write(out / 'status.json', {'status': stage, 'phase': phase, 'at': now(),
                                   'elapsed_s': time.monotonic() - budget.started, **state, **extra})

    try:
        status('running')
        manifest = read(out / 'manifest.json')
        llm_dir = verify_files(out, manifest, args.llm_dir)
        cfg = read(out / 'prereg.json')
        validate_training_config(cfg)
        require(out == (Path(cfg['root']) / cfg['output_directory']).resolve(), 'Output differs from frozen plan')
        require(manifest['parent_snapshot_sha256'] == cfg['parents'], 'Parent pins differ from plan')
        items, ordered, batches = lines(out / 'items.jsonl'), read(out / 'ordered_ids.json'), read(out / 'batches.json')
        pairs = np.load(out / 'pairs.npy', mmap_mode='r', allow_pickle=False)
        lookup = validate_population(items, ordered, batches, pairs)
        eval_sets = read(out / 'eval_sets.json')
        require(set(eval_sets['all_test']) == set(ordered['test']) and len(eval_sets['all_test']) == 1755,
                'Evaluation set differs from ordered test IDs')
        require(len(eval_sets['primary_same_prompt']) == len(set(eval_sets['primary_same_prompt'])) == 902, 'Primary support differs')
        require(set(eval_sets['primary_same_prompt']) <= set(ordered['test']), 'Primary includes non-test ID')
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from e5_scoring_v0 import transform_pair, score_run
        gpu_uuid = check_gpu_idle(out.parent)
        budget.check()
        preflight_seconds = time.monotonic() - budget.started
        # Pool preparation/queue are separate processes; this cap starts at
        # model work, after read-only preflight and before model allocation.
        budget = Budget()
        signal.setitimer(signal.ITIMER_REAL, 240 * 60)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(2)
        torch.set_default_dtype(torch.float32)
        phase = 'load_llm'
        status('running')
        dev = torch.device('cuda')
        torch.manual_seed(0)
        tok = AutoTokenizer.from_pretrained(str(llm_dir), local_files_only=True)
        llm = AutoModelForCausalLM.from_pretrained(str(llm_dir), dtype=torch.bfloat16, local_files_only=True).to(dev).eval()

        def finite_tensors(tensors, label):
            for tensor in tensors:
                require(tensor is not None and bool(torch.isfinite(tensor).all()), 'Nonfinite/missing ' + label)

        for parameter in llm.parameters():
            parameter.requires_grad_(False)
        finite_tensors(llm.parameters(), 'frozen LLM parameter')
        emb = llm.get_input_embeddings()
        hidden = emb.weight.shape[1]
        emb_rms = float(emb.weight.detach().float().pow(2).mean().sqrt())
        require(math.isfinite(emb_rms) and emb_rms > 0, 'Invalid embedding RMS')
        environment = {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
                       'cuda': torch.version.cuda, 'gpu_uuid': gpu_uuid, 'gpu_name': torch.cuda.get_device_name(0),
                       'manifest_sha256': sha(out / 'manifest.json'), 'llm_dir': str(llm_dir),
                       'hidden_size': hidden, 'embedding_rms': emb_rms,
                       'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                       'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                       'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                       'preflight_s': preflight_seconds,
                       'train_order_is_historical_e2_replay': False}
        write(out / 'runtime_environment.json', environment)
        phase = 'freeze_prompts'
        prompts = {}
        with (out / 'prompts.jsonl').open('x') as handle:
            for index, item in enumerate(items):
                budget.check()
                user = source_prompt(item)
                chat = tok.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True)
                require(chat.count('<EO>') == 1, 'Ambiguous EO placeholder')
                pre, post = chat.split('<EO>')
                ids = [tok(text, add_special_tokens=False, return_tensors='pt').input_ids[0]
                       for text in (pre, post, item['answer'] + tok.eos_token)]
                require(all(len(x) for x in ids), 'Empty prompt/answer tokenization')
                prompts[item['id']] = ids
                record = {'id': item['id'], 'pair_index': index, 'user_text': user, 'chat_text': chat,
                          'prefix_ids': ids[0].tolist(), 'suffix_ids': ids[1].tolist(),
                          'answer_ids': ids[2].tolist(), 'source_gold': item['answer'], 'n_eo_tokens': 192}
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())

        def build(proj, item_id, input_arm, with_answer):
            budget.check()
            index = lookup[item_id]
            values, types = transform_pair(pairs[index], input_arm)
            require(values.dtype == np.float32 and values.shape == (192, 768)
                    and np.array_equal(types, np.repeat([0, 1, 3], 64)), 'Transformed input contract differs')
            require(np.isfinite(values).all(), 'Nonfinite transformed source')
            values, types = torch.from_numpy(values).to(dev), torch.from_numpy(types).to(dev)
            projected = proj(values, types).to(torch.bfloat16)
            finite_tensors([projected], 'projected tokens')
            return assemble_sequence(torch, emb, prompts[item_id], projected, with_answer, dev)

        initial_root, model_root = out / 'initial_states', out / 'models'
        initial_root.mkdir()
        model_root.mkdir()
        outcomes = []
        row_count = 0
        phase = 'freeze_all_initial_states'
        initial_records = {}
        for seed in SEEDS:
            budget.check()
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            canonical = make_projector(torch, hidden, emb_rms)
            initial_state = {k: v.detach().cpu().clone() for k, v in canonical.state_dict().items()}
            finite_tensors(initial_state.values(), 'initial parameter')
            initial_file = initial_root / f'seed{seed}.pt'
            torch.save(initial_state, initial_file)
            initial_records[str(seed)] = {'seed': seed, 'file_sha256': sha(initial_file),
                'tensor_sha256': tensor_state_hash(initial_state),
                'parameter_count': sum(p.numel() for p in canonical.parameters())}
            write(initial_root / f'seed{seed}.json', initial_records[str(seed)])
            del canonical, initial_state
        write(initial_root / 'manifest.json', {'all_generated_before_any_training': True,
                                               'seeds': initial_records, 'at': now()})
        phase = 'train_models'
        with (out / 'predictions.jsonl').open('x') as all_predictions:
            for seed in SEEDS:
                initial_file = initial_root / f'seed{seed}.pt'
                initial_sha = initial_records[str(seed)]['file_sha256']
                initial_hash = initial_records[str(seed)]['tensor_sha256']
                require(sha(initial_file) == initial_sha, 'Canonical initial checkpoint changed')
                initial_state = torch.load(initial_file, map_location='cpu', weights_only=True)
                require(tensor_state_hash(initial_state) == initial_hash, 'Canonical initial tensor hash changed')
                for arm in ARMS:
                    state.update(seed=seed, model_arm=arm)
                    budget.model_started = time.monotonic()
                    # The signal enforces the tighter of remaining total and model time.
                    signal.setitimer(signal.ITIMER_REAL, min(45 * 60, max(.001, 240 * 60 - (time.monotonic() - budget.started))))
                    directory = model_root / f'seed{seed}_{arm}'
                    directory.mkdir()
                    shutil.copyfile(initial_file, directory / 'initial.pt')
                    require(sha(directory / 'initial.pt') == initial_sha, 'Initial file copy changed')
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    random.seed(seed)
                    proj = make_projector(torch, hidden, emb_rms).to(dev)
                    proj.load_state_dict({k: v.clone() for k, v in initial_state.items()}, strict=True)
                    require(tensor_state_hash(proj.state_dict()) == initial_hash, 'Initial parameter tensors differ')
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    random.seed(seed)
                    optimizer = torch.optim.AdamW(proj.parameters(), lr=1e-4, weight_decay=.01)
                    groups = [{k: v for k, v in group.items() if k != 'params'} for group in optimizer.param_groups]
                    write(directory / 'training_contract.json', {'seed': seed, 'model_arm': arm,
                          'initial_file_sha256': initial_sha, 'initial_tensor_sha256': initial_hash,
                          'ordered_ids_sha256': sha(out / 'ordered_ids.json'), 'batches_sha256': sha(out / 'batches.json'),
                          'optimizer_groups': groups, 'epochs': 3, 'batch_size': 8,
                          'expected_updates': 1590, 'expected_exposures': 12702})
                    proj.train()
                    require(not llm.training and all(not p.requires_grad for p in llm.parameters()), 'LLM freeze changed')
                    phase = 'training'
                    status('running', step=0, exposures=0)
                    training_start = time.monotonic()
                    updates, exposures, epoch_losses = 0, 0, []
                    with (directory / 'steps.jsonl').open('x') as step_log:
                        for epoch, epoch_batches in enumerate(batches[str(seed)]):
                            losses = []
                            for batch_index, batch_ids in enumerate(epoch_batches):
                                budget.check()
                                sequences = [build(proj, item_id, arm, True) for item_id in batch_ids]
                                length = max(len(e) for e, _ in sequences)
                                embeds = torch.zeros(len(sequences), length, hidden, dtype=torch.bfloat16, device=dev)
                                labels = torch.full((len(sequences), length), -100, dtype=torch.long, device=dev)
                                attention = torch.zeros(len(sequences), length, dtype=torch.long, device=dev)
                                for k, (values, gold) in enumerate(sequences):
                                    embeds[k, :len(values)] = values
                                    labels[k, :len(gold)] = gold
                                    attention[k, :len(values)] = 1
                                result = llm(inputs_embeds=embeds, attention_mask=attention, labels=labels)
                                finite_tensors([result.loss], 'training loss')
                                optimizer.zero_grad()
                                result.loss.backward()
                                finite_tensors((p.grad for p in proj.parameters()), 'projector gradient')
                                optimizer.step()
                                finite_tensors(proj.parameters(), 'updated projector parameter')
                                updates += 1
                                exposures += len(batch_ids)
                                loss = float(result.loss.detach())
                                losses.append(loss)
                                record = {'step': updates, 'epoch': epoch, 'batch_index': batch_index, 'ids': batch_ids,
                                          'batch_size': len(batch_ids), 'exposures': exposures, 'loss': loss,
                                          'elapsed_s': time.monotonic() - training_start,
                                          'finite_loss_grad_parameters': True}
                                step_log.write(json.dumps(record, allow_nan=False) + '\n')
                                step_log.flush()
                                del result, embeds, labels, attention, sequences
                                if updates % 25 == 0:
                                    status('running', step=updates, exposures=exposures)
                                    print(f'E5 seed={seed} arm={arm} step={updates}/1590 loss={loss:.6f}', flush=True)
                            epoch_losses.append(float(np.mean(losses)))
                        os.fsync(step_log.fileno())
                    require((updates, exposures) == (1590, 12702), 'Training exposure/update budget mismatch')
                    train_seconds = time.monotonic() - training_start
                    optimizer_steps = [int(v['step'].item()) for v in optimizer.state.values() if 'step' in v]
                    require(len(optimizer_steps) == len(list(proj.parameters())) and set(optimizer_steps) == {1590}, 'Optimizer step audit failed')
                    final_state = {k: v.detach().cpu().clone() for k, v in proj.state_dict().items()}
                    finite_tensors(final_state.values(), 'final checkpoint')
                    final_tensor_hash = tensor_state_hash(final_state)
                    torch.save(final_state, directory / 'projector.pt')
                    checkpoint_hash = sha(directory / 'projector.pt')
                    loaded = torch.load(directory / 'projector.pt', map_location='cpu', weights_only=True)
                    require(tensor_state_hash(loaded) == final_tensor_hash, 'Saved final checkpoint differs')
                    proj.load_state_dict(loaded, strict=True)
                    proj.eval()
                    del loaded, final_state, optimizer
                    training_record = {'seed': seed, 'model_arm': arm, 'updates': updates, 'exposures': exposures,
                                       'epoch_mean_batch_loss': epoch_losses, 'train_s': train_seconds,
                                       'checkpoint_sha256': checkpoint_hash, 'checkpoint_tensor_sha256': final_tensor_hash,
                                       'steps_sha256': sha(directory / 'steps.jsonl'),
                                       'initial_file_sha256': initial_sha, 'initial_tensor_sha256': initial_hash}
                    write(directory / 'training_completed.json', training_record)
                    phase = 'evaluation'
                    eval_start = time.monotonic()
                    parse_stats = {}
                    evaluation_arms = ['native', 'full_no_delta'] if arm == 'full' else ['native']
                    for eval_arm in evaluation_arms:
                        input_arm = 'pair' if eval_arm == 'full_no_delta' else arm
                        counts = collections.Counter()
                        status('running', eval_arm=eval_arm, generated=0)
                        with (directory / f'answers_{eval_arm}.jsonl').open('x') as answers, torch.no_grad():
                            for generated, item_id in enumerate(ordered['test'], 1):
                                budget.check()
                                index = lookup[item_id]
                                item = items[index]
                                embedded, _ = build(proj, item_id, input_arm, False)
                                generation = llm.generate(inputs_embeds=embedded[None], attention_mask=torch.ones(1, len(embedded), dtype=torch.long, device=dev),
                                    max_new_tokens=16, do_sample=False, pad_token_id=tok.eos_token_id,
                                    return_dict_in_generate=True, output_scores=True)
                                require(len(generation.scores) > 0, 'No generation scores')
                                for scores in generation.scores:
                                    require(not bool(torch.isnan(scores).any()) and not bool(torch.isposinf(scores).any())
                                            and bool(torch.isfinite(scores).any(dim=-1).all()), 'Invalid generation logits')
                                raw = tok.decode(generation.sequences[0], skip_special_tokens=True)
                                row = output_row(seed, arm, eval_arm, item, index, raw)
                                text = json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
                                answers.write(text)
                                answers.flush()
                                all_predictions.write(text)
                                all_predictions.flush()
                                row_count += 1
                                counts[(item['phen'], 'n')] += 1
                                counts[(item['phen'], 'invalid')] += row['parsed'] is None
                                if generated % 100 == 0:
                                    status('running', eval_arm=eval_arm, generated=generated)
                                del generation, embedded
                            os.fsync(answers.fileno())
                        parse_stats[eval_arm] = {}
                        for phen in PHEN:
                            n, failed = counts[(phen, 'n')], counts[(phen, 'invalid')]
                            require(n == (1371 if phen == 'flood' else 384), 'Evaluation phenomenon coverage')
                            parse_stats[eval_arm][phen] = {'n': n, 'unparsed': failed, 'rate': failed / n}
                        write(directory / 'parse_audit.json', parse_stats)
                        require(all(x['rate'] <= .01 for x in parse_stats[eval_arm].values()), 'Parse failure exceeds 1 percent')
                    require(tensor_state_hash(proj.state_dict()) == final_tensor_hash, 'Evaluation changed projector')
                    require(sha(directory / 'projector.pt') == checkpoint_hash, 'Checkpoint changed during evaluation')
                    outcome = {**training_record, 'eval_s': time.monotonic() - eval_start, 'parse': parse_stats,
                               'model_elapsed_s': time.monotonic() - budget.model_started,
                               'answers_sha256': {e: sha(directory / f'answers_{e}.jsonl') for e in evaluation_arms}}
                    write(directory / 'completed.json', outcome)
                    outcomes.append(outcome)
                    write(out / 'partial_models.json', outcomes)
                    del proj
                    torch.cuda.empty_cache()
                    budget.check()
                    budget.model_started = None
                    signal.setitimer(signal.ITIMER_REAL, max(.001, 240 * 60 - (time.monotonic() - budget.started)))
                del initial_state
            os.fsync(all_predictions.fileno())
        require(len(outcomes) == 12 and row_count == 26325, 'Final model/answer coverage mismatch')
        phase = 'final_integrity'
        verify_files(out, manifest, args.llm_dir)
        require(sha(out / 'manifest.json') == os.environ['E5_MANIFEST_SHA256'], 'Manifest changed during execution')
        budget.check()
        for outcome in outcomes:
            directory = model_root / f"seed{outcome['seed']}_{outcome['model_arm']}"
            require(sha(directory / 'projector.pt') == outcome['checkpoint_sha256'], 'Finished checkpoint changed')
            require(sha(directory / 'initial.pt') == outcome['initial_file_sha256'], 'Finished initial state changed')
            require(sha(directory / 'steps.jsonl') == outcome['steps_sha256'], 'Finished training log changed')
            for eval_arm, expected in outcome['answers_sha256'].items():
                require(sha(directory / f'answers_{eval_arm}.jsonl') == expected, 'Finished answer file changed')
        write(out / 'inference_completed.json', {'schema': 'e5-training-inference-completed-v0',
              'n_models': 12, 'n_rows': row_count, 'n_prompts': 5989, 'models': outcomes,
              'predictions_sha256': sha(out / 'predictions.jsonl'), 'prompts_sha256': sha(out / 'prompts.jsonl'),
              'runtime_sha256': sha(out / 'runtime_environment.json'), 'elapsed_s': time.monotonic() - budget.started,
              'scientific_scoring_pending': True})
        status('inference_completed', n_models=12, n_rows=row_count)
        phase = 'scoring'
        training_summary = {'schema': 'e5-training-summary-v0', 'n_models': 12,
                            'updates_per_model': 1590, 'exposures_per_model': 12702,
                            'models': outcomes, 'manifest_sha256': sha(out / 'manifest.json'),
                            'code_snapshot_sha256': manifest['code_snapshot_sha256'],
                            'initial_manifest_sha256': sha(initial_root / 'manifest.json'),
                            'prompts_sha256': sha(out / 'prompts.jsonl'),
                            'predictions_sha256': sha(out / 'predictions.jsonl'),
                            'runtime_sha256': sha(out / 'runtime_environment.json')}
        write(out / 'training_summary.json', training_summary)
        scores = score_run(lines(out / 'predictions.jsonl'), items, eval_sets)
        scores.update(manifest_sha256=sha(out / 'manifest.json'),
                      training_summary_sha256=sha(out / 'training_summary.json'),
                      predictions_sha256=training_summary['predictions_sha256'],
                      prompts_sha256=training_summary['prompts_sha256'],
                      elapsed_s=time.monotonic() - budget.started)
        write(out / 'scores.json', scores)
        require(scores.get('valid') is True, 'Scientific scoring invalid: ' + str(scores.get('invalid_reason', scores.get('invalid_reasons'))))
        budget.check()
        status('completed', n_models=12, n_rows=row_count, verdict=scores.get('verdict'))
        print('E5 COMPLETE ' + str(scores.get('verdict')), flush=True)
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        failure = {'schema': 'e5-run-failure-v0', 'at': now(), 'phase': phase, **state,
                   'exception_type': type(error).__name__, 'reason': str(error),
                   'traceback': traceback.format_exc(), 'elapsed_s': time.monotonic() - budget.started,
                   'resume_allowed': False, 'completed_artifacts_preserved': True}
        write(out / 'failure.json', failure)
        status('invalid', reason=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--llm-dir')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
