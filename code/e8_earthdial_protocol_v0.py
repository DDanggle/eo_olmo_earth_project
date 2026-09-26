#!/usr/bin/env python3
"""E8: does a public EO-VLM (EarthDial, CVPR 2025) answer the landslide question from the images?

Zero-shot, no training. Same content-use controls as E0 (code/earthtalk_content_controls_v0.py),
applied to EarthDial_4B_RGB (optionally _MS) on sentinel_qa_v0_1 Q1 items (Sentinel-2 RGB PNG pairs).
Input contract and sources: docs/E8_EARTHDIAL_INPUT_CONTRACT_20260926.md.

Arms (each paired tile has one 'yes' item spanning the landslide and one later 'no' item):
  real              the item's two images, the item's prompt (dates)
  swap_within_tile  the item's prompt (dates), the IMAGES of the other item on the same tile
  blank             two mid-grey images, the item's prompt
  post_only         the item's second image only, one-image prompt
  pre_only          the item's first image only, one-image prompt

  python3 -B code/e8_earthdial_protocol_v0.py --selftest                       # CPU, no torch
  CUDA_VISIBLE_DEVICES=1 env -u PYTHONPATH .venv-master/bin/python -B code/e8_earthdial_protocol_v0.py --probe
  CUDA_VISIBLE_DEVICES=1 env -u PYTHONPATH .venv-master/bin/python -B code/e8_earthdial_protocol_v0.py --out e8_earthdial_protocol_v0
"""
import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from earthtalk_content_controls_v0 import balanced_acc, bootstrap, d_stat, parse, within_tile_pairs  # noqa: E402

ROOT = Path("/home/work/data/olmoearth")
ITEMS = ROOT / "sentinel_qa_v0_1/items.jsonl"
MODELS = {"RGB": ROOT / "models/EarthDial_4B_RGB", "MS": ROOT / "models/EarthDial_4B_MS"}
EARTHDIAL_SRC = ROOT / "third_party/EarthDial/src"   # git clone https://github.com/hiyamdebary/EarthDial (92c1260)
ARMS = ("real", "swap_within_tile", "blank", "post_only", "pre_only")
BLANK = "<blank>"          # stands for a 512x512 mid-grey (128,128,128) image
GREY = (128, 128, 128)
PROBE_TILES = 4
MAX_NEW_TOKENS = 16

# EarthDial's own bi-temporal template, verbatim from its change-detection eval questions
# (src/earthdial/eval/rs_change_detection/results/*.jsonl): "[changedet] [hr_rgb_temp_0.5] <image> <image> \n <question>".
# The "\n" there is the two characters backslash + n, not a newline; we keep it verbatim.
# [hr_rgb_temp_0.5] is the only temporal sensor tag in the tokenizer; there is no Sentinel-2 temporal tag.
# Dates are added to the question so that the swap arm can put text (dates) and images in conflict.
LIT_NL = "\\n"
TEMPLATES = {
    "earthdial": {
        2: "[changedet] {tag2} <image> <image> " + LIT_NL + " Did a landslide occur between the first image "
           "(taken on {d1}) and the second image (taken on {d2})? Please answer yes or no.",
        1: "{tag1} <image> " + LIT_NL + " Is there a landslide visible in this image? Please answer yes or no.",
    },
    "plain": {
        2: "Image-1 ({d1}): <image>\nImage-2 ({d2}): <image>\n"
           "Did a landslide occur between these two Sentinel-2 images? Answer yes or no.",
        1: "<image>\nIs there a landslide visible in this image? Answer yes or no.",
    },
}
SENSOR_TAGS = {"hr_temp": ("[hr_rgb_temp_0.5]", "[s2_rgb_10]"),   # (two-image tag, one-image tag)
               "s2": ("[s2_rgb_10]", "[s2_rgb_10]")}


# ---------------------------------------------------------------- pure logic (tested by --selftest)

def question(template, sensor, n_images, dates):
    tag2, tag1 = SENSOR_TAGS[sensor]
    q = TEMPLATES[template][n_images].format(tag2=tag2, tag1=tag1, d1=dates[0], d2=dates[-1])
    assert q.count("<image>") == n_images, q
    return q


def plan_arms(q1):
    """Return ({arm: [job]}, pairs). job = dict(src, images, n_images, emb_gold, emb_item)."""
    pairs = within_tile_pairs(q1)
    jobs = {a: [] for a in ARMS}
    for t in sorted(pairs):
        pos, neg = pairs[t]
        for it, other in ((pos, neg), (neg, pos)):
            jobs["real"].append(dict(src=it, images=list(it["images"]), n_images=2, emb_gold=it["answer"], emb_item=it["id"]))
            jobs["swap_within_tile"].append(dict(src=it, images=list(other["images"]), n_images=2,
                                                 emb_gold=other["answer"], emb_item=other["id"]))
            jobs["blank"].append(dict(src=it, images=[BLANK, BLANK], n_images=2, emb_gold=None, emb_item=None))
            jobs["post_only"].append(dict(src=it, images=[it["images"][1]], n_images=1, emb_gold=it["answer"], emb_item=it["id"]))
            jobs["pre_only"].append(dict(src=it, images=[it["images"][0]], n_images=1, emb_gold=it["answer"], emb_item=it["id"]))
    return jobs, pairs


def select_items(items, probe=False):
    q1 = [x for x in items if x["type"] == "Q1"]
    if probe:
        keep = sorted(within_tile_pairs(q1))[:PROBE_TILES]
        q1 = [x for x in q1 if x["tile"] in keep]
    return q1


def row(arm, job, raw):
    s = job["src"]
    return {"id": s["id"], "tile": s["tile"], "fold": s.get("fold"), "arm": arm, "emb_item": job["emb_item"],
            "images": job["images"], "text_gold": s["answer"], "emb_gold": job["emb_gold"],
            "parsed": parse("Q1", raw), "raw": raw}


def bootstrap_diff(rows_a, rows_b, by_a, by_b, n=2000, seed=20260925):
    """Tile bootstrap of d(rows_a) - d(rows_b), resampling the same tiles for both arms."""
    ta, tb = defaultdict(list), defaultdict(list)
    for r in rows_a:
        ta[r["tile"]].append(r)
    for r in rows_b:
        tb[r["tile"]].append(r)
    tiles = sorted(set(ta) & set(tb))
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        pick = [rng.choice(tiles) for _ in tiles]
        a = d_stat([r for t in pick for r in ta[t]], by_a)
        b = d_stat([r for t in pick for r in tb[t]], by_b)
        if a is not None and b is not None:
            vals.append(a - b)
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]]


def verdict(s):
    """Provisional, same thresholds as E0; freeze in a prereg before the full run."""
    if s["validity"]["max_parse_fail"] > 0.10:
        return "invalid_run"
    if s["d_real"]["value"] is None or s["d_real"]["value"] < 0.10:
        return "uninformative_low_signal"
    d, (lo, hi) = s["d_swap"]["value"], s["d_swap"]["ci95"]
    if d >= 0.10 and lo > 0:
        return "reads_images"
    if d <= -0.10 and hi < 0:
        return "follows_text"
    return "not_demonstrated"


def summarize(rows_by_arm):
    s = {"n": {a: len(r) for a, r in rows_by_arm.items()}}
    parse_fail = {a: sum(r["parsed"] is None for r in rows) / max(len(rows), 1) for a, rows in rows_by_arm.items()}
    s["validity"] = {"parse_fail": parse_fail, "max_parse_fail": max(parse_fail.values())}
    s["p_yes"] = {a: sum(r["parsed"] == "yes" for r in rows) / max(len(rows), 1) for a, rows in rows_by_arm.items()}
    s["balanced_acc"] = {a: {"vs_prompt_gold": balanced_acc(rows, "text_gold"),
                             "vs_image_gold": balanced_acc(rows, "emb_gold")} for a, rows in rows_by_arm.items()}
    # d = P(yes | gold yes) - P(yes | gold no). 'emb' = gold of the item whose images were shown.
    for name, arm, by in (("d_real", "real", "emb"), ("d_swap", "swap_within_tile", "emb"),
                          ("d_blank", "blank", "text"), ("d_post", "post_only", "emb"), ("d_pre", "pre_only", "emb")):
        rows = rows_by_arm.get(arm, [])
        v = d_stat(rows, by) if rows else None
        s[name] = {"value": v, "by": by, "ci95": bootstrap(rows, by) if v is not None else None}
    # descriptive: is the two-image answer more than a single-(post)-image scar detector?
    if s["d_real"]["value"] is not None and s["d_post"]["value"] is not None:
        s["d_real_minus_d_post"] = {"value": s["d_real"]["value"] - s["d_post"]["value"],
                                    "ci95": bootstrap_diff(rows_by_arm["real"], rows_by_arm["post_only"], "emb", "emb")}
    s["verdict"] = verdict(s)
    return s


def selftest():
    items = []
    for t in range(30):
        fold = "holdout_a" if t < 15 else "holdout_b"
        a, b, c, d = "2018-01-01", "2018-03-01", "2018-05-01", "2018-07-01"    # landslide between a and b
        img = lambda day: f"img/t{t}_{day}.png"
        items += [{"id": f"t{t}_q1_pos", "tile": f"t{t}", "fold": fold, "type": "Q1", "dates": [a, b], "images": [img(a), img(b)], "answer": "yes"},
                  {"id": f"t{t}_q1_neg", "tile": f"t{t}", "fold": fold, "type": "Q1", "dates": [c, d], "images": [img(c), img(d)], "answer": "no"},
                  {"id": f"t{t}_q3", "tile": f"t{t}", "fold": fold, "type": "Q3", "dates": [a, b], "images": [img(a), img(b)], "answer": ["NW"]}]
    items.append({"id": "n0_q1_zero", "tile": "n0", "fold": "holdout_a", "type": "Q1", "dates": ["x", "y"], "images": ["p", "q"], "answer": "no"})
    q1 = select_items(items)
    assert all(x["type"] == "Q1" for x in q1)
    jobs, pairs = plan_arms(q1)
    assert len(pairs) == 30 and "n0" not in pairs, "tiles without both answers are dropped"
    assert all(len(jobs[a]) == 60 for a in ARMS)
    by_id = {x["id"]: x for x in q1}
    for j in jobs["swap_within_tile"]:
        other = by_id[j["emb_item"]]
        assert other["tile"] == j["src"]["tile"] and other["id"] != j["src"]["id"]
        assert j["images"] == other["images"] and j["emb_gold"] == other["answer"] != j["src"]["answer"]
    assert all(j["images"] == [BLANK, BLANK] and j["emb_gold"] is None for j in jobs["blank"])
    assert all(j["images"] == [j["src"]["images"][1]] for j in jobs["post_only"])
    assert all(j["images"] == [j["src"]["images"][0]] for j in jobs["pre_only"])
    assert len(select_items(items, probe=True)) == 2 * PROBE_TILES
    for tpl in TEMPLATES:
        for sensor in SENSOR_TAGS:
            q2 = question(tpl, sensor, 2, ["2018-01-01", "2018-03-01"])
            assert "2018-01-01" in q2 and "2018-03-01" in q2 and q2.count("<image>") == 2
            assert question(tpl, sensor, 1, ["2018-03-01"]).count("<image>") == 1
    assert question("earthdial", "hr_temp", 2, ["a", "b"]).startswith("[changedet] [hr_rgb_temp_0.5] <image> <image> \\n ")

    # simulated models; the 'image' is identified by its path, which carries the date
    event = "2018-02-01"
    scar = lambda p: p != BLANK and p.split("_")[-1][:-4] > event        # landslide scar visible after the event
    def image_change_reader(job):     # two images: yes iff scar appears between them; one image: scar visible
        im = job["images"]
        if BLANK in im:
            return "No."
        return ("Yes." if scar(im[1]) and not scar(im[0]) else "No.") if len(im) == 2 else ("Yes" if scar(im[0]) else "No")
    def post_scar_reader(job):        # ignores the first image entirely
        return "yes" if scar(job["images"][-1]) else "no"
    texter = lambda job: "Yes" if job["src"]["dates"][0] < event else "No"   # reads only the dates in the prompt
    run_all = lambda fn: {a: [row(a, j, fn(j)) for j in jobs[a]] for a in ARMS}

    s = summarize(run_all(image_change_reader))
    assert s["verdict"] == "reads_images", s["verdict"]
    assert s["d_real"]["value"] == 1.0 and s["d_swap"]["value"] == 1.0 and s["d_blank"]["value"] == 0.0
    assert s["d_post"]["value"] == 0.0 and s["d_pre"]["value"] == -1.0, (s["d_post"], s["d_pre"])
    assert s["d_real_minus_d_post"]["value"] == 1.0 and s["d_real_minus_d_post"]["ci95"][0] > 0
    assert s["balanced_acc"]["swap_within_tile"] == {"vs_prompt_gold": 0.0, "vs_image_gold": 1.0}
    assert s["balanced_acc"]["blank"]["vs_image_gold"] is None
    s = summarize(run_all(post_scar_reader))
    assert s["d_real"]["value"] == 0.0 and s["verdict"] == "uninformative_low_signal", "post-only scar gives no Q1 signal"
    s = summarize(run_all(texter))
    assert s["verdict"] == "follows_text" and s["d_blank"]["value"] == 1.0, s["verdict"]
    s = summarize(run_all(lambda j: "no"))
    assert s["verdict"] == "uninformative_low_signal" and s["p_yes"]["real"] == 0.0
    s = summarize(run_all(lambda j: "I cannot tell."))
    assert s["verdict"] == "invalid_run" and s["validity"]["max_parse_fail"] == 1.0
    assert parse("Q1", "Yes, a landslide occurred.") == "yes" and parse("Q1", "Nope") is None
    print("selftest ok: Q1 pairing, 5 arms (swap/blank/post/pre construction), templates, "
          "d/CI incl. d_real-d_post, verdict branches (reads/follows/low-signal/invalid)")


# ---------------------------------------------------------------- model run (GPU)

def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    """Copy of EarthDial src/earthdial/train/dataset.py:770-823 (inlined to avoid its decord/cv2 imports)."""
    w, h = image.size
    aspect = w / h
    ratios = sorted({(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
                     if min_num <= i * j <= max_num}, key=lambda x: x[0] * x[1])
    best, best_diff = (1, 1), float("inf")
    for r in ratios:
        diff = abs(aspect - r[0] / r[1])
        if diff < best_diff:
            best_diff, best = diff, r
        elif diff == best_diff and w * h > 0.5 * image_size * image_size * r[0] * r[1]:
            best = r
    tw, th, blocks = image_size * best[0], image_size * best[1], best[0] * best[1]
    img = image.resize((tw, th))
    cols = tw // image_size
    out = [img.crop(((i % cols) * image_size, (i // cols) * image_size,
                     (i % cols + 1) * image_size, (i // cols + 1) * image_size)) for i in range(blocks)]
    if use_thumbnail and len(out) != 1:
        out.append(image.resize((image_size, image_size)))
    return out


def run(a):
    import torch
    import torchvision.transforms as T
    import transformers
    from PIL import Image
    from torchvision.transforms.functional import InterpolationMode

    model_dir = Path(a.model_dir) if a.model_dir else MODELS[a.ckpt]
    sys.path.insert(0, str(a.earthdial_src))
    from earthdial.model.internvl_chat import InternVLChatModel
    from transformers import AutoTokenizer

    out_dir = ROOT / (a.out or ("e8_earthdial_protocol_v0" + ("_probe" if a.probe else "")))
    out_dir.mkdir(parents=True, exist_ok=False)
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().cuda()
    cfg = model.config
    image_size = cfg.force_image_size or cfg.vision_config.image_size
    # eval transform for RGB: build_transform(is_train=False, normalize_type='imagenet'), dataset.py:330-336
    transform = T.Compose([T.Lambda(lambda im: im.convert("RGB")),
                           T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                           T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    def load(p):
        return Image.new("RGB", (512, 512), GREY) if p == BLANK else Image.open(p).convert("RGB")

    def answer(job, q):
        # one entry of num_patches_list per image, as in training (dataloader.py:318-386, dataset.py:602-604);
        # EarthDial's own CD eval omits it, which puts all image tokens at the first <image>.
        tiles, counts = [], []
        max_num = max(1, cfg.max_dynamic_patch // job["n_images"])     # dataloader.py:343
        for p in job["images"]:
            t = dynamic_preprocess(load(p), max_num=max_num, image_size=image_size, use_thumbnail=cfg.use_thumbnail)
            tiles += t
            counts.append(len(t))
        pv = torch.stack([transform(t) for t in tiles]).to(torch.bfloat16).cuda()
        gen = dict(num_beams=1, max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=1, do_sample=False)
        return model.chat(tokenizer=tok, pixel_values=pv, question=q, generation_config=gen,
                          num_patches_list=counts, verbose=False), counts

    items = [json.loads(l) for l in ITEMS.read_text().splitlines() if l.strip()]
    q1 = select_items(items, probe=a.probe)
    jobs, pairs = plan_arms(q1)
    arms = [x for x in a.arms.split(",") if x] if a.arms else list(ARMS)
    head = EARTHDIAL_SRC.parent / ".git/HEAD"
    manifest = {"experiment": "E8", "ckpt": a.ckpt, "model_dir": str(model_dir),
                "model_config_sha256": sha256(model_dir / "config.json"),
                "model_index_sha256": sha256(model_dir / "model.safetensors.index.json"),
                "earthdial_src": str(a.earthdial_src), "earthdial_head": head.read_text().strip() if head.exists() else None,
                "code_sha256": sha256(__file__), "controls_sha256": sha256(Path(__file__).resolve().parent / "earthtalk_content_controls_v0.py"),
                "transformers": transformers.__version__, "torch": torch.__version__,
                "template": a.template, "sensor": a.sensor, "image_size": image_size,
                "use_thumbnail": cfg.use_thumbnail, "max_dynamic_patch": cfg.max_dynamic_patch, "llm_template": cfg.template,
                "question_2img_example": question(a.template, a.sensor, 2, ["<d1>", "<d2>"]),
                "question_1img": question(a.template, a.sensor, 1, ["<d>"]),
                "probe": a.probe, "n_q1_items": len(q1), "n_pairs": len(pairs), "arms": arms,
                "jobs": {k: len(jobs[k]) for k in arms}, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if not transformers.__version__.startswith("4.37"):
        manifest["warning"] = "EarthDial pins transformers==4.37.2; bundled Phi-3 code is untested on this version"
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1), flush=True)

    rows_by_arm = {}
    t0 = time.perf_counter()
    with torch.no_grad():
        for arm in arms:
            rows = []
            for job in jobs[arm]:
                q = question(a.template, a.sensor, job["n_images"], job["src"]["dates"])
                raw, counts = answer(job, q)
                r = row(arm, job, raw)
                r.update(question=q, num_patches=counts)
                rows.append(r)
            rows_by_arm[arm] = rows
            (out_dir / f"answers_{arm}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            print(f"{arm}: {len(rows)} done {time.perf_counter() - t0:.0f}s  e.g. {rows[0]['raw']!r}", flush=True)
    if set(arms) == set(ARMS):
        scores = summarize(rows_by_arm)
        scores["manifest"] = manifest
        scores["elapsed_s"] = time.perf_counter() - t0
        (out_dir / "scores.json").write_text(json.dumps(scores, indent=1))
        print(json.dumps({k: scores[k] for k in ("verdict", "p_yes", "d_real", "d_swap", "d_blank", "d_post", "d_pre",
                                                 "d_real_minus_d_post", "validity")}, indent=1))
    print("E8 DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--probe", action="store_true", help=f"first {PROBE_TILES} paired tiles only")
    ap.add_argument("--ckpt", choices=sorted(MODELS), default="RGB")
    ap.add_argument("--model_dir", default=None, help="override the checkpoint directory")
    ap.add_argument("--earthdial_src", default=str(EARTHDIAL_SRC), help="EarthDial repo src/ (HF repos ship no modeling code)")
    ap.add_argument("--template", choices=sorted(TEMPLATES), default="earthdial")
    ap.add_argument("--sensor", choices=sorted(SENSOR_TAGS), default="hr_temp")
    ap.add_argument("--arms", default="", help="comma list; default all (scores.json only when all arms run)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    selftest() if args.selftest else run(args)
