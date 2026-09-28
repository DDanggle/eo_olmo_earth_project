"""Training-only named-support context projection; no query gold or model execution."""
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from context_reference import support_annotation

CATALOG_SHA = "30fe53cb935002c464c9d38b42d01e20cb724e70af8fcd2dbefddbe37bd7a377"
OBJECTS_SHA = "c8b6ce724eaf3acc1e1f159ab2ec46943bb88eb49cde1aa0172e12181da53e11"
PROMPT = ("Find regions matching the positive examples and exclude the counterexamples. "
          "Return a mask and the acquired observation dates supporting it.")
CONDITIONS = ("generic", "names_only", "matched_knowledge", "removed_knowledge_diagnostic", "swapped_knowledge_diagnostic")

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def dump(p,v): Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+"\n")

def render(note, cards, condition):
    if condition not in CONDITIONS: raise ValueError("unknown condition")
    if condition == "generic":
        return {"instruction": PROMPT, "positive": "The target is defined by the positive support examples.",
                "counterexample": "The confusing alternative is defined by the counterexample support examples."}
    roles=("positive","counterexample")
    selected=[cards[note["role_card_ids"][r]] for r in roles]
    texts=[c["concept_name"] for c in selected]
    if condition in ("matched_knowledge","swapped_knowledge_diagnostic"):
        facts=selected if condition=="matched_knowledge" else selected[::-1]
        texts=[name+". Background: "+c["claim_text"]+" Scope: "+c["transfer_limit"] for name,c in zip(texts,facts)]
    return {"instruction":PROMPT, **dict(zip(roles,texts))}

def prepare(episodes, objects, kc):
    if kc.get("expert_reviewed") is not False or kc.get("patch_level_expert_annotations") != 0:
        raise ValueError("This draft requires explicit nonexpert provenance")
    cards={c["class_id"]:c for c in kc["cards"]}
    if set(cards)!={8,14} or len(cards)!=len(kc["cards"]): raise ValueError("two unique expansion classes required")
    sourceids={s["id"] for s in kc["sources"]}
    for c in cards.values():
        if not c["source_ids"] or not set(c["source_ids"])<=sourceids: raise ValueError("missing source")
        for k in ("concept_name","claim_text","transfer_limit"):
            if not isinstance(c[k],str) or not c[k].strip() or "<|" in c[k]: raise ValueError("invalid plain-text card")
    byid={c["card_id"]:c for c in cards.values()}
    if len(byid)!=2: raise ValueError("duplicate card id")
    omap={o["object_key"]:o for o in objects if o["episode_role"]=="train_pool"}
    records=[]
    for e in episodes:
        n=support_annotation(e,omap,cards)
        # A fixed projection prevents audit IDs, source paths and query labels entering the model.
        contexts={c:render(n,byid,c) for c in CONDITIONS}
        assert contexts["removed_knowledge_diagnostic"]==contexts["names_only"]
        for ctx in contexts.values():
            assert set(ctx)=={"instruction","positive","counterexample"}
            text=json.dumps(ctx)
            assert n["episode_id"] not in text and n["query_patch_id_audit_only"] not in text
            assert all(key not in text for ks in n["support_object_lineage_audit_only"].values() for key in ks)
        records.append({"episode_id":n["episode_id"],"audit_only":n,"model_context_by_condition":contexts})
    return records

def run(a):
    if sha(a.catalog)!=CATALOG_SHA or sha(a.objects)!=OBJECTS_SHA: raise ValueError("pinned source mismatch")
    kc=json.loads(a.cards.read_text());records=prepare(rows(a.catalog),rows(a.objects),kc)
    assert len(records)==384 and len({r["episode_id"] for r in records})==384
    assert len({r["audit_only"]["query_patch_id_audit_only"] for r in records})==48
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/"contexts.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in records))
    dump(a.out/"knowledge_cards.json",kc)
    dump(a.out/"example_contexts.json",records[0]["model_context_by_condition"])
    receipt={"created_utc":datetime.now(timezone.utc).isoformat(),"status":"train_contexts_prepared_no_model_execution",
      "opened_input_hashes":{str(p):sha(p) for p in (a.catalog,a.objects,a.cards)},"code_sha256":sha(__file__),
      "reference_code_sha256":sha(Path(__file__).with_name("context_reference.py")),
      "episodes":384,"unique_train_queries":48,"conditions":list(CONDITIONS),"trained_conditions":[],
      "core_contrast":"matched_knowledge minus names_only with the same future architecture, images, masks and class names",
      "query_gold_opened":False,"raw_image_files_opened":0,"dev_or_heldout_opened":False,"expert_responses":0,"new_gpu_seconds":0,
      "old_p2_task_replaced":False,"actual_qwen_token_counts_verified":False,"model_gradient_verified":False,
      "model_input_projection":"Only model_context_by_condition[condition], exactly instruction/positive/counterexample; never serialize audit_only into model input",
      "diagnostic_limits":"Removed is exactly names_only; swapped keeps names fixed but may introduce obvious contradictions and does not prove agronomic reasoning.",
      "task_limits":"Names and sourced summaries are additional public support-label supervision, not human corrections, image evidence or a novel method.",
      "next":"Frozen real tokenizer and separate text-mask dependency tests; full actual EO/Qwen gradient pilot only after P2 ends and is audited."}
    dump(a.out/"preparation_receipt.json",receipt)
    dump(a.out/"export_manifest.json",{"files":[{"path":p.name,"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(a.out.iterdir()) if p.is_file()]})
    print(json.dumps(receipt))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("catalog","objects","cards","out"): p.add_argument("--"+name,type=Path,required=True)
    run(p.parse_args())
