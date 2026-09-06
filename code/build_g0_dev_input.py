"""Rebuild the G0 development input from raw few-shot reports: per-seed rows (not a macro placeholder), measured adaptation cost (gpu_s, raw_bytes_read summed over the 8 query regions), labels consumed, and FIXED anchors that pre-date this analysis (lower = full-data raw P2 'raw_strong', upper = full-data cache P4). Still a development smoke input (2 tasks already seen), not a G0 result."""
import json, numpy as np
from collections import defaultdict
from pathlib import Path
SRC={"sen12_landslide":("artifacts/fewshot_confirmatory/fu/report_fixed_update.json",0.196558375,0.27216575),
     "solar_farm":("artifacts/task2_fewshot/fu/report.json",0.33252233333333336,0.5931661666666667)}
ACT={"A0":"CACHED_HEAD","A1":"HEAD_ADAPT","A4w":"RAW_FINETUNE"}; rows=[]
for task,(p,lo,hi) in SRC.items():
    r=json.load(open(p)); agg=defaultdict(list); cost=defaultdict(lambda:{"gpu":0.0,"bytes":0,"labels":0,"n":0})
    for x in r["runs"]:
        if x["arm"] not in ACT or (x["arm"]!="A0" and x["K"]!=5): continue
        k=(x["arm"],x["seed"]); agg[k].append(x["eval"]["iou_fp_matched"]); c=cost[k]; c["gpu"]+=x["train"].get("gpu_s",0); c["bytes"]+=x["train"].get("raw_bytes_read",0); c["labels"]+=(0 if x["arm"]=="A0" else 5); c["n"]+=1
    for (arm,seed),v in sorted(agg.items()):
        c=cost[(arm,seed)]; rows.append({"episode_id":task,"task":task,"action":ACT[arm],"seed":f"s{seed}","score":float(np.mean(v)),"higher_is_better":True,"gpu_seconds":round(c["gpu"],2),"raw_bytes":int(c["bytes"]),"cache_bytes":0,"support_label_count":c["labels"],"lower_anchor":lo,"upper_anchor":hi,"n_regions":c["n"]})
out={"_note":"development smoke input v2 (2 already-seen tasks); per-seed region-macro FP-matched IoU at K=5 (A0 has no K); costs are MEASURED adaptation-time gpu_s and raw bytes read summed over the 8 query regions (source-head training and cache extraction excluded, reported separately); labels = 5 per region x 8 regions; anchors fixed from M65/MS-98 full-data P2 (lower) and P4 (upper), which pre-date this analysis",
     "excluded_actions":"REEMBED (true re-embedding) still unmeasured; PEFT_REEMBED unmeasured","rows":rows,"budgets":[{"name":"unlimited"},{"name":"no_raw_read","max_raw_bytes":0},{"name":"labels_le_20","max_support_labels":20}]}
Path("artifacts/g0_dev/g0_dev_input_v2.json").write_text(json.dumps(out,indent=1,ensure_ascii=False)); print("rows",len(rows))
for r in rows: print(r["task"],r["action"],r["seed"],round(r["score"],4),"gpu",r["gpu_seconds"],"bytes",r["raw_bytes"],"labels",r["support_label_count"])
