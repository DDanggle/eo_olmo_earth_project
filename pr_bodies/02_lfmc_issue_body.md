`docs/lfmc.md` says the model reaches MSE 580.6 on the test set. I can't get that number from the released checkpoint, but I do get it (and a bit better) by retraining on the released dataset. So I suspect the file on HF is from a different run than the one in the docs — could you check?

**Setup:** olmoearth-runner 0.1.14, rslearn 0.0.27, lightning 2.5.1, Python 3.11, one H200.
Dataset: `projects/lfmc/20251029/dataset.tar` (44,022 windows). Checkpoint: `allenai/OlmoEarth-v1-FT-LFMC-Base/model.ckpt` (1.14 GB, last modified 2025-11-03).

**What I measured (same data, same eval command):**

| weights | split | MSE |
|---|---|---|
| docs claim | test | 580.6 |
| released ckpt | test (4,585 windows) | 951.9 |
| released ckpt | val | 995.3 |
| retrained from scratch with repo `model.yaml`, best epoch 33/100 | test | 558.8 |

```
rslearn model test --config model.yaml --ckpt_path model.ckpt \
    --data.init_args.test_config.tags.split=test
```

(Two small config edits were needed to run on PyPI rslearn 0.0.27: drop `enable_confusion_matrix`, use lightning `ModelCheckpoint` instead of `BestLastCheckpoint`.)

**Things I ruled out:**
- Library version: released ckpt gives 995.3 on rslearn 0.0.27 and 995.4 on master.
- Split mix-up: it's worse on both val and test; the retrain was evaluated with the exact same command.
- The ckpt metadata says epoch 91 / 60,260 steps = 662 steps/epoch, while the released dataset gives 655 steps/epoch, so it may have been trained on a slightly different data snapshot.

Happy to share full logs, configs, or the 558.8 checkpoint if that helps.
