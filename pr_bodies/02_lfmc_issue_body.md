For context, I've been using OlmoEarth on a few side projects since the release, including a before/after review map for the [2026 Nepal disaster](https://github.com/DDanggle/eo-rasuwa), year-to-year monitoring in Korea.

I came across this while setting up LFMC as a reference for one of those projects.

[`docs/lfmc.md`](https://github.com/allenai/olmoearth_projects/blob/main/docs/lfmc.md) says the model gets MSE 580.6 on the test set. I couldn't get anywhere near that with the released checkpoint, but retraining on the released dataset with the repo config got me there (slightly better, actually). So I think the file on HF might be from a different run than the one the docs describe. Could you take a look?

Setup: olmoearth-runner 0.1.14, rslearn 0.0.27, lightning 2.5.1, Python 3.11, one H200. Dataset is [`projects/lfmc/20251029/dataset.tar`](https://storage.googleapis.com/ai2-olmoearth-projects-public-data/projects/lfmc/20251029/dataset.tar) (44,022 windows), checkpoint is [`allenai/OlmoEarth-v1-FT-LFMC-Base/model.ckpt`](https://huggingface.co/allenai/OlmoEarth-v1-FT-LFMC-Base/tree/main) (1.14 GB, last modified 2025-11-03).

Same data, same eval command for every row:

| weights | split | MSE |
|---|---|---|
| docs | test | 580.6 |
| released ckpt | test (4,585 windows) | 951.9 |
| released ckpt | val | 995.3 |
| retrained from scratch, repo [`model.yaml`](https://github.com/allenai/olmoearth_projects/blob/main/olmoearth_run_data/lfmc/model.yaml), best epoch 33/100 | test | 558.8 |

```
rslearn model test --config model.yaml --ckpt_path model.ckpt \
    --data.init_args.test_config.tags.split=test
```

To run on PyPI rslearn 0.0.27 I had to drop `enable_confusion_matrix` and swap `BestLastCheckpoint` for lightning's `ModelCheckpoint`; nothing else changed.

A few things I checked so it's hopefully not me:

- Not the library version. The released ckpt gives 995.3 on rslearn 0.0.27 and 995.4 on master.
- Not a split mix-up. It's off on both val and test, and the retrain was scored with the exact same command.
- One hint: the ckpt metadata says epoch 91 / 60,260 steps, i.e. 662 steps per epoch, while the released dataset gives 655. So it may have been trained on a slightly different snapshot of the data.

Happy to send logs, configs, or the 558.8 checkpoint if that's useful.
