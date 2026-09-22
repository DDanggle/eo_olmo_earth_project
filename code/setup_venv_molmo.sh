#!/usr/bin/env bash
# Isolated env for Molmo2 (model card requires transformers==4.57.1 + trust_remote_code). Shared venvs untouched.
set -u; cd /home/work/data/olmoearth || exit 1
[ -d .venv-molmo ] || /usr/bin/python3 -m venv .venv-molmo
.venv-molmo/bin/pip install -q --upgrade pip
.venv-molmo/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128 && echo TORCH_OK
.venv-molmo/bin/pip install -q "transformers==4.57.1" accelerate pillow einops numpy && echo TF_OK
.venv-molmo/bin/python -c "import torch,transformers;print(torch.__version__,torch.cuda.is_available(),transformers.__version__)"
echo VENV_MOLMO_DONE
