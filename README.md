# dreambooth

Install

something like

```
python -m venv .venv --prompt dreambooth && \
. .venv/bin/activate && \
pip install --upgrade pip && \
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116 && \
pip install -r requirements.txt && \
pip install ipykernel  && \
pip install jupyterlab  && \
python -m ipykernel install --user --name=venv
accelerate config
```


Run an experiment like:


```
export INPUT_DIR=<path-to-input-image-folder>
export MODEL_DIR=<path-to-output-model>
export LR=1e-6
export TOKEN=aabbccddeeffgg

accelerate launch --gpu_ids 0 \
  train_dreambooth.py \
  --config_file config.yaml \
  instance_data_dir="$INPUT_DIR" \
  output_dir="$MODEL_DIR" \
  max_train_steps=1000 \
  learning_rate="$LR" \
  instance_str="$TOKEN" \
  wandb_mode=disabled \
  with_prior_preservation=false \
  use_tf32=true
```

any arguments passed as `<name>=<value>` will be added to the config. Arguments passed as `--<name> <value>` before `train_dreambooth.py` are expected to be for huggingface accelerate, and those after are options for the script, see `python train_dreambooth.py --help`
