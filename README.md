# dreambooth

Install

something like

```
python -m venv .venv --prompt dreambooth
. .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
accelerate config
```


Run an experiment like:

`accelerate launch --gpu_ids 0,1 train_dreambooth.py --config_file config.yaml learning_rate=5e-6`

any arguments passed as `<name>=<value>` will be added to the config. Arguments passed as `--<name> <value>` before `train_dreambooth.py` are expected to be for huggingface accelerate, and those after are options for the script, see `python train_dreambooth.py --help`
