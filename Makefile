MODEL_NAME="CompVis/stable-diffusion-v1-4"
INSTANCE_DIR="datasets/huw-edwards-cropped-tight"
OUTPUT_DIR="logs"
CLASS_DIR="datasets/person-class"

BASE_ARGS=--pretrained_model_name_or_path=$(MODEL_NAME)  \
	--instance_data_dir=$(INSTANCE_DIR) \
	--output_dir=$(OUTPUT_DIR)/baseline \
	--instance_prompt="a photo of sks person" \
	--resolution=512 \
	--train_batch_size=2 \
	--gradient_accumulation_steps=1 \
	--lr_scheduler="constant" \
	--lr_warmup_steps=0 \
	--max_train_steps=2000 \
	--val_prompts=val_prompts.txt

baseline:
	accelerate launch train_dreambooth.py $(BASE_ARGS) --learning_rate=5e-6

prior-preservation:
	accelerate launch train_dreambooth.py $(BASE_ARGS) --learning_rate=5e-6 \
		--class_data_dir=$(CLASS_DIR) --class_prompt="a photo of person" \
		--with_prior_preservation --prior_loss_weight=1.0

low-lr:
	accelerate launch train_dreambooth.py $(BASE_ARGS) --learning_rate=1e-6

train-text-encoder:
	accelerate launch train_dreambooth.py $(BASE_ARGS) --learning_rate=5e-6 --train_text_encoder

train-text-encoder-low-lr:
	accelerate launch train_dreambooth.py $(BASE_ARGS) --learning_rate=1e-6 --train_text_encoder

prior-and-text:
	accelerate launch train_dreambooth.py $(BASE_ARGS) --learning_rate=5e-6 --train_text_encoder \
		--class_data_dir=$(CLASS_DIR) --class_prompt="a photo of person" \
		--with_prior_preservation --prior_loss_weight=1.0
