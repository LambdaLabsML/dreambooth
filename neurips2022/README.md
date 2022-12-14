# NeuRIPS Workshop Guide

1. [Step One: Launch a Lambda Cloud Instance](#step-one-launch-a-lambda-cloud-instance)
2. [Step Two: Download Notebooks](#step-two-download-notebooks)
3. [Step Three: Run Notebooks](#step-three-run-notebooks)
5. [Q&A](#qa)

## Step One: Launch a Lambda Cloud Instance

The hands-on practice requires a Lambda Cloud account and a `1xA100` GPU instance, which costs `$1.10 USD/h`. 

Remember to drop by our photo booth if you do not wish to sign up. Our staff will take photos and train a model for you. Leave us your email address, and we will send the results within 12 hours.

If you wish to sign up, go to [https://cloud.lambdalabs.com/](https://cloud.lambdalabs.com/) and follow the signup step.

Once signed in with your Lambda Cloud account, click the `Launch Instance` button.

<img src="./images/lambda_cloud_dashboard.jpg" alt="drawing" style="width:640px;"/>

Lambda Cloud will ask you for payment information when launching the first instance. Just follow the instructions and be aware that Lambda Cloud will place a __temporary__ `$10 USD` pre-auth to verify your card, which will disappear within seven days. Once payment information is provided, you can launch an instance. For this workshop:
* Choose 1xA100 instance (40GB SXM4 or 40GB PCIe are both fine).
* Any region will work.
* Don't need to attach a filesystem.
* Follow the guide to add or generate an SSH key -- this step can not be skipped. However, this workshop won't use this key because all practice can be accomplished in the Cloud IDE (so no SSH session is needed).

It takes about two mins to get the instance running (the green tick as shown in the picture below). 

<img src="./images/lambda_cloud_dashboard_instance_ready.png" alt="drawing" style="width:640px;"/>

You need to click the Cloud IDE `Launch` button (the purple button on the right end) to get access to the Jupyter Hub. If you see a message saying, "Your Jupyter Notebook is down," that means the Jupyter Hub isn't ready and please give it another minute or so. Eventually, it will look like this once it is ready:

<img src="./images/lambda_cloud_jupyter_hub.jpg" alt="drawing" style="width:640px;"/>


## Step Two: Download Notebooks

Create a terminal by clicking the `Terminal` icon, and run the following command in the terminal to download a few notebooks to your home directory.

```
wget https://raw.githubusercontent.com/LambdaLabsML/dreambooth/master/neurips2022/setup.ipynb && \
wget https://raw.githubusercontent.com/LambdaLabsML/dreambooth/master/neurips2022/train.ipynb && \
wget https://raw.githubusercontent.com/LambdaLabsML/dreambooth/master/neurips2022/test_param.ipynb && \
wget https://raw.githubusercontent.com/LambdaLabsML/dreambooth/master/neurips2022/test_prompt.ipynb && \
wget https://raw.githubusercontent.com/LambdaLabsML/dreambooth/master/neurips2022/test_many_prompts.ipynb
```

You need to click the refresh button in the `File Browser` (on the left side of the IDE) to see the notebooks. 

<img src="./images/lambda_cloud_dashboard_download_ipynb.jpg" alt="drawing" style="width:640px;"/>

Now you are ready to kick off the DreamBooth practice!

## Step Three: Run Notebooks

### Run `setup.ipynb`

This notebook will clone the DreamBooth repo and install several python packages. It will also create several folders in the home directory:
* `/home/ubuntu/data`: This directory stores training photos.
* `/home/ubuntu/model`: This is the directory where the trained model will be saved.
* `/home/ubuntu/output`: This is the directory where the sampled images will be saved.

The last step in this notebook will ask for an access token for downloading the Stable Diffusion model from Huggingface. You need to:  
* Create a [huggingface](https://huggingface.co/) account if you don't have one.
* Create your access token from "Settings - Access Tokens - New Token," and paste the token into the login field at the end of the notebook.
* Accept the [license of Stable Diffusion v1-4 Model Card](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) if you agree. (Otherwise can't use the model)
<img src="./images/hf_token.jpg" alt="drawing" style="width:640px;"/>
<img src="./images/hf_model_card.jpg" alt="drawing" style="width:640px;"/>

### Upload Images

We recommend preparing ~20 photos: ten close-ups of your face with various poses and facial expressions, five photos from your chest and up, and a few full-body shots.

### Run `train.ipynb`
This notebook trains a DreamBooth model use the images inside of `/home/ubuntu/data`.

Once trained, it will also run a few inferences and display the prompts and sampled images at the end of the notebook.

<img src="./images/train.jpg" alt="drawing" style="width:640px;"/>

Also don't forget to checkout [our case studies](https://wandb.ai/justinpinkney/dreambooth/reports/Training-comparisons--VmlldzozMDM0MzY2) for different training settings.

### Run `test_prompt.ipynb`, `test_many_prompts.ipynb`, and `test_param.ipynb`
You can use these notebooks to play with the model you just trained. 

* `test_prompt.ipynb`: A notebook for prompt engineering. You will use fixed latent input to conduct controlled experiments for the impact of prompt engineering on the model output.

* `test_many_prompts.ipynb`: A notebook that let you try some pre-collected prompts.

* `test_param.ipynb`: A notebook for trying different parameters for inference. Again, you will use fixed latent input to conduct controlled experiments for the impact of these parameters on the model output. 


## Q&A

### How to free GPU memory of unused notebooks?

When a notebook is completed and closed, its kernel may still be alive. To free the GPU memory, you need to go to the `Running Terminals and Kernels` on the left side of the screen and manually shut down the unwanted kernels:

<img src="./images/lambda_cloud_jupyter_hub_kill_kernel.jpg" alt="drawing" style="width:640px;"/>


### Where can I find inspirations for prompts?

Try [publicprompts.art](https://publicprompts.art/) and [Gustavosta's Prompt dataset](https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts).