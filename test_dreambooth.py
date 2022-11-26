import os
import time
from pathlib import Path
import argparse

import torch

from diffusers import StableDiffusionPipeline, DDIMScheduler

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a testing script.")
    parser.add_argument(
        "--pred_path",
        type=str,
        default=None,
        required=True,
        help="Path to save generate images.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=True,
        help="Path to model.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default="aabbccddeeffgg",
        help="Special token.",
    )
    parser.add_argument(
        "--class_str",
        type=str,
        default="person",
        help="Class string to help the model understand what class the token is.",
    )
    parser.add_argument(
        "--tests",
        type=str,
        default="all",
        help="list of test ids. default all uses all the test prompts",
    )
    parser.add_argument(
        "--num_pred_steps",
        type=int,
        default=75,
        help="Number of steps for inference.",
    )
    parser.add_argument(
        "--guide",
        type=float,
        default=7.5,
        help="Guide power.",
    )
    parser.add_argument(
        "--num_preds",
        type=int,
        default=1,
        help="Number of predictions for each prompt.",
    )
    parser.add_argument(
        '--ddim', 
        action='store_true',
        help="Flag to activate the DDIM scheduler.",
    )

    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    
    token_class_str = args.token + " " + args.class_str
    
    tests = {
            "1": ["photo, colorful cinematic portrait of " + token_class_str + ", armor, cyberpunk,background made of brain cells, back light, organic, art by greg rutkowski, ultrarealistic, leica 30mm", args.num_pred_steps, args.guide, "rutkowski"],
            "2": ["pencil sketch portrait of  " + token_class_str + " inpired by greg rutkowski, digital art by artgem", args.num_pred_steps, args.guide, "rutkowskiartgem"],
            "3": ["photo,colorful cinematic portrait of " + token_class_str + ", " + token_class_str + " with long hair, color lights, on stage, ultrarealistic", args.num_pred_steps, args.guide, "longhair"],
            "4": ["photo portrait of  " + token_class_str + " astronaut, astronaut, helmet in alien world abstract oil painting, greg rutkowski, detailed face", args.num_pred_steps, args.guide, "astronautrutkowski"],
            "5": ["photo portrait of  " + token_class_str + " as firefighter, helmet, ultrarealistic, leica 30mm", args.num_pred_steps, args.guide,  "firefighter"],
            "6": ["photo portrait of  " + token_class_str + " as steampunk warrior, neon organic vines, digital painting", args.num_pred_steps, args.guide, "steampunk"],
            "7": ["impressionist portrait painting of  " + token_class_str + " by Daniel F Gerhartz, (( " + token_class_str + " with painted in an impressionist style)), nature, trees", args.num_pred_steps, args.guide, "danielgerhartz"],
    }

    if args.ddim:
        ddim_scheduler = DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            )

        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_path, 
            torch_dtype=torch.float16,
            scheduler=ddim_scheduler,
            safety_checker=None).to("cuda")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_path, 
            torch_dtype=torch.float16,
            safety_checker=None).to("cuda")    

    Path(args.pred_path).mkdir(parents=True, exist_ok=True)

    for i in range(args.num_preds):
        for key in tests:
            image = pipe(tests[key][0], num_inference_steps=tests[key][1], guidance_scale=tests[key][2]).images[0]
            timestr = time.strftime("%Y%m%d-%H%M%S")
            image.save(args.pred_path + "/" + key + "-" + tests[key][3] + "-" + timestr + ".png")
        
if __name__ == "__main__":
    main()
