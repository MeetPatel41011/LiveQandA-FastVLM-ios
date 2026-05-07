#
# Modified from LLaVA/predict.py
# Please see ACKNOWLEDGEMENTS for details about LICENSE
#
import os
import argparse

import torch
from PIL import Image
import numpy as np
import cv2

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from inference import EdgeAgent, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def predict(args):
    # Remove generation config from model folder
    # to read generation parameters from args
    model_path = os.path.expanduser(args.model_path)
    generation_config = None
    if os.path.exists(os.path.join(model_path, 'generation_config.json')):
        generation_config = os.path.join(model_path, '.generation_config.json')
        os.rename(os.path.join(model_path, 'generation_config.json'),
                  generation_config)

    # Load Agent
    disable_torch_init()
    # If user specifies a device via CLI, we use it; otherwise, the Agent auto-detects
    cli_device = args.device if args.device != "cpu" else None 
    agent = EdgeAgent(model_path=model_path, device=cli_device)

    # Load image and convert to BGR numpy array for the agent
    image = Image.open(args.image_file).convert('RGB')
    open_cv_image = np.array(image) 
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 

    # Run agentic inference
    print(f"\n\U0001f50e Analyzing: {args.prompt}")
    stream = agent.generate_stream(image=open_cv_image, prompt=args.prompt)
    
    print("-" * 30)
    for block in stream:
        print(block, end="", flush=True)
    print("\n" + "-" * 30)

    # Restore generation config
    if generation_config is not None:
        os.rename(generation_config, os.path.join(model_path, 'generation_config.json'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None, help="location of image file")
    parser.add_argument("--prompt", type=str, default="Describe the image.", help="Prompt for VLM.")
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: cpu, cuda, or mps")
    args = parser.parse_args()

    predict(args)
