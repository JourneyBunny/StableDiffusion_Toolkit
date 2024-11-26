### https://pastebin.com/PvL9s0qL
# Stable Diffusion Toolkit
#   An easy-to-use program that will download and operate Stable Diffusion image models
#   Simply calling the script with a prompt will generate an image
#       Optional paramaters like --output filename.png are described in main function
#   The script may also be called with the --inpaint source_file.png parameter
#       The inpaint option causes this script to perform an inpaint on any area 
#       where there is transparency, using your prompt.
#   At this time, SD3.5 does not have an inpaint pipeline
#
# Author: Journey Bunny
#
#    A dedicated graphics card is rather strongly recommended. 
#    The SDXL and SDXL-Turbo models might be suitable to run on CPU. (SDXL is default)
#
#    The first time any model is run, it will download the model. This is going to take a while.
#    If you are trying to load SD3* models, you will get an access denied error.
#        The SD3.5 models are 'gated' meaning you need to accept 
#        terms of service for each model on the website.
#        After gaining access, your Huggingface API token can dl a model.
#        Create a free access key on huggingface and place it in a file 
#        called huggingface_token.txt inside this folder.
#        Gated Models you can accept:
#            https://huggingface.co/stabilityai/stable-diffusion-3.5-large
#            https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo
#            https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
#
#    Use:
#        See down below the main function for an example.
#        This script can be run with nothing but a text prompt:
#            It would use SDXL, 50 steps, cfg 3.5 to produce output.png
#        You can specify any overrides you desire as optional paramaters, 
#        following the example provided.
#
#    Setup:
#        You will need the libraries indicated in the imports below.
#        I'd recommend running in a Python 3.10 environment. You'll need:
#        pip install torch diffusers pillow numpy
###


import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, StableDiffusion3Pipeline
import os
import argparse
from PIL import Image, ImageFilter
import numpy as np

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
inference_steps = 50
default_torch = torch.float16 if device == "cuda" else torch.float32
default_guidance = 3.5

# Model and pipeline information dictionary
model_info = {
    "SDXL": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline": StableDiffusionXLPipeline,
        "inpaint_pipeline": StableDiffusionXLInpaintPipeline,
        "inpaintCFG": 8,
        "steps": 50
    },
    "SDXLT": {
        "model_id": "stabilityai/sdxl-turbo",
        "pipeline": StableDiffusionXLPipeline,
        "inpaint_pipeline": StableDiffusionXLInpaintPipeline,
        "guidance": 0.0,
        "steps": 4
    },
    "SD3.5M": {
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "pipeline": StableDiffusion3Pipeline,
        "steps": 50,
        "type": torch.bfloat16
    },
    "SD3.5L": {
        "model_id": "stabilityai/stable-diffusion-3.5-large",
        "pipeline": StableDiffusion3Pipeline,
        "steps": 50,
        "type": torch.bfloat16
    },
    "SD3.5LT": {
        "model_id": "stabilityai/stable-diffusion-3.5-large-turbo",
        "steps": 4,
        "type": torch.bfloat16,
        "guidance": 0.0
    }
}

# Check if the token file exists and read the token
token_file_path = os.path.join(os.path.dirname(__file__), "huggingface_token.txt")
if os.path.exists(token_file_path):
    with open(token_file_path, "r") as token_file:
        token = token_file.read().strip()
    os.environ["HUGGINGFACE_TOKEN"] = token
    print("Hugging Face token loaded from file.")
else:
    os.environ["HUGGINGFACE_TOKEN"] = ""
    print("No Hugging Face token file found. Token set to an empty string.")

def generate_image(prompt, model_name="SDXL", output="output.png", steps=None, cfg=None):
    """
    Generates an image based on a single prompt using the specified model and pipeline.
    """
    if model_name not in model_info:
        raise ValueError(f"Invalid model_name '{model_name}'. Available options: {list(model_info.keys())}")
    
    model_id = model_info[model_name]["model_id"]
    pipeline_class = model_info[model_name]["pipeline"]

    steps = steps if steps is not None else model_info[model_name].get("steps", inference_steps)
    guidance = cfg if cfg is not None else model_info[model_name].get("guidance", default_guidance)
    ttype = model_info[model_name].get("type", default_torch)

    pipe = pipeline_class.from_pretrained(
        model_id, 
        token=os.getenv("HUGGINGFACE_TOKEN"), 
        torch_dtype=ttype
    )
    pipe.enable_model_cpu_offload()

    print(f"({device}) Generating image for prompt: '{prompt}' with {steps} steps and CFG scale {guidance}")
    image = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance).images[0]
    
    image.save(output)
    print(f"Image saved to {output}")

def inpaint_image(prompt, image_path, model_name="SDXL", output="output_inpaint.png", steps=None, cfg=None):
    """
    Performs inpainting on an image with transparent areas using the specified model and pipeline.
    """
    if model_name not in model_info:
        raise ValueError(f"Invalid model_name '{model_name}'. Available options: {list(model_info.keys())}")

    # Check if inpainting is supported
    if "inpaint_pipeline" not in model_info[model_name]:
        print(f"Inpainting not supported for {model_name}. Returning the original image.")
        init_image = Image.open(image_path).convert('RGBA')
        init_image.save(output)
        print(f"Original image saved to {output} without inpainting.")
        return

    model_id = model_info[model_name].get("inpaint_model_id", model_info[model_name]["model_id"])
    pipeline_class = model_info[model_name]["inpaint_pipeline"]

    steps = steps if steps is not None else model_info[model_name].get("steps", inference_steps)
    guidance = cfg if cfg is not None else model_info[model_name].get("inpaintCFG", model_info[model_name].get("guidance", default_guidance))
    ttype = model_info[model_name].get("type", default_torch)

    pipe = pipeline_class.from_pretrained(
        model_id,
        token=os.getenv("HUGGINGFACE_TOKEN"),
        torch_dtype=ttype
    )
    pipe.enable_model_cpu_offload()

    init_image = Image.open(image_path).convert('RGBA')
    original_size = init_image.size

    # Create mask from alpha channel (1 for transparent areas, 0 for non-transparent)
    alpha = np.array(init_image.split()[-1])
    mask = np.where(alpha < 255, 1, 0).astype(np.float32)

    # Convert mask to PIL Image for blurring
    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
    blurred_mask = mask_image.filter(ImageFilter.GaussianBlur(radius=3))

    # Convert image to RGB for inpainting
    init_image = init_image.convert('RGB')

    print(f"({device}) Inpainting image '{image_path}' with prompt: '{prompt}', {steps} steps, and CFG scale {guidance}")
    
    # Only handle SDXL models
    inpainted_image = pipe(
        prompt=prompt,
        strength=0.9,
        image=init_image,
        mask_image=blurred_mask,
        num_inference_steps=steps,
        guidance_scale=guidance
    ).images[0]

    inpainted_image.save(output)
    print(f"Inpainted image saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an image or perform inpainting using Stable Diffusion models.")
    parser.add_argument("prompt", type=str, help="The text prompt to generate the image or guide the inpainting.")
    parser.add_argument("--model_name", type=str, default="SDXL", help="Name of the model to use. Available options: SDXL, SDXLT, SD3.5M, SD3.5L, SD3.5LT. Defaults to 'SDXL'.")
    parser.add_argument("--output", type=str, default="output.png", help="Name of the output image file. Defaults to 'output.png'.")
    parser.add_argument("--steps", type=int, help="Number of inference steps. Uses global default or model default if not specified.")
    parser.add_argument("--cfg", type=float, help="CFG scale value. Uses global default or model default if not specified.")
    parser.add_argument("--inpaint", type=str, nargs='?', const='', help="Path to an image file to perform inpainting on. The image should be in RGBA format.")

    args = parser.parse_args()

    if args.inpaint:
        inpaint_image(
            prompt=args.prompt,
            image_path=args.inpaint,
            model_name=args.model_name,
            output=args.output,
            steps=args.steps,
            cfg=args.cfg
        )
    else:
        generate_image(
            prompt=args.prompt,
            model_name=args.model_name,
            output=args.output,
            steps=args.steps,
            cfg=args.cfg
        )

# Example usage:
# To generate an image using the script saved as make_picture.py, run the following command:
# python make_picture.py "A futuristic cityscape at sunset with flying cars" --model_name SDXL --output futuristic_city.png --steps 40 --cfg 7.0
#
# To inpaint an image using the script, run the following command:
# python make_picture.py "A cute cartoon puppy dog" --inpaint input_image.png --output inpainted_forest.png --steps 50 --cfg 7.5