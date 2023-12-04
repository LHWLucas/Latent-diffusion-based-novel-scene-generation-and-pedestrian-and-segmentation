from PIL import Image, ImageDraw
import cv2
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
import torch
import glob
from tqdm import tqdm
import os, subprocess
import csv

def setup():
    install_cmds = [
        ['pip', 'install', 'gradio'],
        ['pip', 'install', 'open_clip_torch'],
        ['pip', 'install', 'clip-interrogator'],
    ]
    for cmd in install_cmds:
        print(subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode('utf-8'))

setup()

import gradio as gr
from clip_interrogator import Config, Interrogator
import argparse

prompt_mode = 'best' #@param ["best","fast","classic","negative"]
output_mode = 'desc' #@param ["desc.csv","rename"]
max_filename_len = 128

caption_model_name = 'blip-large' #@param ["blip-base", "blip-large", "git-large-coco"]
clip_model_name = 'ViT-L-14/openai' #@param ["ViT-L-14/openai", "ViT-H-14/laion2b_s32b_b79k"]

config = Config()
config.clip_model_name = clip_model_name
config.caption_model_name = caption_model_name
ci = Interrogator(config)

def image_analysis(image):
    image = image.convert('RGB')
    image_features = ci.image_to_features(image)

    top_mediums = ci.mediums.rank(image_features, 5)
    top_artists = ci.artists.rank(image_features, 5)
    top_movements = ci.movements.rank(image_features, 5)
    top_trendings = ci.trendings.rank(image_features, 5)
    top_flavors = ci.flavors.rank(image_features, 5)

    medium_ranks = {medium: sim for medium, sim in zip(top_mediums, ci.similarities(image_features, top_mediums))}
    artist_ranks = {artist: sim for artist, sim in zip(top_artists, ci.similarities(image_features, top_artists))}
    movement_ranks = {movement: sim for movement, sim in zip(top_movements, ci.similarities(image_features, top_movements))}
    trending_ranks = {trending: sim for trending, sim in zip(top_trendings, ci.similarities(image_features, top_trendings))}
    flavor_ranks = {flavor: sim for flavor, sim in zip(top_flavors, ci.similarities(image_features, top_flavors))}
    
    return medium_ranks, artist_ranks, movement_ranks, trending_ranks, flavor_ranks

def image_to_prompt(image, mode):
    ci.config.chunk_size = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    ci.config.flavor_intermediate_count = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    image = image.convert('RGB')
    if mode == 'best':
        return ci.interrogate(image)
    elif mode == 'classic':
        return ci.interrogate_classic(image)
    elif mode == 'fast':
        return ci.interrogate_fast(image)
    elif mode == 'negative':
        return ci.interrogate_negative(image)


def sanitize_for_filename(prompt: str, max_len: int) -> str:
    name = "".join(c for c in prompt if (c.isalnum() or c in ",._-! "))
    name = name.strip()[:(max_len-4)] # extra space for extension
    return name

ci.config.quiet = True

def create_mask(width, height):
    mask = Image.new('L', (width, height), color="white")
    return mask

def mask_edit(mask, x, y, width, height):
    draw = ImageDraw.Draw(mask)
    draw.rectangle((x, y, x + width, y + height), fill="black")
    return mask

def two_images_combine(image1, image2):
    combined = Image.new('RGB', (image1.width + image2.width, image1.height))
    combined.paste(image1, (0, 0))
    combined.paste(image2, (image1.width, 0))
    return combined

def apply_mask(image, mask):
    image.putalpha(mask)
    return image

# negative_prompt = "3d, cartoon, animated"
# A generated image of unmasked areas by diffusers is changed a little bit
# So we need to replace the unmasked areas with the original image
def fix_converted_image(
    original_image: Image.Image, generated_image: Image.Image, mask_image: Image.Image
) -> Image.Image:
    # PIL.Image to numpy
    original_image_array = np.array(original_image)
    generated_image_array = np.array(generated_image)

    # invert mask
    mask_image_array = np.array(mask_image)
    mask_image_inverted = cv2.bitwise_not(mask_image_array)
    mask_image_inverted = mask_image_inverted.reshape(mask_image_inverted.shape + (1,))

    # replace pixels
    converted_image_array = np.where(
        mask_image_inverted == 255, original_image_array, generated_image_array
    )

    # numpy to PIL.Image
    converted_image = Image.fromarray(converted_image_array)
    return converted_image

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def has_stitching_seam(image_path, threshold_intensity_change=40000, smoothing_window=1):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(img.shape)

    # Apply an edge detection algorithm to find the seam
    edges = cv2.Canny(img, 0, 100)

    # Sum along the rows to find the intensity profile
    intensity_profile = np.sum(edges, axis=0)

    # Apply smoothing to the intensity profile and change in intensity
    smoothed_intensity_profile = moving_average(intensity_profile, smoothing_window)

    # Calculate the change in intensity
    intensity_change = np.abs(np.diff(smoothed_intensity_profile.astype(np.int64)))
    smoothed_intensity_change = moving_average(intensity_change, 100)

    # Check if there are peaks in the intensity change indicating a seam
    # seam_detected = np.max(np.abs(np.diff(smoothed_intensity_profile[(smoothed_intensity_profile.shape[-1]//2)-40:(smoothed_intensity_profile.shape[-1]//2)+40].astype(np.int64)))) > np.max(smoothed_intensity_profile)/5
    # seam_detected1 = np.max(np.abs(np.diff(smoothed_intensity_profile[(smoothed_intensity_profile.shape[-1]//1.4)-40:(smoothed_intensity_profile.shape[-1]//1.4)+40].astype(np.int64)))) > np.max(smoothed_intensity_profile)/5
    seam_detected = np.max(smoothed_intensity_profile[(smoothed_intensity_profile.shape[-1]//2)-100:(smoothed_intensity_profile.shape[-1]//2)+100].astype(np.int64)) > np.mean(smoothed_intensity_profile)*2.5
    seam_detected1 = np.max(smoothed_intensity_profile[int(smoothed_intensity_profile.shape[-1]//1.4)-100:int(smoothed_intensity_profile.shape[-1]//1.4)+100].astype(np.int64)) > np.mean(smoothed_intensity_profile)*2.5

    return seam_detected, seam_detected1

os.makedirs(f"./finetuned_stable_diffusion_outpainting6/outpainted_images2blended", exist_ok=True)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "./finetuning3",
    torch_dtype=torch.float16,
)
pipe.to("cuda")

already = [i for i in glob.glob(f"./finetuned_stable_diffusion_outpainting6/outpainted_images2blended/*")]
prompts = {"prompts":[]}

output_size = (1536, 512)

os.makedirs(f"./finetuned_stable_diffusion_outpainting6/outpainted_images3blended", exist_ok=True)

already2 = [i for i in glob.glob(f"./finetuned_stable_diffusion_outpainting6/outpainted_images3blended/*")]
counter = {"counter":0}

def recur(i, already, already2, prompt_mode):
    negative_prompt = "3d, cartoon, animated"
    if i.replace("stitched_testing",f"finetuned_stable_diffusion_outpainting6/outpainted_images2blended") not in already:
        image = Image.open(i.replace("stitched_testing","FRONT_BLEND_TESTING"))
        prompt = image_to_prompt(image, prompt_mode)
        prompt+=", pedestrians, front right"
        print(prompt)
        # prompts.append(prompt)
        prompts["prompts"].append(prompt)
        mask = create_mask(image.width, image.height)
        mask = mask_edit(mask, 0, 0, 220, 512)
        # image.show()
        # mask.show()

        # Get the dimensions of the original image
        width, height = image.size

        # Calculate the coordinates for the right half of the image
        left = width // 2
        right = width
        top = 0
        bottom = height

        # Crop the right half of the image
        right_half = image.crop((left, top, right, bottom))
        left_half = image.crop((0, top, width // 2, bottom))
        # right_half.show()
        # left_half.show()

        # Create a new image with white pixels
        new_width = width - left
        new_image = Image.new('RGB', (new_width, height), (255, 255, 255))

        # Paste the cropped right half onto the new image
        k = two_images_combine(right_half, new_image)

        #image and mask_image should be PIL images.
        #The mask structure is white for inpainting and black for keeping as is
        im = pipe(prompt=prompt, image=k, mask_image=mask, negative_prompt=negative_prompt).images[0]
        # im.save("./test.png")
        # fix the image
        k2 = fix_converted_image(original_image=k, generated_image=im, mask_image=mask)

        k2 = two_images_combine(left_half, k2)

        k2.save(f'./finetuned_stable_diffusion_outpainting6/outpainted_images2blended/{os.path.basename(i)}')

    if i.replace("stitched_testing",f"finetuned_stable_diffusion_outpainting6/outpainted_images3blended") not in already2:
        image = Image.open(i.replace("stitched_testing",f"finetuned_stable_diffusion_outpainting6/outpainted_images2blended"))
        # print(image.width, image.height)
        mask = create_mask(512, image.height)
        mask = mask_edit(mask, 0, 0, 220, 512)
        # image.show()
        # mask.show()

        # Get the dimensions of the original image
        width, height = image.size

        # Calculate the coordinates for the right half of the image
        left = 2*width // 3
        right = width
        top = 0
        bottom = height

        # Crop the right half of the image
        right_half = image.crop((left, top, right, bottom))
        left_half = image.crop((0, top, 2*width // 3, bottom))
        # right_half.show()
        # left_half.show()

        # Create a new image with white pixels
        new_width = width - left
        new_image = Image.new('RGB', (new_width, height), (255, 255, 255))

        # Paste the cropped right half onto the new image
        k = two_images_combine(right_half, new_image)

        #image and mask_image should be PIL images.
        #The mask structure is white for inpainting and black for keeping as is
        im = pipe(prompt=prompts["prompts"][-1], image=k, mask_image=mask, negative_prompt=negative_prompt).images[0]
        # im.save("./test.png")

        # fix the image
        k2 = fix_converted_image(original_image=k, generated_image=im, mask_image=mask)

        k2 = two_images_combine(left_half, k2)
        k2 = k2.resize(output_size)

        k2.save(f'./finetuned_stable_diffusion_outpainting6/outpainted_images3blended/{os.path.basename(i)}')
        print(f"counter: {counter['counter']}")

        result,result1 = has_stitching_seam(f'./finetuned_stable_diffusion_outpainting6/outpainted_images3blended/{os.path.basename(i)}')
        if counter["counter"]<10:
            if result or result1:
                os.remove(f'./finetuned_stable_diffusion_outpainting6/outpainted_images2blended/{os.path.basename(i)}')
                os.remove(f'./finetuned_stable_diffusion_outpainting6/outpainted_images3blended/{os.path.basename(i)}')
                counter["counter"]+=1
                recur(i, already, already2, prompt_mode)
            else:
                counter["counter"] = 0
                return
        else:
            os.remove(f'./finetuned_stable_diffusion_outpainting6/outpainted_images2blended/{os.path.basename(i)}')
            os.remove(f'./finetuned_stable_diffusion_outpainting6/outpainted_images3blended/{os.path.basename(i)}')
            counter["counter"] = 0
            return

for i in tqdm(glob.glob("./stitched_testing/*")[:100]):
    print(i,os.path.basename(i),i.replace("stitched_testing","FRONT_BLEND_TESTING"))
    recur(i, already, already2, prompt_mode)