from PIL import Image, ImageDraw
import cv2
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
import torch
import glob
import os
from tqdm import tqdm
import PIL
import requests
import torch
from io import BytesIO
import math
# import modules.scripts as scripts
import gradio as gr


class Script():
    def title(self):
        return "Poor man's outpainting"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        if not is_img2img:
            return None

        pixels = gr.Slider(label="Pixels to expand", minimum=8, maximum=256, step=8, value=128, elem_id=self.elem_id("pixels"))
        mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4, elem_id=self.elem_id("mask_blur"))
        inpainting_fill = gr.Radio(label='Masked content', choices=['fill', 'original', 'latent noise', 'latent nothing'], value='fill', type="index", elem_id=self.elem_id("inpainting_fill"))
        direction = gr.CheckboxGroup(label="Outpainting direction", choices=['left', 'right', 'up', 'down'], value=['left', 'right', 'up', 'down'], elem_id=self.elem_id("direction"))

        return [pixels, mask_blur, inpainting_fill, direction]

    def run(self, p, pixels, mask_blur, inpainting_fill, direction):
        initial_seed = None
        initial_info = None

        p.mask_blur = mask_blur * 2
        p.inpainting_fill = inpainting_fill
        p.inpaint_full_res = False

        left = pixels if "left" in direction else 0
        right = pixels if "right" in direction else 0
        up = pixels if "up" in direction else 0
        down = pixels if "down" in direction else 0

        init_img = p.init_images[0]
        target_w = math.ceil((init_img.width + left + right) / 64) * 64
        target_h = math.ceil((init_img.height + up + down) / 64) * 64

        if left > 0:
            left = left * (target_w - init_img.width) // (left + right)
        if right > 0:
            right = target_w - init_img.width - left

        if up > 0:
            up = up * (target_h - init_img.height) // (up + down)

        if down > 0:
            down = target_h - init_img.height - up

        img = Image.new("RGB", (target_w, target_h))
        img.paste(init_img, (left, up))

        mask = Image.new("L", (img.width, img.height), "white")
        draw = ImageDraw.Draw(mask)
        draw.rectangle((
            left + (mask_blur * 2 if left > 0 else 0),
            up + (mask_blur * 2 if up > 0 else 0),
            mask.width - right - (mask_blur * 2 if right > 0 else 0),
            mask.height - down - (mask_blur * 2 if down > 0 else 0)
        ), fill="black")

        latent_mask = Image.new("L", (img.width, img.height), "white")
        latent_draw = ImageDraw.Draw(latent_mask)
        latent_draw.rectangle((
             left + (mask_blur//2 if left > 0 else 0),
             up + (mask_blur//2 if up > 0 else 0),
             mask.width - right - (mask_blur//2 if right > 0 else 0),
             mask.height - down - (mask_blur//2 if down > 0 else 0)
        ), fill="black")

        devices.torch_gc()

        grid = images.split_grid(img, tile_w=p.width, tile_h=p.height, overlap=pixels)
        grid_mask = images.split_grid(mask, tile_w=p.width, tile_h=p.height, overlap=pixels)
        grid_latent_mask = images.split_grid(latent_mask, tile_w=p.width, tile_h=p.height, overlap=pixels)

        p.n_iter = 1
        p.batch_size = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True

        work = []
        work_mask = []
        work_latent_mask = []
        work_results = []

        for (y, h, row), (_, _, row_mask), (_, _, row_latent_mask) in zip(grid.tiles, grid_mask.tiles, grid_latent_mask.tiles):
            for tiledata, tiledata_mask, tiledata_latent_mask in zip(row, row_mask, row_latent_mask):
                x, w = tiledata[0:2]

                if x >= left and x+w <= img.width - right and y >= up and y+h <= img.height - down:
                    continue

                work.append(tiledata[2])
                work_mask.append(tiledata_mask[2])
                work_latent_mask.append(tiledata_latent_mask[2])

        batch_count = len(work)
        print(f"Poor man's outpainting will process a total of {len(work)} images tiled as {len(grid.tiles[0][2])}x{len(grid.tiles)}.")

        state.job_count = batch_count

        for i in range(batch_count):
            p.init_images = [work[i]]
            p.image_mask = work_mask[i]
            p.latent_mask = work_latent_mask[i]

            state.job = f"Batch {i + 1} out of {batch_count}"
            processed = process_images(p)

            if initial_seed is None:
                initial_seed = processed.seed
                initial_info = processed.info

            p.seed = processed.seed + 1
            work_results += processed.images


        image_index = 0
        for y, h, row in grid.tiles:
            for tiledata in row:
                x, w = tiledata[0:2]

                if x >= left and x+w <= img.width - right and y >= up and y+h <= img.height - down:
                    continue

                tiledata[2] = work_results[image_index] if image_index < len(work_results) else Image.new("RGB", (p.width, p.height))
                image_index += 1

        combined_image = images.combine_grid(grid)

        if opts.samples_save:
            images.save_image(combined_image, p.outpath_samples, "", initial_seed, p.prompt, opts.samples_format, info=initial_info, p=p)

        processed = Processed(p, [combined_image], initial_seed, initial_info)

        return processed

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

prompt_list = None
with open("./prompt.txt","r") as f:
    prompt_list = [i.replace("\n","") for i in f.readlines() if i.replace("\n","")!='']
[os.makedirs(f"./base_stable_diffusion_finetuned/{n+1}/outpainted_images2blended", exist_ok=True) for n,i in enumerate(prompt_list)]

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "./finetuning",
    torch_dtype=torch.float16,
)
pipe.to("cuda")

for n,prompt in enumerate(prompt_list):

    already = [i for i in glob.glob(f"./base_stable_diffusion_finetuned/{n+1}/outpainted_images2blended/*")]

    for i in tqdm(glob.glob("./stitched_testing/*")[:100]):
        print(i,os.path.basename(i),i.replace("stitched_testing","FRONT_BLEND_TESTING"))
        if i.replace("stitched_testing",f"base_stable_diffusion_finetuned/{n+1}/outpainted_images2blended") not in already:
            image = Image.open(i.replace("stitched_testing","FRONT_BLEND_TESTING"))
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
            im = pipe(prompt=prompt, image=k, mask_image=mask).images[0]
            im = im.resize((512,512))
            # im.save("./test.png")
            # fix the image
            k2 = fix_converted_image(original_image=k, generated_image=im, mask_image=mask)

            k2 = two_images_combine(left_half, k2)

            k2.save(f'./base_stable_diffusion_finetuned/{n+1}/outpainted_images2blended/{os.path.basename(i)}')


output_size = (1536, 512)

[os.makedirs(f"./base_stable_diffusion_finetuned/{n+1}/outpainted_images3blended", exist_ok=True) for n,i in enumerate(prompt_list)]

for n,prompt in enumerate(prompt_list):
    already = [i for i in glob.glob(f"./base_stable_diffusion_finetuned/{n+1}/outpainted_images3blended/*")]

    for i in tqdm(glob.glob(f"./base_stable_diffusion_finetuned/{n+1}/outpainted_images2blended/*")[:100]):
        print(i,os.path.basename(i))
        if i.replace("outpainted_images2blended","outpainted_images3blended") not in already:
            image = Image.open(i)
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
            im = pipe(prompt=prompt, image=k, mask_image=mask).images[0]
            im = im.resize((512,512))
            # im.save("./test.png")

            # fix the image
            k2 = fix_converted_image(original_image=k, generated_image=im, mask_image=mask)

            k2 = two_images_combine(left_half, k2)
            k2 = k2.resize(output_size)

            k2.save(f'./base_stable_diffusion_finetuned/{n+1}/outpainted_images3blended/{os.path.basename(i)}')