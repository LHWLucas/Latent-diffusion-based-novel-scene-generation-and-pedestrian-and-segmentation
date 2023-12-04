from PIL import Image, ImageDraw
import cv2
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
import torch
import glob
import os
from tqdm import tqdm

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

model_path = './chkpt_outpaint'
chkpt_list = ["checkpoint-500","checkpoint-1000","checkpoint-1500","checkpoint-2000","checkpoint-2500","checkpoint-3000"]
[os.makedirs(f"./{i}/outpainted_images2blended", exist_ok=True) for i in chkpt_list]

# pipe = StableDiffusionInpaintPipeline.from_pretrained(
#     f'./chkpt_outpaint/{mod}',
#     torch_dtype=torch.float16,
# )

for mod in chkpt_list:
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        f'./chkpt_outpaint{mod[mod.find("-")+1:]}/',
        torch_dtype=torch.float16,
    )
    # pipe.unet.load_attn_procs(model_path, subfolder=mod, weight_name="pytorch_model.bin")
    pipe.to("cuda")
    prompt = "photo of a waymodrivscene driving scene"

    already = [i for i in glob.glob(f"./{mod}/outpainted_images2blended/*")]

    for i in tqdm(glob.glob("./stitched_testing/*")):
        print(i,os.path.basename(i),i.replace("stitched_testing","FRONT_BLEND_TESTING"))
        if i.replace("stitched_testing",f"{mod}/outpainted_images2blended") not in already:
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
            # im.save("./test.png")
            # fix the image
            k2 = fix_converted_image(original_image=k, generated_image=im, mask_image=mask)

            k2 = two_images_combine(left_half, k2)

            k2.save(f'./{mod}/outpainted_images2blended/{os.path.basename(i)}')


output_size = (1536, 512)

[os.makedirs(f"./{i}/outpainted_images3blended", exist_ok=True) for i in chkpt_list]

# pipe = StableDiffusionInpaintPipeline.from_pretrained(
#     f'./chkpt_outpaint/{mod}',
#     torch_dtype=torch.float16,
# )
for mod in chkpt_list:
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        f'./chkpt_outpaint{mod[mod.find("-")+1:]}/',
        torch_dtype=torch.float16,
    )
    # pipe.unet.load_attn_procs(model_path, subfolder=mod, weight_name="pytorch_model.bin")
    pipe.to("cuda")
    prompt = "photo of a waymodrivscene driving scene"

    already = [i for i in glob.glob(f"./{mod}/outpainted_images3blended/*")]

    for i in tqdm(glob.glob(f"./{mod}/outpainted_images2blended/*")):
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
            # im.save("./test.png")

            # fix the image
            k2 = fix_converted_image(original_image=k, generated_image=im, mask_image=mask)

            k2 = two_images_combine(left_half, k2)
            k2 = k2.resize(output_size)

            k2.save(f'./{mod}/outpainted_images3blended/{os.path.basename(i)}')