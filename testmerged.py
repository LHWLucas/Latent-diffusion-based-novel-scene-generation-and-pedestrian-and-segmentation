from PIL import Image, ImageDraw
import cv2
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
import torch
import glob
import os
import random
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

prompts = ['cars are driving down a street with a lot of traffic, so - s 1 4 8 ft light, empty streetscapes, phoenix, half - length photo, pixvy, partially biomedical design, midjourney, the clear sky, soft light - n 9, infrastructure, isomeric view, full width, connector, fig.1, avenue, front right',
           'there are many houses on the street with a car parked in front, 2 4 mm leica anamorphic lens, photograph of san francisco, in a candy land style house, in the hillside, taken in the late 2010s, ornamental aesthetics, other smaller buildings, shot on 16mm film, taken with canon eos 5 d, standing in township street, front right',
            'cars parked on the side of the road in a residential area, in a las vegas street, aspect ratio 1:3, neighborhood themed, found on google street view, the infrastructure of humanity, late 2000â€™s, dynamic comparison, 2 0 1 0 photo, narrow footpath, front right',
            'there is a empty street with a stop sign on the side, 120 degree view, beutifull, captured with sony a3 camera, buses, rossier, perfect weather, borja, beautiful - n 9, photo 3 d, realistic - n 9, solitude, front right'
            ]

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    './mergedckpt/',
    torch_dtype=torch.float16,
)
# pipe.unet.load_attn_procs(model_path, subfolder=mod, weight_name="pytorch_model.bin")
pipe.to("cuda")
# prompt = "photo of a waymodrivscene driving scene"
os.makedirs("./outpainted_images2blendedmerged", exist_ok=True)
already = [i for i in glob.glob(f"./outpainted_images2blendedmerged/*")]

for i in tqdm(glob.glob("./stitched_testing/*")):
    print(i,os.path.basename(i),i.replace("stitched_testing","FRONT_BLEND_TESTING"))
    if i.replace("stitched_testing","outpainted_images2blendedmerged") not in already:
        image = Image.open(i.replace("stitched_testing","FRONT_BLEND_TESTING"))
        mask = create_mask(image.width, image.height)
        mask = mask_edit(mask, 0, 0, 220, 512)
        # image.show()
        # mask.show()
        prompt = random.choice(prompts)

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

        k2.save(f'./outpainted_images2blendedmerged/{os.path.basename(i)}')


output_size = (1536, 512)

os.makedirs("./outpainted_images3blendedmerged", exist_ok=True)


pipe = StableDiffusionInpaintPipeline.from_pretrained(
    './mergedckpt/',
    torch_dtype=torch.float16,
)
# pipe.unet.load_attn_procs(model_path, subfolder=mod, weight_name="pytorch_model.bin")
pipe.to("cuda")
# prompt = "photo of a waymodrivscene driving scene"

already = [i for i in glob.glob("./outpainted_images3blendedmerged/*")]

for i in tqdm(glob.glob("./outpainted_images2blendedmerged/*")):
    print(i,os.path.basename(i))
    if i.replace("outpainted_images2blendedmerged","outpainted_images3blendedmerged") not in already:
        image = Image.open(i)
        # print(image.width, image.height)
        mask = create_mask(512, image.height)
        mask = mask_edit(mask, 0, 0, 220, 512)
        # image.show()
        # mask.show()
        prompt = random.choice(prompts)

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

        k2.save(f'./outpainted_images3blendedmerged/{os.path.basename(i)}')