import torch
from torchvision import transforms
from PIL import Image
import glob
from tqdm import tqdm
import utils
import cv2
import os
import numpy as np
from FastSAM.fastsam import FastSAM, FastSAMPrompt
import ast
import gc
from FastSAM.utils.tools import convert_box_xywh_to_xyxy
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.optim.lr_scheduler import StepLR
from engine import train_one_epoch, evaluate
import utils
import transforms as T

def detect_lines(image_path, threshold=100):
        # Load the image
        image = cv2.imread(image_path, 0)[:, 470:520]  # Load as grayscale

        # Apply edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        # Apply Hough Line Transform
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=threshold, minLineLength=100, maxLineGap=10)

        if lines is not None:
            # Lines detected
            return False
        else:
            # No lines detected
            return True

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_paths)

# Define the transformations to be applied to the images
transform = transforms.Compose([
    transforms.Resize((1280, 1920)),  # Resize the image to 1920x1280 pixels
    transforms.ToTensor()
    # transforms.ToPILImage(),  # Convert image to tensor
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize image pixels
])

# List of image paths after filtering for bad outpainted images
image_paths = []
for i in tqdm(sorted(glob.glob("./checkpoint-3000/outpainted_images3blended/*"))):
  if detect_lines(i):
    image_paths.append(i)

# Create the custom dataset
dataset = CustomDataset(image_paths, transform=transform)

# Create the DataLoader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
num_classes = 4

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=5e-4,)
                            # momentum=0.1, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()
print(param_size)
print(buffer_size)
# put the model in evaluation mode
model.load_state_dict(torch.load('./maskRCNN_vehicleped_model_blurstitched.pt'))
model.eval()

# Define the file path where you want to save the image
output_path = "saved_image.png"  # Change the filename and extension as needed
model_path = './FastSAM.pt'
img_path = f'./{output_path}'
iou = 0.9
text_prompt = None
conf = 0.6
output = "./output/"
randomcolor = True
point_prompt = "[[0,0]]" #help="[[x1,y1],[x2,y2]]"
point_label = "[0]" #help="[1,0] 0:background, 1:foreground"
box_prompt = "[[0,0,0,0]]" #help="[[x,y,w,h],[x2,y2,w2,h2]] support multiple boxes"
better_quality = False
device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
retina = True
withContours = False #help="draw the edges of the masks"

model_sam = FastSAM(model_path)
point_prompt = ast.literal_eval(point_prompt)
box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(box_prompt))
point_label = ast.literal_eval(point_label)
data = []
for i in dataloader:
    data.append(i)
data = np.concatenate(data,axis=0)
for m,j in enumerate(tqdm(data)):
    m = str(m)
    # pick one image from the test set
    # index = 5
    # img = [i for i in dataloader][0][index]
    img = torch.tensor(j)
    # # put the model in evaluation mode
    # model.load_state_dict(torch.load('./maskRCNN_vehicleped_model_blurstitched.pt'))
    # model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

    masks_mrcnn = prediction[0]['masks'].cpu().detach().numpy()

    # gc.collect()
    # torch.cuda.empty_cache()

    # # Define the file path where you want to save the image
    # output_path = "saved_image.png"  # Change the filename and extension as needed

    # Save the image
    # img = img * 255  # Assuming the image values are in the range [0, 1], scale them to [0, 255]
    img = img.permute(1, 2, 0)  # Change tensor format to HxWxC and convert to bytes
    output_image = Image.fromarray((img.detach().cpu().numpy()*255).astype(np.uint8))
    os.makedirs(output+m+"/", exist_ok=True)
    os.makedirs(output+m+"/masks/", exist_ok=True)
    output_image.save(output+m+"/"+m+".png")

    # model_path = './FastSAM.pt'
    # img_path = f'./{output_path}'
    # iou = 0.9
    # text_prompt = None
    # conf = 0.6
    # output = "./output/"
    # randomcolor = True
    # point_prompt = "[[0,0]]" #help="[[x1,y1],[x2,y2]]"
    # point_label = "[0]" #help="[1,0] 0:background, 1:foreground"
    # box_prompt = "[[0,0,0,0]]" #help="[[x,y,w,h],[x2,y2,w2,h2]] support multiple boxes"
    # better_quality = False
    # device = torch.device(
    #         "cuda"
    #         if torch.cuda.is_available()
    #         else "mps"
    #         if torch.backends.mps.is_available()
    #         else "cpu"
    #     )
    # retina = True
    # withContours = False #help="draw the edges of the masks"

    # model_sam = FastSAM(model_path)
    # point_prompt = ast.literal_eval(point_prompt)
    # box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(box_prompt))
    # point_label = ast.literal_eval(point_label)
    input = Image.open(output+m+"/"+m+".png")
    input = input.convert("RGB")
    everything_results = model_sam(
        input,
        device=device,
        retina_masks=retina,
        imgsz=max(np.shape(input)),
        conf=conf,
        iou=iou
        )
    bboxes = None
    points = None
    point_label = None
    prompt_process = FastSAMPrompt(input, everything_results, device=device)
    if box_prompt[0][2] != 0 and box_prompt[0][3] != 0:
            ann = prompt_process.box_prompt(bboxes=box_prompt)
            bboxes = box_prompt
    elif text_prompt != None:
        ann = prompt_process.text_prompt(text=text_prompt)
    elif point_prompt[0] != [0, 0]:
        ann = prompt_process.point_prompt(
            points=point_prompt, pointlabel=point_label
        )
        points = point_prompt
        point_label = point_label
    else:
        ann = prompt_process.everything_prompt()
    # prompt_process.plot(
    #     annotations=ann,
    #     output_path=output+img_path.split("/")[-1],
    #     bboxes = bboxes,
    #     points = points,
    #     point_label = point_label,
    #     withContours=withContours,
    #     better_quality=better_quality,
    # )

    masks_sam = ann.cpu().detach().numpy()

    masks_mrcnn = masks_mrcnn.reshape(masks_mrcnn.shape[0], masks_mrcnn.shape[2], masks_mrcnn.shape[3])

    # Initialize an empty list to store the selected masks from 'masks_sam'
    selected_masks2 = []

    # Iterate through masks in 'masks_sam'
    for mask2 in masks_sam:
        for mask1 in masks_mrcnn:
            # Check if any mask in 'masks_mrcnn' intersects with the current 'mask2'
            intersects = np.logical_and(mask1, mask2)

            # If any intersection is found, keep the current 'mask2'
            if np.any(intersects):
                u = np.unique(intersects, return_counts=True)
                if u[1][1]/np.sum(mask2)>0.9:
                    selected_masks2.append(mask2)
                    break

    # Convert the selected masks from 'selected_masks2' back to a numpy array
    selected_masks2 = np.array(selected_masks2)
    for ii,jj in enumerate(selected_masks2):
        Image.fromarray((jj*255).astype(np.uint8)).save(output+m+"/masks/"+f"{ii}.png")

    prompt_process.plot(
        annotations=torch.tensor(selected_masks2),
        output_path=output+m+"/"+m+"_selectedmaskedimg.png",
        bboxes = bboxes,
        points = points,
        point_label = point_label,
        withContours=withContours,
        better_quality=better_quality,
    )

    # print(selected_masks2)
    print(np.shape(masks_mrcnn))
    print(np.shape(masks_sam))
    print(np.shape(selected_masks2))

    gc.collect()
    torch.cuda.empty_cache()