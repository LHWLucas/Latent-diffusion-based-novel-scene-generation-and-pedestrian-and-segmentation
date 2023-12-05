# latent-diffusion-based-novel-scene-generation-and-pedestrian-and-vehicle-segmentation

WAYMO DATASET WITH STABLE DIFFUSION AND MASKRCNN
https://colab.research.google.com/drive/1pR2BD0CH-pzen7yRJr2-GLrC_LwPJoH9?usp=sharing
Objective: here the objective was to segment-predict the pedestrians/ people and vehicles in the right view given an image for the front view
1) Google Colab  - Intitially we started with the premise to train waymo open dataset for human and vehicle segmentation using MaskRCNN model using pytorch and torchvision. The results were pretty mediocre but not good enough. Then we experimented with data by utilizing front and front right images as inputs and targets comparing with the performance of front image as input and target made it clear that the performance with front input and target was vastly superior with front input and front right targets due to any model’s incapability to extrapolate out of distribution samples. We also tested a small CNN as an outpainting model which was good enough but at that time was really not the objective.
We also experimented by stitching front and front right images and then training that image as segmentation input for MaskRCNN. Here also we were stuck on stitching not matching where we were stitching the image so we successfully stitched the image with blending to tackle the discontinuous stitching issue using weighted blending algorithm in the video:  https://www.youtube.com/watch?v=D9rAOAL12SY&ab_channel=FirstPrinciplesofComputerVision 
After the blending results improved slightly, but to get better results more data was needed and hence need to create synthetic data arose. For creating new dataset, Stable Diffusion was chosen due it’s robust performance and easily fine tuning training procedures like dreambooth, LoRA, etc.
We used the dreambooth training procedure to train inpaint SD model on waymo dataset and then used it to create a scene comprising of front and front right scene in each image. There was again a problem of a seam discontinuity and the inpaint model was changing the whole image ever so slightly so the problem was a bug in the inpaint model itself which we resolved using another mask on the non outpainted area and overlay with the original image which replaces the changed area. The results weren’t visually pleasing and we decided to move to university server/cluster after this as we only used first 200 tfrecords altogether with mostly used by maskrcnn training, 200 only used for training inpaint model and some more couple thousand images to create the outpainted dataset.

2) Scripts on University Server/Cluster - Here we’ll outline each script we created and what it does and 	are listed in chronological order from starting to finish:
   
•	processingtfrecords_FRONT_FRONTRIGHT.py : we create dataset from waymo open dataset first 200 tfrecords for training purpose of stable diffusion inpaint model

•	blender.py : stitching and blending the newly created front and front right images from first 200 tfrecords

•	client_secret_95146630750-sclh21du0rji3814t4hig6ieuulagq4i.apps.googleusercontent.com.json : json file created after authenticating and creating a web service for app on localhost on google cloud dev for uploading and downloading on files directly to and from server and google drive.

•	settings.yaml : yaml for authorization settings for the google drive account using google cloud backend to work with pydrive2 library

•	credentials.json : json file with local authorization settings for pydrive2 and google drive

•	accelerate launch ./diffs/examples/research_projects/dreambooth_inpaint/train_dreambooth_inpaint.py  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-inpainting"   --instance_data_dir="./training_imgs/"  --output_dir="./chkpt_outpaint3/" --instance_prompt="photo of a waymodrivscene driving scene"  --resolution=512  --train_batch_size=4  --gradient_accumulation_steps=1 --learning_rate=5e-6  --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=3000 : training inpaint model on first 200 tfrecord images that were selected after stitching

•	testdreamboothmodels.py : testing the newly trained inpaint model for outpainting waymo style images

•	processingtfrecords_FRONT_FRONTRIGHT_testingimages.py : we create dataset from waymo open dataset between 200:300 indexed tfrecords for testing purpose of stable diffusion inpaint model

•	blender_outpaint_testing.py : creating blended stitched dataset from the testing dataset

•	testbasesd.py : testing base inpaint sd model without any fine tuning on outpainting task but results weren’t satisfactory

•	imagecaptioner.py : we added manual prompts by loading from a text file with base inpaint sd model to see changes in results but still results weren’t visually appealing but a slight improvement

•	clipcaptioner.py : we used clip model for generating prompts from input images for both front and front right images and generate these prompts automatically and save it as a image, prompt pair for both front and front right images separately

•	metadatcreator.py : merges the front and front right csv files with adding front or front right as keyword to each promt depending on the image is from front or front right csv

•	testmerged.py : we tested the merged model for outpainting but got the worst results as outpainted region was just random noise

•	maskrcnndreambooth.py : tried SAM (segment anything model) in conjunction with maskrcnn model to get crispier and visually appealing masks by keeping only those good clean masks from sam model which are intersecting the maskrcnn model’s masks by more than 90% pixel area wise ensuring we only keep those masks of pedestrians and vehicles which maskrcnn model was trained to recognise 

•	gsutil.docx : gsutil commands for getting the tfrecord datasets from waymo open dataset

•	testbasesdfinetuned.py : outpainting using base sd model and poor man’s outpainting technique as merged model failed but results weren’t good enough

•	accelerate launch ./diffs/examples/text_to_image/train_text_to_image.py --pretrained_model_name_or_path="stabilityai/stable-diffusion-2" --train_data_dir="./trainingdataset/" --use_ema --resolution=512 --center_crop --random_flip  --train_batch_size=1 --gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=15000 --learning_rate=1e-05 --mixed_precision="fp16"  --max_grad_norm=1 --lr_scheduler="constant" --lr_warmup_steps=0  --output_dir="finetuning" : to train inpaint model with the provided script that’s online but this gave error and didn’t even trained the inpaint model so we used our training script instead

•	train_text_to_image_inpaintmodel.py : script we created to train fine tune the inpaint model without dreambooth or LoRA

•	accelerate launch ./diffs/examples/text_to_image/train_text_to_image_inpaintmodel.py --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-inpainting" --train_data_dir="./trainingdataset/" --use_ema --resolution=512 --center_crop --random_flip --train_batch_size=1 --gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=15000 --learning_rate=1e-05 --mixed_precision="fp16" --max_grad_norm=1 --lr_scheduler="constant" --lr_warmup_steps=0 --output_dir="finetuning2" --instance_data_dir="./trainingdataset/" --cache_dir="./kkk" : training the sd inpaint model without dreamboth or LoRA training procedure using the script we created

•	testfinetunedsdoutpaint.py : testing the trained fine tuned sd inpaint model using custom prompts from txt file with “front right” keyword appended to each prompt

•	testfinetunedsdoutpaintclipcaptioned.py : for the next test using our fine tuned sd inpaint model we replace our txt file prompts with clip produced automatic prompts but results degraded with our hypothesis being that clip captions are somehow long and creating hallucinatory effect and confusion for the model

•	testfinetunedsdoutpaintclipcaptionedwithdiscarding.py : here we firstly kept the clip captioner and tested our new discarding algorithm which I described in our chat as follows:
for discontinuity i first converted image to grayscale
then took sum of all columns to get pixel sum intensities across the width of the image
and i checked  for intensity changes at the middle part and 3/4th part of the image as these are the places we do the outpainting and discontinuities are always there only
you can see the change in intensities in plot where there is discontinuity( a sharp peak) which does not happen in norma continuous images
now this change in intensity is what we find and match against a threshold
if threshold is weak then there will be a lot of false positives
if threshold is very strong we will be stuck in generation to get a good image for a long time
that's what we fixed last time with a threshold that is a good compromise between both
 
 ![image](https://github.com/LHWLucas/latent-diffusion-based-novel-scene-generation-and-pedestrian-and-vehicle-segmentation/assets/89898376/fa74eed0-0778-49fc-b1f5-2b5f0b9b3828)

Then we later removed the clip captions with this I mentioned in the chat:
prompt: "street view scenery, buildings, plants and trees, pedestrians, front right"
negative prompt: "3d, cartoon, animated, distorted, unrealistic, disfigured, drawing, painting"
so for this prompt combination created 10 images with suffix new in "clipcaptionedinpaintfinetunedwithdiscarding2" folder
this gave the best results of all the experiments we have completed

•	maskrcnninpaintfinetuned.py : as we have used maskrcnn with segment anything model we were getting part masks covering the object and we needed one mask per object not multiple so we fixed that by simply using which masks are inside some masks or 99% area of mask is inside the bigger mask or not and then we merged them to get good results finally
