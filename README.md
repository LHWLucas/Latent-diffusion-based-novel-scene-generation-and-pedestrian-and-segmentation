# latent-diffusion-based-novel-scene-generation-and-pedestrian-and-vehicle-segmentation

WAYMO DATASET WITH STABLE DIFFUSION AND MASKRCNN
https://colab.research.google.com/drive/1pR2BD0CH-pzen7yRJr2-GLrC_LwPJoH9?usp=sharing

Objective: here the objective was to segment-predict the pedestrians/ people and vehicles in the right view given an image for the front view

example 1 with stable diffusion outpainted right side:
![img](https://github.com/LHWLucas/latent-diffusion-based-novel-scene-generation-and-pedestrian-and-vehicle-segmentation/assets/89898376/c79cc8a9-c37b-438c-b5e5-0e8a4f896756)
segmented example 1:
![maskedimg](https://github.com/LHWLucas/latent-diffusion-based-novel-scene-generation-and-pedestrian-and-vehicle-segmentation/assets/89898376/4d62caf3-a88b-49fa-812b-72a2395402fe)
mask stitched example 1:
![image](https://github.com/LHWLucas/latent-diffusion-based-novel-scene-generation-and-pedestrian-and-vehicle-segmentation/assets/89898376/5a514172-f2c7-45f2-92b7-3d56ca56ea58)

example 2 with stable diffusion outpainted right side:
![img](https://github.com/LHWLucas/latent-diffusion-based-novel-scene-generation-and-pedestrian-and-vehicle-segmentation/assets/89898376/cd64af55-50d2-4271-b55f-1a33d57ba10b)
segmented example 2:
![maskedimg](https://github.com/LHWLucas/latent-diffusion-based-novel-scene-generation-and-pedestrian-and-vehicle-segmentation/assets/89898376/c662ef0b-f7b5-4611-9848-68cb3a04726c)
mask stitched example 2:
![image](https://github.com/LHWLucas/latent-diffusion-based-novel-scene-generation-and-pedestrian-and-vehicle-segmentation/assets/89898376/0b5a1bd7-596c-43fb-af67-b08fe1c860ee)

example 3 with stable diffusion outpainted right side:
![img](https://github.com/LHWLucas/latent-diffusion-based-novel-scene-generation-and-pedestrian-and-vehicle-segmentation/assets/89898376/72dd569f-2d20-4220-8ab6-8c36a3fdf6f5)
segmented example 3:
![maskedimg](https://github.com/LHWLucas/latent-diffusion-based-novel-scene-generation-and-pedestrian-and-vehicle-segmentation/assets/89898376/fc8dbe8f-de29-4111-bc70-6592c9ac89ef)
mask stitched example 3:
![image](https://github.com/LHWLucas/latent-diffusion-based-novel-scene-generation-and-pedestrian-and-vehicle-segmentation/assets/89898376/3e897e2d-a5dd-422d-b30b-b41d90a403c8)



Scripts on University Server/Cluster - Here outline each key scripts I created and what it does and are listed in chronological order from starting to finish. Also utilised diffusers library (https://github.com/huggingface/diffusers/) for certain training procedures of inpaint stable diffusion model:
   
•	processingtfrecords_FRONT_FRONTRIGHT.py : create dataset from waymo open dataset first 200 tfrecords for training purpose of stable diffusion inpaint model

•	blender.py : stitching and blending the newly created front and front right images from first 200 tfrecords

•	accelerate launch ./diffs/examples/research_projects/dreambooth_inpaint/train_dreambooth_inpaint.py  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-inpainting"   --instance_data_dir="./training_imgs/"  --output_dir="./chkpt_outpaint3/" --instance_prompt="photo of a waymodrivscene driving scene"  --resolution=512  --train_batch_size=4  --gradient_accumulation_steps=1 --learning_rate=5e-6  --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=3000 : training inpaint model on first 200 tfrecord images that were selected after stitching

•	testdreamboothmodels.py : testing the newly trained inpaint model for outpainting waymo style images

•	processingtfrecords_FRONT_FRONTRIGHT_testingimages.py : create dataset from waymo open dataset between 200:300 indexed tfrecords for testing purpose of stable diffusion inpaint model

•	blender_outpaint_testing.py : creating blended stitched dataset from the testing dataset

•	testbasesd.py : testing base inpaint sd model without any fine tuning on outpainting task 

•	imagecaptioner.py : added manual prompts by loading from a text file with base inpaint sd model to see changes 

•	clipcaptioner.py :  used clip model for generating prompts 

•	metadatcreator.py : merges the front and front right csv files with adding front or front right as keyword to each promt 

•	testmerged.py : tested the merged model for outpainting but got the worst results 

•	maskrcnndreambooth.py :  SAM (segment anything model) in conjunction with maskrcnn model 

•	gsutil.docx : gsutil commands for getting the tfrecord datasets from waymo open dataset

•	testbasesdfinetuned.py : outpainting using base sd model and merged model, failed results weren’t good enough

•	accelerate launch ./diffs/examples/text_to_image/train_text_to_image.py --pretrained_model_name_or_path="stabilityai/stable-diffusion-2" --train_data_dir="./trainingdataset/" --use_ema --resolution=512 --center_crop --random_flip  --train_batch_size=1 --gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=15000 --learning_rate=1e-05 --mixed_precision="fp16"  --max_grad_norm=1 --lr_scheduler="constant" --lr_warmup_steps=0  --output_dir="finetuning" : to train inpaint model with the provided script that’s online but this gave error and didn’t even trained the inpaint model so we used our training script instead

•	train_text_to_image_inpaintmodel.py : script created to train fine tune the inpaint model without dreambooth or LoRA

•	accelerate launch ./diffs/examples/text_to_image/train_text_to_image_inpaintmodel.py --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-inpainting" --train_data_dir="./trainingdataset/" --use_ema --resolution=512 --center_crop --random_flip --train_batch_size=1 --gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=15000 --learning_rate=1e-05 --mixed_precision="fp16" --max_grad_norm=1 --lr_scheduler="constant" --lr_warmup_steps=0 --output_dir="finetuning2" --instance_data_dir="./trainingdataset/" --cache_dir="./kkk" : training the sd inpaint model without dreamboth or LoRA training procedure using the script we created

•	testfinetunedsdoutpaint.py : testing the trained fine tuned sd inpaint model using custom prompts from txt file with “front right” keyword appended to each prompt

•	testfinetunedsdoutpaintclipcaptioned.py : for the next test using our fine tuned sd inpaint model I replace txt file prompts with clip produced automatic prompts

•	testfinetunedsdoutpaintclipcaptionedwithdiscarding.py :  tested our new discarding algorithm 
 
 ![image](https://github.com/LHWLucas/latent-diffusion-based-novel-scene-generation-and-pedestrian-and-vehicle-segmentation/assets/89898376/fa74eed0-0778-49fc-b1f5-2b5f0b9b3828)

prompt: "street view scenery, buildings, plants and trees, pedestrians, front right"
negative prompt: "3d, cartoon, animated, distorted, unrealistic, disfigured, drawing, painting"
so for this prompt combination created 10 images with suffix new in "clipcaptionedinpaintfinetunedwithdiscarding2" folder
this gave the best results of all the experiments we have completed

•	maskrcnninpaintfinetuned.py : as we needed one mask per object not multiple.
