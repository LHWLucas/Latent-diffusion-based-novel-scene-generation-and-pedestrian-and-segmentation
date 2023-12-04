import immutabledict
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import camera_segmentation_utils
import glob
import matplotlib.patches as patches
import cv2
from tqdm import tqdm
import tensorflow as tf
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from google.protobuf.json_format import MessageToDict

num_images = 3
count = 0
count_right = 0

for g,f in tqdm(enumerate(sorted(glob.glob("./data/waymo-od/*.tfrecord"))),desc="creating test images from tfrecords"):
  if g<200:
    continue
  if g==300:
    break

  with open("outpaint_testing_txt_path.txt", "a") as blend_file:
    blend_file.write(f + "\n")

  frames_list = []
  dataset = tf.data.TFRecordDataset(f, compression_type='')
  sequence_id = None
  for data in dataset:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    # Save frames which contain CameraSegmentationLabel messages. We assume that
    # if the first image has segmentation labels, all images in this frame will.
    if frame.images[0].camera_segmentation_label.panoptic_label:
      frames_list.append(frame)
      if sequence_id is None:
        sequence_id = frame.images[0].camera_segmentation_label.sequence_id

  for ind,frame in enumerate(frames_list):
    images = []

    for index, image in enumerate(frame.images):
      if open_dataset.CameraName.Name.Name(image.name)=='FRONT':
        images.append(cv2.resize(cv2.cvtColor(tf.image.decode_jpeg(image.image).numpy(), cv2.COLOR_RGB2BGR),(512,512)))
        cv2.imwrite(f'./FRONT_BLEND_TESTING/{count}.png', images[-1])
        count+=1

      if open_dataset.CameraName.Name.Name(image.name)=='FRONT_RIGHT':
        images.append(cv2.resize(cv2.cvtColor(tf.image.decode_jpeg(image.image).numpy(), cv2.COLOR_RGB2BGR),(512,512)))
        cv2.imwrite(f'./FRONTRIGHT_BLEND_TESTING/{count_right}.png', images[-1])
        count_right+=1

    images = np.squeeze(np.array(images))