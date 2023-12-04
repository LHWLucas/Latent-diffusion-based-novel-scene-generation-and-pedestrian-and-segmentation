import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
import imutils
import sys
from imutils import paths
from scipy.ndimage.morphology import distance_transform_edt
import os
import glob
cv2.ocl.setUseOpenCL(False)

def cropper(ref_img):
  # Load the image
  image = ref_img

  # Get the width and height of the image
  height, width = image.shape[:2]

  # Get the left quarter of the image
  left_quarter_image = image[:, :width // 4]

  # Convert the left quarter image to grayscale
  gray_image = cv2.cvtColor(left_quarter_image, cv2.COLOR_BGR2GRAY)

  # Apply thresholding to get binary image
  _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

  # Find contours in the binary image
  contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Initialize variables to store the top and bottom border locations
  top_border = height
  bottom_border = 0

  # Find the top and bottom border locations
  for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)
      if y < top_border:
          top_border = y
      if y + h > bottom_border:
          bottom_border = y + h

  # Crop the borders from the original image
  cropped_image = image[top_border:bottom_border, :]
  return cropped_image

def goodMatches(ref_des, des, matcher):
    matches = matcher.knnMatch(des, ref_des, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    return good

def createFrame(ref_img, img):
    top = int(img.shape[0])
    bottom = top
    right = int(img.shape[1])
    left = right
    ref_img = cv2.copyMakeBorder(ref_img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)
    return ref_img

def blending(I1, I2):
    w1 = distance_transform_edt(I1)
    w1 = np.divide(w1, np.max(w1))
    w2 = distance_transform_edt(I2)
    w2 = np.divide(w2, np.max(w2))
    I_blended = np.add(np.multiply(I1, w1), np.multiply(I2, w2))
    w_tot = w1 + w2
    I_blended = np.divide(I_blended, w_tot, out=np.zeros_like(I_blended), where=w_tot != 0).astype("uint8")
    return I_blended

def cropping(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = img[y:y + h, x:x + w]
    return crop

def best_parameters(ref_img, images, matcher):

    # contains the matching points for the number with the maximum number of correspondencies
    max_matching = []

    # contains the parameters to retrieve the image with greater number of correspondencies
    # and the ref_img with its new surrounding black frame
    best_params = {}

    for i in tqdm(range(len(images))):
        img = images[i]
        (kp, des) = points[i]
        # create a frame where we put the ref_img
        ref_img = createFrame(ref_img, img)

        gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        # to compute the keypoints and descriptors for the reference image with the augmented border
        (ref_kp, ref_des) = sift.detectAndCompute(gray, None)

        if len(ref_kp) >= 2:
            matches = goodMatches(ref_des=ref_des, des=des, matcher=flann)
            if len(max_matching) < len(matches):
                max_matching = matches
                best_params = {
                    "matches" : max_matching,
                    "ref_kp" : ref_kp,
                    "ref_des" : ref_des,
                    "idx" : i,
                    "ref_img" : ref_img
                    }
        else:
            print("[ERROR] Reference image has not enough keypoints")
            exit()
        ref_img = cropping(ref_img)
    return best_params

SAVE_PATH = "./stitched"
os.makedirs(SAVE_PATH, exist_ok=True)
stitch_imgs = glob.glob(SAVE_PATH+"/*")

# load all images
src_imgs = sorted(glob.glob("./FRONT_BLEND/*"))
for ii in tqdm(src_imgs):
    print(ii, ii.replace("FRONT_BLEND","FRONTRIGHT_BLEND"), ii.replace("FRONT_BLEND","stitched"))
    print(ii.replace("FRONT_BLEND","stitched") not in stitch_imgs)
    if ii.replace("FRONT_BLEND","stitched") not in stitch_imgs:
        try:
            imagePaths = [ii, ii.replace("FRONT_BLEND","FRONTRIGHT_BLEND")]
            images = []

            # For every given path we read the corresponding image and we put it in the list of images
            for path in imagePaths:
                img = cv2.imread(path)
                images.append(img)

            sift = cv2.SIFT_create()
            points = []

            for i, img in enumerate(images):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                kp, des = sift.detectAndCompute(gray, None)
                gray = cv2.drawKeypoints(gray, kp, gray, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                points.append((kp, des))

            # I take the first image as a reference for the matching with the others
            first = 0  #  (len(images)-1)//2
            ref_img = images.pop(first)
            del points[first]

            # we use this to compute the matches between pair of images
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=2)
            search_params = dict() #  dict(checks=500)
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            MIN_MATCH_COUNT = 10

            while len(images) > 0:
                max_matching = []
                best_params = {}
                for i in tqdm(range(len(images))):
                    img = images[i]
                    (kp, des) = points[i]
                    ref_img = createFrame(ref_img, img)  # create a frame where we put the ref_img
                    gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
                    (ref_kp, ref_des) = sift.detectAndCompute(gray, None)
                    if len(ref_kp) >= 2:
                        matches = goodMatches(ref_des=ref_des, des=des, matcher=flann)
                        if len(max_matching) < len(matches):
                            max_matching = matches
                            best_params = {
                                "matches" : max_matching,
                                "ref_kp" : ref_kp,
                                "ref_des" : ref_des,
                                "idx" : i,
                                "ref_img" : ref_img
                            }
                    else:
                        print("[WARNING] Reference image has not enough keypoints")
                        sys.exit()
                    ref_img = cropping(ref_img)

                ref_img = best_params["ref_img"]
                (ref_kp, ref_des) = (best_params["ref_kp"], best_params["ref_des"])
                i = best_params["idx"]
                matches = best_params["matches"]
                width = ref_img.shape[1]
                height = ref_img.shape[0]
                if len(matches) >= MIN_MATCH_COUNT:
                    img = images.pop(i)
                    (kp, des) = points.pop(i)
                    src_pts = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1,1,2) # source points
                    dst_pts = np.float32([ref_kp[m.trainIdx].pt for m in matches]).reshape(-1,1,2) # destination points
                    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    warped_img = cv2.warpPerspective(img, H, (width, height))
                    ref_img = blending(ref_img, warped_img)
                    ref_img = cropping(ref_img)
                    ref_img = cropper(ref_img)
                    cv2.imwrite(f'{ii.replace("FRONT_BLEND","stitched")}', ref_img)
                else:
                    print(f"[WARNING] not enough matches found for image_{i}")
                    break
            # break
        except Exception as e:
          pass