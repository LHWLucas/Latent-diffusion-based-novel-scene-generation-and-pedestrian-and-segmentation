# from pydrive2.auth import GoogleAuth
# from pydrive2.drive import GoogleDrive
# import glob
# from tqdm import tqdm

# gauth = GoogleAuth()
# gauth.CommandLineAuth()
# drive = GoogleDrive(gauth)
# top_list = drive.ListFile({'q': "'16njQ1yXCNzbdXg_yNIjD1neY2MERs_2T' in parents and trashed=false"}).GetList()

# for i in tqdm(glob.glob('./base_stable_diffusion/*')):
# 	for j in glob.glob(i+"/outpainted_images3blended/*"):
# 		for file in top_list:
# 			# print('title: %s, id: %s' % (file['title'], file['id']))
# 			if file['title']==j.split("/")[2]:
# 				gfile = drive.CreateFile({'parents': [{'id': file['id']}]})
# 				gfile.SetContentFile(j)
# 				gfile.Upload()

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import glob
from tqdm import tqdm
import argparse  # Import the argparse module

# Define a function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Folder number")
    parser.add_argument(
        "--folder",
        type=int, 
        required=True
    )
    return parser.parse_args()

# Parse the command-line arguments
args = parse_args()
# print(args.folder)

gauth = GoogleAuth()
gauth.CommandLineAuth()
drive = GoogleDrive(gauth)
top_list = drive.ListFile({'q': "'16njQ1yXCNzbdXg_yNIjD1neY2MERs_2T' in parents and trashed=false"}).GetList()

for i in tqdm(glob.glob(f'./base_stable_diffusion/{args.folder}/outpainted_images3blended/*')):
    # for j in glob.glob(i+"/outpainted_images3blended/*"):
    for file in top_list:
        if file['title'] == i.split("/")[2]:
            gfile = drive.CreateFile({'parents': [{'id': file['id']}]})
            gfile.SetContentFile(i)
            gfile.Upload()
