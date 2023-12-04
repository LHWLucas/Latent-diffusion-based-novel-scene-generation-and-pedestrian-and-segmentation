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
top_list = drive.ListFile({'q': "'15QlW7xNlrg82LFyP8efTRQ5AIvckzWxX' in parents and trashed=false"}).GetList()

for i in tqdm(glob.glob(f'./finetuned_stable_diffusion_outpainting/{args.folder}/outpainted_images3blended/*')):
    # for j in glob.glob(i+"/outpainted_images3blended/*"):
    for file in top_list:
        if file['title'] == i.split("/")[2]:
            gfile = drive.CreateFile({'parents': [{'id': file['id']}]})
            gfile.SetContentFile(i)
            gfile.Upload()
