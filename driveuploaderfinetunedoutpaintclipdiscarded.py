from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import glob
from tqdm import tqdm

gauth = GoogleAuth()
gauth.CommandLineAuth()
drive = GoogleDrive(gauth)
# top_list = drive.ListFile({'q': "'1-3lW5C-RikzeEuk6w0Ubd5bihDKk2r-A' in parents and trashed=false"}).GetList()

for i in tqdm(glob.glob(f'./finetuned_stable_diffusion_outpainting6/outpainted_images3blended/*')):
    # for file in top_list:
    gfile = drive.CreateFile({'parents': [{'id': '15bccYX0pmcQph-s8iqRypKFo3ZdVcxjr'}]})
    gfile.SetContentFile(i)
    gfile.Upload()