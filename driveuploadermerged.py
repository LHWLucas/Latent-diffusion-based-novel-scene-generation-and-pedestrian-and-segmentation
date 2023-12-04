from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import glob
from tqdm import tqdm

gauth = GoogleAuth()
gauth.CommandLineAuth()
drive = GoogleDrive(gauth)

for j in glob.glob("./outpainted_images3blendedmerged/*"):
	gfile = drive.CreateFile({'parents': [{'id': "1pRQnlg-lHXPb71mC6IT4Ozmi7BaGHUny"}]})
	gfile.SetContentFile(j)
	gfile.Upload()