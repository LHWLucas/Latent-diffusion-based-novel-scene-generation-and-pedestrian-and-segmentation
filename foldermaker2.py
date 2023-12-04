from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import glob
from tqdm import tqdm
import os

gauth = GoogleAuth()
gauth.CommandLineAuth()
drive = GoogleDrive(gauth)

for i in tqdm(glob.glob("./output2/*")):
	f = {
        'title': os.path.basename(i),
        # Define the file type as folder
        'mimeType': 'application/vnd.google-apps.folder',
		# ID of the parent folder        
		'parents': [{"kind": "drive#fileLink", "id": "1z4T9_TxxjG4VTa5NqjMEYsNchXCsEHhu"}]
    }
	gfile = drive.CreateFile(f)
	# gfile.SetContentFile(j)
	gfile.Upload()