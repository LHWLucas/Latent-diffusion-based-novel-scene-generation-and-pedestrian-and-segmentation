from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import glob
from tqdm import tqdm
import os

gauth = GoogleAuth()
gauth.CommandLineAuth()
drive = GoogleDrive(gauth)
top_list = drive.ListFile({'q': "'1z4T9_TxxjG4VTa5NqjMEYsNchXCsEHhu' in parents and trashed=false"}).GetList()

for i in tqdm(sorted(glob.glob("./output2/*"))):
	for j in glob.glob(i+"/*"):
		if ".png" in j:
			for file in top_list:
				if file['title'] == os.path.basename(i):
					if len(drive.ListFile({'q': f"'{file['id']}' in parents and trashed=false"}).GetList())<3:
						gfile = drive.CreateFile({'parents': [{'id': file['id']}]})
						gfile.SetContentFile(j)
						gfile.Upload()
		else:
			for file in top_list:
				if file['title'] == os.path.basename(i):
					if len(drive.ListFile({'q': f"'{file['id']}' in parents and trashed=false"}).GetList())<3:
						f = {
							'title': "masks",
							# Define the file type as folder
							'mimeType': 'application/vnd.google-apps.folder',
							# ID of the parent folder        
							'parents': [{"kind": "drive#fileLink", "id": file['id']}]
						}
						gfile = drive.CreateFile(f)
						# gfile.SetContentFile(j)
						gfile.Upload()
						file_list = drive.ListFile({'q': f"'{file['id']}' in parents and trashed=false"}).GetList()

						for k in glob.glob(j+"/*"):
							if "masksmrcnn" not in k:
								if ".png" in k:
									for file in file_list:
										if file['title'] == "masks":
											gfile = drive.CreateFile({'parents': [{'id': file['id']}]})
											gfile.SetContentFile(k)
											gfile.Upload()