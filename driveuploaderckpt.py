from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
# import glob
# gauth = GoogleAuth()
# gauth.LoadClientConfigFile('./client_secret_95146630750-sclh21du0rji3814t4hig6ieuulagq4i.apps.googleusercontent.com.json') # fails here
# drive = GoogleDrive(gauth)
# gfile = drive.CreateFile()
# gfile.SetContentFile('./stitched.zip')
# gfile.Upload()

gauth = GoogleAuth()
gauth.CommandLineAuth()
drive = GoogleDrive(gauth)

# for i in glob.glob('./stitched/*'):
gfile = drive.CreateFile()
gfile.SetContentFile('./finetuning/finetuned.ckpt')
gfile.Upload()