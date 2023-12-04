from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

gauth = GoogleAuth()
gauth.CommandLineAuth()
drive = GoogleDrive(gauth)
folder_id = ''

file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()

for file1 in file_list:
    if file1['title'] == 'finetuned_inpainting.inpainting.ckpt':
        file1.GetContentFile("./mergedckpt/"+file1['title'])
        # folder_id = file1['id']
        # print(folder_id)

# for i, file1 in enumerate(sorted(drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList(), key = lambda x: x['title']), start=1):
#     print('Downloading {} from GDrive ({}/{})'.format(file1['title'], i, len(file_list)))
#     file1.GetContentFile("./mergedckpt/"+file1['title'])