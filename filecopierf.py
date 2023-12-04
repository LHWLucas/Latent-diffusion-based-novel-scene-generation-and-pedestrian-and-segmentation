import os
import shutil
from tqdm import tqdm
# Source and destination folder paths
source_folder = './FRONT_BLEND/'
destination_folder = './trainingdataset/train/'

# Iterate over files in the source folder
for filename in tqdm(os.listdir(source_folder)):
    if filename.endswith('.png'):
        # new_filename = filename.replace('.png', 'fr.png')
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)
        shutil.copy2(source_file, destination_file)