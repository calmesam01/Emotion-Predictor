import glob
import pandas as pd
import shutil
import os

def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)

dataset_folder = "IMFDB_final/*"
save_data = "training_images"

all_actors = glob.glob(dataset_folder)

for i in all_actors:
    for j in glob.glob(os.path.join(i, '*')):
        try:
            annotations = glob.glob(os.path.join(j, '*.txt'))
            images = glob.glob(os.path.join(j, 'images/*'))
            df = pd.read_csv(annotations[0], sep='\t', header=None)
            for i in images:
                expr = tuple(df[df[2] == i.split('/')[-1]][11])[0]
                save_path = os.path.join(save_data, expr)
                create_directory(save_path)
                shutil.copy(i, save_path)
        except Exception as e:
            pass