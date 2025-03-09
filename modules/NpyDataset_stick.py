import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class NpyDataset(Dataset):
    def __init__(self, root, folders = None, transform=None, demo = False):
        self.root = root
        self.transform = transform
        self.folders = folders

        # Loop through subfolders within the root directory and get images
        self.files = []
        
        # Initialize a mapping between original labels and new indices
        self.label_mapping = {}
        index = 0

        if demo == False:
            for dirpath, dirnames, filenames in os.walk(root):    
                for filename in filenames:
                    if filename.endswith('.jpeg') or filename.endswith('.jpg'):
                        file_path = os.path.join(dirpath, filename)
                        # get label for individual
                        start_index = dirpath.find('{') + 1
                        end_index = dirpath.find('}')
                        label = int(dirpath[start_index:end_index])
        
                                                                                
                        if ((self.folders is not None) and (label in self.folders)) or (self.folders is None):   
                            if label not in self.label_mapping:
                                # Assign a new index to the label
                                self.label_mapping[label] = index
                                index += 1
                            self.files.append((file_path, self.label_mapping[label]))
        else: #demo = True
            for participant in os.listdir(root):
                if participant == '.DS_Store':
                        continue
                participant_path = os.path.join(root, participant)
                label = participant
                for env_folder in os.listdir(participant_path):
                    if env_folder == '.DS_Store':
                        continue
                    env_path = os.path.join(participant_path, env_folder)
                    for filename in os.listdir(env_path):
                        if filename.endswith('.DS_Store'):
                            continue
                        file_path = os.path.join(env_path, filename)
                        if label not in self.label_mapping:
                            # Assign a new index to the label
                            self.label_mapping[label] = index
                            index += 1
                        self.files.append((file_path, self.label_mapping[label]))


                        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file_name, label = self.files[idx]

        
        file_path = os.path.join(self.root, file_name)

        image_data = Image.open(file_path)

        if self.transform:
            image_data = self.transform(image_data)

        
        return image_data, label
    
    def get_image_path(self, idx):
        return self.files[idx]
    
    def create_path(file_path):
        parts = file_path.split("/")
        path = "/".join(parts[-3:])
        return path