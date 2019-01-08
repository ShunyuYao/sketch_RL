import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from PIL import Image

class FaceEdgeFolder(data.Dataset):
    """
    Output a seq of face motions
    """

    def __init__(self, root, df_folder2idx, samp_len = 6, samp_interv = [1, 4, 6, 10],
                 transform=transforms.ToTensor(), target_transform=None):
        self.root = root
        self.root_folder_names = sorted(os.listdir(root))
        self.df_folder2idx = df_folder2idx
        self.samp_len = samp_len
        self.samp_interv = samp_interv
        self.transform = transform
        self.target_transform = target_transform
        self.img_num_folder = self._count_img_num(root, self.root_folder_names)

    def __len__(self):
        return len(self.root_folder_names)

    def __getitem__(self, folder_idx):
        succeed_tensors = torch.Tensor([])
        root_folder_name = self.root_folder_names[folder_idx]
        samp_complete = False
        img_names = sorted(os.listdir(os.path.join(self.root, root_folder_name)))
        img_num =  self.img_num_folder[folder_idx]
        emotion_label = self.df_folder2idx[self.df_folder2idx.iloc[:, 0] == root_folder_name].iloc[0, 1]

        # sample the clip
        while not samp_complete:
            start_img_idx = np.random.randint(0, img_num - self.samp_len)
            samp_intervs = list(np.random.choice(self.samp_interv, size=len(self.samp_interv), replace=False))
            for interv in samp_intervs:
                if (start_img_idx + interv * self.samp_len) <= img_num:
                    samp_complete = True
                    self.selected_interv = interv
                break

        for i, img_idx in enumerate(range(start_img_idx, start_img_idx + interv * self.samp_len, interv)):
            img = Image.open(os.path.join(self.root, root_folder_name, img_names[img_idx]))
            img_tensor = self.transform(img)
            img_tensor = img_tensor.unsqueeze(0)
            if i == 0:
                first_tensor = img_tensor
            else:
                succeed_tensors = torch.cat((succeed_tensors, img_tensor), 1)

        sample = {'first_img':first_tensor, 'succeed_imgs':succeed_tensors, 'emotion_label':emotion_label}

        return sample


    def _count_img_num(self, root, root_folder_names):
        img_nums = []
        for name in root_folder_names:
            img_num = len(os.listdir(os.path.join(root, name)))
            img_nums.append(img_num)

        return img_nums
