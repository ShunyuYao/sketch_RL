
# coding: utf-8

# In[1]:


import torch
import pandas as pd
import numpy as np
import os
import sys
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt


sys.path.append('../')
from util.base_dataset import FaceEdgeFolder
from baselines.models.C3D.C3D_model import C3DVidPredNet
import baselines.models.C3D.networks as network


# In[4]:


Epoch = 2000
BatchSize = 4
ImgSize = (128, 128)
Shuffle = True
NumWorkers = 4
UseLsgan = True


# In[5]:


df = pd.read_csv('../dataset/emotion_id_clean.csv')

train_dataset = FaceEdgeFolder('../dataset/face_landmark/front_data/preprocess_edgeImgs/', df, samp_len=7,
                        transform=transforms.Compose((transforms.Resize(ImgSize),
                                                     transforms.ToTensor())),
                              samp_interv = [6, 8, 10])
dataloader = DataLoader(train_dataset, batch_size=BatchSize,
                       shuffle=Shuffle, num_workers=NumWorkers)


# In[6]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tensor_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
model = C3DVidPredNet()
disc_model = network.define_D_patch(7, 64, 3, gpu_ids=[0])
model.to(device)

dist_loss = nn.L1Loss()
GAN_loss = network.GANLoss(UseLsgan, tensor=tensor_type)
optimizer_G = torch.optim.Adam(model.parameters())
optimizer_D = torch.optim.Adam(disc_model.parameters())


# In[ ]:


model.train()
optimizer_G = torch.optim.Adam(model.parameters(), 1e-4)
optimizer_D = torch.optim.Adam(disc_model.parameters(), 1e-4)
with torch.set_grad_enabled(True):
    for epoch in range(Epoch):
        print("Epoch {}/{}".format(epoch+1, Epoch))
        print('-' * 10)

        running_loss = 0.0
        for i, samp in enumerate(dataloader):
            train_tensor = samp['first_img'].to(device)
            true_tensor = samp['succeed_imgs'].to(device)
            emotion_tensor = samp['emotion_label'].to(device)
            # print(emotion_tensor.shape)

            model.zero_grad()
            pred_tensor = model(train_tensor, emotion_tensor)

            fake_pairs = torch.cat((pred_tensor.squeeze(1).detach(), train_tensor.squeeze(2)), dim=1)
            pred_fake_pool = disc_model(fake_pairs)
            loss_D_fake = GAN_loss(pred_fake_pool, False)

            true_pairs = torch.cat((true_tensor.squeeze(1), train_tensor.squeeze(2)), dim=1)
            pred_true = disc_model(true_pairs)
            loss_D_real = GAN_loss(pred_true, True)

            fake_true_pairs = torch.cat((pred_tensor.squeeze(1), train_tensor.squeeze(2)), dim=1)
            pred_fake = disc_model(fake_true_pairs)
            loss_G_C3D = GAN_loss(pred_fake, True)

            loss_dist = dist_loss(pred_tensor, true_tensor) * 2

            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_G = loss_G_C3D + loss_dist

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            if i % 2 == 0:
                print('loss D_real: {:.3f}, loss D_fake: {:.3f}, loss G: {:.3f}, mse loss: {:.3f}'                       .format(loss_D_real, loss_D_fake, loss_G_C3D, loss_dist))



# In[ ]:


true_tensor.squeeze(1).shape


# In[8]:


len(train_dataset) * 5


# In[22]:


input_tensor.shape


# In[10]:


Image.fromarray((input_tensor[0,0,0]*255).cpu().detach().numpy().astype('uint8'), 'L')


# In[11]:


Image.fromarray((true_tensor[0,0,5]*255).cpu().detach().numpy().astype('uint8'), 'L')


# In[18]:


Image.fromarray((pred_tensor[0,0,1]*255).cpu().detach().numpy().astype('uint8'), 'L')


# In[13]:


torch.cat((condition_label.view(BatchSize, 1, 1, *ImgSize), train_img), dim=1).size()
