import os
import shutil
import numpy as np

src_path = '../../dataset/face_landmark/all_imgs_canny/all_edges_img'
train_save_path = '../../dataset/face_landmark/train_edges_img'
test_save_path = '../../dataset/face_landmark/test_edges_img'
src_names = os.listdir(src_path)
total_num = len(src_names)
test_ratio = 0.2
ids = []

for name in src_names:
    id = name.split('-')[0]
    vid_num = name.split('-')[1]
    ids.append(id)

ids_set = set(ids)
ids = list(ids_set)
print('total ids: ', ids)

id_train = []
id_test = []
for id in ids:
    if np.random.rand() < test_ratio:
        id_test.append(id)
    else:
        id_train.append(id)

print('train id: ', id_train)
print(' test id: ', id_test)
if len(id_test) >= 5:

    if not os.path.isdir(train_save_path):
        os.makedirs(train_save_path)

    if not os.path.isdir(test_save_path):
        os.makedirs(test_save_path)

    for i, name in enumerate(src_names):
        id = name.split('-')[0]
        if id in id_train:
            print('train folder {}'.format(name))
            shutil.copytree(os.path.join(src_path, name), os.path.join(train_save_path, name))
        elif id in id_test:
            print('test folder {}'.format(name))
            shutil.copytree(os.path.join(src_path, name), os.path.join(test_save_path, name))
