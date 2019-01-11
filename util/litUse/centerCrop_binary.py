import os
from PIL import Image
import numpy as np

src_path = '../../dataset/face_landmark/front_data/edge_imgs'
sav_path = '../../dataset/face_landmark/front_data/preprocess_edgeImgs'
Size = (128, 128)
midDirs = os.listdir(src_path)

def centerCrop(img):
    np_img = np.array(img)
    h, w = np_img.shape
    non_empty = np.where(np_img == True)
    mid_h = (non_empty[0].min() + non_empty[0].max()) // 2
    mid_w = (non_empty[1].min() + non_empty[1].max()) // 2
    half_w = w // 2
    img_crop = np_img[max(mid_h - half_w, 0): min(mid_h + half_w, h),
                      max(mid_w - half_w, 0): min(mid_w + half_w, w)]
    img_crop = Image.fromarray(img_crop, 'L').convert('1')

    return img_crop

for i, dirname in enumerate(midDirs):
    print('process dictory {}'.format(dirname))
    for filename in os.listdir(os.path.join(src_path, dirname)):
        img = Image.open(os.path.join(src_path, dirname, filename))
        img_crop = centerCrop(img.convert('1')).resize(Size)
        save_dirPath = os.path.join(sav_path, dirname)
        if not os.path.isdir(save_dirPath):
            os.makedirs(save_dirPath)

        img_crop.save(os.path.join(save_dirPath, filename))
