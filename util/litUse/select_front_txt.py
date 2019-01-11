import os
import glob
import shutil
import pandas as pd

src_path = '/media/yaosy/办公/research300/sketch_rl/sketch_RL/dataset/face_landmark/all_data/all_imgs_canny/all_edges_img'
mov_path = '/media/yaosy/办公/research300/sketch_rl/sketch_RL/dataset/face_landmark/all_data/all_keypoints'
dst_path = '/media/yaosy/办公/research300/sketch_rl/sketch_RL/dataset/face_landmark/front_data/keypoints'
df = pd.read_csv('../../dataset/emotion_id.csv')
dirlist = os.listdir(src_path)
keyname = ''
for dirname in dirlist:
    if df[df['name'] == dirname]['emotion_id'].values > 6:
        print('{} has an emotion id > 6, continue'.format(dirname))
        continue
    # if dirname != 'S005-106':
    #     continue
    try:
        file_name = os.listdir(os.path.join(src_path, dirname))[0]
    except:
        print('{} is empty!'.format(dirname))

    if 'r' in file_name:
        keyname = 'r'

    elif 'l' in file_name:
        keyname = 'l'

    else:
        keyname = 'none'

    print('  {} has the key {}'.format(dirname, keyname))
    for filename in os.listdir(os.path.join(mov_path, dirname)):
        if keyname == 'none':
            shutil.copytree(os.path.join(mov_path, dirname), os.path.join(dst_path, dirname))
            break
        else:
            if not os.path.isdir(os.path.join(dst_path, dirname)):
                os.makedirs(os.path.join(dst_path, dirname))
            if keyname in filename:
                print('    copy the file {}'.format(filename))
                shutil.copy(os.path.join(mov_path, dirname, filename), os.path.join(dst_path, dirname))
