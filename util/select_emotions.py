import glob
import os
import shutil

def select_emotions(src_path, dst_path, save_path):
    mid_names = os.listdir(src_path)
    for name in mid_names:
        file_path = glob.glob(os.path.join(src_path, name) + '/*.xml')
        basename = os.path.basename(file_path)

        miss_list = []
        dst_dir = os.path.join(dst_path, basename)
        if not os.path.isdir(dst_dir):
            print('{} not found!'.format(basename))
            miss_list.append(basename)
            continue

        save_dir = os.path.join(save_path, basename)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        shutil.copytree(dst_path, save_path)

if __name__ == '__main__':
    src_path = ''
    dst_path = ''
    save_path = ''
    select_emotions(src_path, dst_path, save_path)
