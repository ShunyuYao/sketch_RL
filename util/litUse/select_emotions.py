import glob
import os
import shutil

def select_emotions(src_path, dst_path, save_path, task_type='frame'):
    mid_names = os.listdir(src_path)
    miss_list = []
    for name in mid_names:
        file_path = glob.glob(os.path.join(src_path, name) + '/*.avi')
        basename = os.path.basename(file_path[0])
        basename = basename.split('.')[0]

        dst_dir = os.path.join(dst_path, basename)
        if not os.path.isdir(dst_dir):
            print('{} not found!'.format(basename))
            miss_list.append(basename)
            continue

        # save_dir = os.path.join(save_path, task_type, basename)
        # # if not os.path.isdir(save_dir):
        # #     os.makedirs(save_dir)
        #
        # shutil.copytree(dst_dir, save_dir)

    return miss_list

if __name__ == '__main__':
    src_path = '/home/yaosy/Diskb/research300/videoSegData/mmi/mmi-facial-expression-database_download_2018-12-29_13_28_54/Sessions'
    dst_path = '/home/yaosy/Diskb/research300/videoSegData/mmi/mmif/frame'
    save_path = '/home/yaosy/Diskb/research300/sketch_rl/sketch_RL/dataset'
    miss_list = select_emotions(src_path, dst_path, save_path)
    print("miss list: \n", miss_list)
