import os
import glob
import shutil

def movFewerFiles(src_path):
    mid_names = os.listdir(src_path)
    unkown_files = []

    for i, mid_name in enumerate(mid_names):
        if i > 80:
            break
        l_num = 0
        r_num = 0
        # print(mid_name)
        mid_paths = os.path.join(src_path, mid_name)
        seek_files = os.listdir(mid_paths)
        for seek_file in seek_files:
            if 'r' in seek_file:
                r_num += 1
            elif 'l' in seek_file:
                l_num += 1
            else:
                break

        if l_num == 0 or r_num == 0:
            continue
        print('l nums: {}, r nums: {}'.format(l_num, r_num))


        save_path = os.path.join(src_path, '../../mov_edges_img', mid_name)

        if r_num - l_num > 5:
            print('mov left part files!')
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            mov_files = glob.glob(mid_paths + '/*l*.jpg')
            for mov_file in mov_files:
                shutil.move(mov_file, save_path)
        elif l_num - r_num > 5:
            print('mov right part files')
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            mov_files = glob.glob(mid_paths + '/*r*.jpg')
            for mov_file in mov_files:
                shutil.move(mov_file, save_path)
        else:
            print('unkown how to move')
            unkown_files.append(mid_name)
        print()

    return unkown_files

if __name__ == '__main__':
    src_path = '/media/yaosy/办公/research300/sketch_rl/sketch_RL/dataset/face_landmark/all_imgs_canny/all_edges_img'
    unkown_files = movFewerFiles(src_path)
    print(unkown_files)
