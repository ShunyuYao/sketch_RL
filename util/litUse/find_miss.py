import glob
import os

def find_miss(src_path, dst_path):
    src_list = sorted(glob.glob(src_path + '/*/*.avi'))
    for src in src_list:
        src_basename = os.path.basename(src)
        src_basename = src_basename.split('.')[0]
        print(src_basename)

if __name__ == '__main__':
    src_path = '/home/yaosy/Diskb/research300/videoSegData/mmi/mmi-facial-expression-database_download_2018-12-29_13_28_54/Sessions'
    find_miss(src_path, '')
