import os
import glob
import cv2
import xml.dom.minidom
import numpy as np


def read_view(xml_path):
    dom = xml.dom.minidom.parse(xml_path)
    root = dom.documentElement
    element = root.getElementsByTagName('track')[0]
    view_num = element.getAttribute('view')

    return view_num

def vid2pic(src_path, miss_list, save_path, task_type='frame'):

    width = 361
    height = 576
    mid_names = sorted(os.listdir(src_path))
    for name in mid_names:
        file_path = sorted(glob.glob(os.path.join(src_path, name) + '/*.xml'), reverse=True)
        for opt_file in file_path:
            if 'session' in opt_file:
                view = read_view(opt_file)
                #print(view)
                # if view != '0' and view != '1' and view != '2':
                #     print(name)

            elif 'oao' not in opt_file:

                basename = os.path.basename(opt_file)
                basename = basename.split('.')[0]
                # if 'S054' not in basename:
                #     continue
                save_dir = os.path.join(save_path, basename)
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                cap = cv2.VideoCapture(os.path.join(src_path, name, basename+'.avi'))
                print('Process video: {}'.format(basename))

                idx = 1
                i = -1
                while(cap.isOpened()):
                    flag, img = cap.read()
                    if flag is False:
                        break
                    size = img.shape
                    i += 1

                    if view == '2':
                        img_r = img[-height:, -width:]
                        cv2.imwrite(os.path.join(save_dir, 'img_r{:05d}.jpg'.format(idx) ), img_r)
                        img_l = img[:height, :width]
                        cv2.imwrite(os.path.join(save_dir, 'img_l{:05d}.jpg'.format(idx) ), img_l)
                        idx += 1
                    elif view == '1':
                        img = np.rot90(img)
                        cv2.imwrite(os.path.join(save_dir, 'img_{:05d}.jpg'.format(idx) ), img)
                        idx += 1
                    elif view == '0':
                        shape = img.shape[:2]
                        if shape[0] != 480:
                            img = np.rot90(img, -1)
                        cv2.imwrite(os.path.join(save_dir, 'img_{:05d}.jpg'.format(idx) ), img)
                        idx += 1
                    else:
                        raise Exception('unkown view: {}'.format(view))

    return True

if __name__ == '__main__':
    # cap = cv2.VideoCapture('/home/yaosy/Diskb/research300/videoSegData/mmi/mmi-facial-expression-database_download_2018-12-29_13_28_54/Sessions/1841/S035-004.avi')
    # print(cap.isOpened())

    miss_list = ['S001-100', 'S001-101', 'S001-102', 'S001-103', 'S001-104', 'S001-105', 'S001-106', 'S001-107', 'S001-108', 'S001-109', 'S001-110', 'S001-111', 'S001 -112', 'S001-113', 'S001-114', 'S001-115', 'S001-116', 'S001-117', 'S001-118', 'S015-002',
    'S016-001', 'S017-081', 'S021-003', 'S021-004', 'S021-005', 'S046-011', 'S046-014', 'S047-001', 'S047-004', 'S049-002', 'S049-003', 'S049-008', 'S049-010', 'S049-012', 'S040-003', 'S040-007', 'S049-014', 'S050-004', 'S050-007', 'S050-011', 'S050-012', 'S050-015', 'S053-003', 'S053-004', 'S053-005',
    'S053-014', 'S054-001', 'S054-023', 'S002-099', 'S002-100', 'S002-101', 'S002-102', 'S002-103', 'S002-104', 'S002-105', 'S045-005', 'S045-007', 'S045-011', 'S046-002', 'S046-003', 'S046-004', 'S046-005', 'S046-008 ', 'S021-006', 'S049-013', 'S002-106', 'S032-010', 'S002-107', 'S002-109', 'S002-110',
    'S002-111', 'S002-112', 'S002-113', 'S002-114', 'S002-115', 'S002-116', 'S002-117', 'S002-118', 'S002-119', 'S003-102', 'S003-103', 'S003-105', 'S003-106', 'S005-105', 'S005-106', 'S005-107', 'S005-108', 'S006-108', 'S006-109', 'S006-110', 'S033-008', 'S035-001', 'S035-004']
    src_path = '/media/yaosy/办公/research300/videoSegData/mmi/mmi-facial-expression-database_download_2018-12-29_13_28_54/Sessions'
    save_path = '/media/yaosy/办公/research300/videoSegData/mmi/mmif/new_frame'
    print(vid2pic(src_path, miss_list, save_path))
