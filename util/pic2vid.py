import cv2
import numpy as np
import os

train_dir = '/home/yaosy/Diskb/research300/videoSegData/WAD/dataPreprocess'
pre_edgename = 'road03_cam_6_video_1_image_list_train_edges'

pic_numseq = [[119, 151],
[162, 187],
[190, 203],
[356, 367],
[544, 592],
[659, 683],
[688, 749],
[777, 808]]

fps = 1
size = (3384, 2710)

edges_dir = os.path.join(train_dir, pre_edgename)
for numseq in pic_numseq:
    start = numseq[0]
    end = numseq[1]
    videowriter= cv2.VideoWriter('/home/yaosy/Diskb/research300/videoSegData/WAD/grayscalevideo/{}_frame{}-{}.avi'.format(pre_edgename, start, end),
                                 cv2.VideoWriter_fourcc(*'XVID'), fps, size)
    for i in range(start, end+1):
        edge_path = os.path.join(edges_dir, '{}.jpg'.format(i))
        edge = cv2.imread(edge_path)
        # print(edge)
        videowriter.write(edge)
