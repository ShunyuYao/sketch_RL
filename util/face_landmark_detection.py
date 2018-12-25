import os
import glob
from skimage import io
import numpy as np
import dlib
import sys

def face_landmark_detection(dataset_path, predictor_path, dst_path, phase='train'):

    faces_folder_path = dataset_path #os.path.join(dataset_path, phase + '_img/')
    #predictor_path = os.path.join(dataset_path, 'shape_predictor_68_face_landmarks.dat')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    img_paths = sorted(glob.glob(faces_folder_path + "/*"))

    for i in range(len(img_paths)):
        if i > 2:
            break

        f = img_paths[i]
        print("Processing video: {}".format(f))
        save_path = os.path.join(dst_path, phase + '_keypoints', os.path.basename(f))
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        for img_name in sorted(glob.glob(os.path.join(f, '*.jpg'))):
            img = io.imread(img_name)
            dets = detector(img, 1)
            if len(dets) > 0:
                shape = predictor(img, dets[0])
                points = np.empty([68, 2], dtype=int)
                for b in range(68):
                    points[b,0] = shape.part(b).x
                    points[b,1] = shape.part(b).y

                save_name = os.path.join(save_path, os.path.basename(img_name)[:-4] + '.txt')
                np.savetxt(save_name, points, fmt='%d', delimiter=',')

if __name__ == '__main__':
    dataset_path = '/home/yaosy/Diskb/research300/videoSegData/mmi/mmif/frame'
    predictor_path = '/home/yaosy/Diskb/research300/sketch_rl/sketch_RL/util/face_extract/shape_predictor_68_face_landmarks.dat'
    save_path = '/home/yaosy/Diskb/research300/videoSegData/mmi/mmif_edge'
    face_landmark_detection(dataset_path, predictor_path, save_path)
