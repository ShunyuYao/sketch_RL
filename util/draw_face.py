import os.path
# import torchvision.transforms as transforms
# import torch
from PIL import Image
import numpy as np
import cv2
import glob
from skimage import feature
from keypoint2img import interpPoints, drawEdge

NoCannyEdge = False

def face_edges_draw(img_folder_path, kp_folder_path, dst_path, phase='all'):

    kp_paths = sorted(glob.glob(kp_folder_path + '/*'))
    for i in range(len(kp_paths)):
        # if i > 2:
        #     break
        f = kp_paths[i]
        print('Processing video: {}'.format(f))
        dir_basename = os.path.basename(f)
        save_path = os.path.join(dst_path, phase + '_edges_img', dir_basename)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        for kp_name in sorted(glob.glob(os.path.join(f, '*.txt'))):

            name = os.path.basename(kp_name).split('.')[0]
            img_name = os.path.join(img_folder_path, dir_basename, name+'.jpg')
            img = Image.open(img_name)
            size = img.size

            Image.fromarray(get_face_image(kp_name, size, img)).save(os.path.join(save_path, name+'.jpg'))

def get_face_image(A_path, size, img, transform_A=None, transform_L=None):
    # read face keypoints from path and crop face region
    keypoints, part_list, part_labels = read_keypoints(A_path, size)

    # draw edges and possibly add distance transform maps
    # add_dist_map = not self.opt.no_dist_map
    im_edges = draw_face_edges(keypoints, part_list, transform_A, size, None)

    # canny edge for background
    if not NoCannyEdge:
        edges = feature.canny(np.array(img.convert('L')))
        edges = edges * (part_labels == 0)  # remove edges within face
        im_edges += (edges * 255).astype(np.uint8)
    # edge_tensor = transform_A(Image.fromarray(self.crop(im_edges)))

    # final input tensor
    # input_tensor = torch.cat([edge_tensor, dist_tensor]) if add_dist_map else edge_tensor
    # label_tensor = transform_L(Image.fromarray(self.crop(part_labels.astype(np.uint8)))) * 255.0
    # return input_tensor, label_tensor
    return im_edges

def read_keypoints(A_path, size):
    # mapping from keypoints to face part
    part_list = [[list(range(0, 17)) + list(range(68, 83)) + [0]], # face
                 [range(17, 22)],                                  # right eyebrow
                 [range(22, 27)],                                  # left eyebrow
                 [[28, 31], range(31, 36), [35, 28]],              # nose
                 [[36,37,38,39], [39,40,41,36]],                   # right eye
                 [[42,43,44,45], [45,46,47,42]],                   # left eye
                 [range(48, 55), [54,55,56,57,58,59,48]],          # mouth
                 [range(60, 65), [64,65,66,67,60]]                 # tongue
                ]
    label_list = [1, 2, 2, 3, 4, 4, 5, 6] # labeling for different facial parts
    keypoints = np.loadtxt(A_path, delimiter=',')

    # add upper half face by symmetry
    pts = keypoints[:17, :].astype(np.int32)
    baseline_y = (pts[0,1] + pts[-1,1]) / 2
    upper_pts = pts[1:-1,:].copy()
    upper_pts[:,1] = baseline_y + (baseline_y-upper_pts[:,1]) * 2 // 3
    keypoints = np.vstack((keypoints, upper_pts[::-1,:]))

    # label map for facial part
    w, h = size
    part_labels = np.zeros((h, w), np.uint8)
    for p, edge_list in enumerate(part_list):
        indices = [item for sublist in edge_list for item in sublist]
        pts = keypoints[indices, :].astype(np.int32)
        cv2.fillPoly(part_labels, pts=[pts], color=label_list[p])

    # move the keypoints a bit
    # if not self.opt.isTrain and self.opt.random_scale_points:
    #     self.scale_points(keypoints, part_list[1] + part_list[2], 1, sym=True)
    #     self.scale_points(keypoints, part_list[4] + part_list[5], 3, sym=True)
    #     for i, part in enumerate(part_list):
    #         self.scale_points(keypoints, part, label_list[i]-1)

    return keypoints, part_list, part_labels

def draw_face_edges(keypoints, part_list, transform_A, size, add_dist_map):
    w, h = size
    edge_len = 3  # interpolate 3 keypoints to form a curve when drawing edges
    # edge map for face region from keypoints
    im_edges = np.zeros((h, w), np.uint8) # edge map for all edges
    # dist_tensor = 0
    e = 1
    for edge_list in part_list:
        for edge in edge_list:
            im_edge = np.zeros((h, w), np.uint8) # edge map for the current edge
            for i in range(0, max(1, len(edge)-1), edge_len-1): # divide a long edge into multiple small edges when drawing
                sub_edge = edge[i:i+edge_len]
                x = keypoints[sub_edge, 0]
                y = keypoints[sub_edge, 1]

                curve_x, curve_y = interpPoints(x, y) # interp keypoints to get the curve shape
                drawEdge(im_edges, curve_x, curve_y)
                # if add_dist_map:
                #     drawEdge(im_edge, curve_x, curve_y)

            # if add_dist_map: # add distance transform map on each facial part
            #     im_dist = cv2.distanceTransform(255-im_edge, cv2.DIST_L1, 3)
            #     im_dist = np.clip((im_dist / 3), 0, 255).astype(np.uint8)
            #     im_dist = Image.fromarray(im_dist)
            #     tensor_cropped = transform_A(self.crop(im_dist))
            #     dist_tensor = tensor_cropped if e == 1 else torch.cat([dist_tensor, tensor_cropped])
            #     e += 1

    return im_edges #, dist_tensor

if __name__ == '__main__':
    img_folder_path = '/media/yaosy/办公/research300/sketch_rl/sketch_RL/dataset/frame'
    kp_folder_path = '/media/yaosy/办公/research300/sketch_rl/sketch_RL/dataset/face_landmark/all_keypoints'
    dst_path = '/media/yaosy/办公/research300/sketch_rl/sketch_RL/dataset/face_landmark/all_imgs_canny'
    face_edges_draw(img_folder_path, kp_folder_path, dst_path)
