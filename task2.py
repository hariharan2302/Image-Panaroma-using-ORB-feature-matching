# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 3. Not following the project guidelines will result in a 10% reduction in grades
# 4 . If you want to show an image for debugging, please use show_image() function in helper.py.
# 5. Please do NOT save any intermediate files in your final submission.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import array as arr

def parse_args():
    parser = argparse.ArgumentParser(description="cse 573 homework 4.")
    parser.add_argument(
        "--input_path", type=str, default="data/images_panaroma",
        help="path to images for panaroma construction")
    parser.add_argument(
        "--output_overlap", type=str, default="./task2_overlap.txt",
        help="path to the overlap result")
    parser.add_argument(
        "--output_panaroma", type=str, default="./task2_result.png",
        help="path to final panaroma image ")

    args = parser.parse_args()
    return args

def warpTwoImages(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate

    result = cv2.warpPerspective(img2, Ht@H, (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1], t[0]:w1+t[0]] = img1
    return result

def stitch(inp_path, imgmark, N=4, savepath=''): 
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'{inp_path}/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
    def my_kp_des_detection(imgs):
        # I am using ORB(Oriented FAST(Feature Detection) and Rotated BRIEF(feature Description)) method beacuse of
        # It's robustness in speed and accuracy in these kind of applications.
        orb = cv2.ORB_create()  # Creating an ORB object
        key_points = []
        descriptors = []
        max_descriptor_length = 0
        for image in imgs:
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kp, des = orb.detectAndCompute(gray_img, None)
            key_points.append(kp)
            descriptors.append(des)
            if des is not None and len(des) > max_descriptor_length:
                max_descriptor_length = len(des)
        padded_descriptors = []
        for des in descriptors:
            if des is None:
                padded_descriptors.append(np.zeros((max_descriptor_length, des.shape[1]), dtype=np.float32))
            else:
                pad_length = max_descriptor_length - len(des)
                padded_descriptors.append(np.pad(des,((0, pad_length), (0, 0)), mode='constant', constant_values=0))
        return key_points, np.array(padded_descriptors, dtype=np.float32)

    def customised_matching(descriptor1, descriptor2, threshold=0.5):
        matches = []
        for idx1, desc1 in enumerate(descriptor1):
            best_match_idx = -1
            best_match_distance = float('inf')
            for idx2, desc2 in enumerate(descriptor2):
                distance = np.linalg.norm(desc1 - desc2)
                if distance < best_match_distance:
                    best_match_distance = distance 
                    best_match_idx = idx2
            if best_match_distance < threshold:
                matches.append((idx1, best_match_idx))
        return matches
    def compute_overlap_percentage(image1, image2, homography):
        h, w = image1.shape[:2]
        pts1 = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]], dtype=np.float32).reshape(-1, 1, 2)
        pts2 = cv2.perspectiveTransform(pts1, homography)
        area1 = cv2.contourArea(pts2)
        area2 = cv2.contourArea(pts1)
        overlap_percentage = min(area1, area2) / max(area1, area2) * 100
        return 1 if overlap_percentage >= 20 else 0
    def compute_overlap(images, keypoints, descriptors):
        overlaps = np.zeros((len(images), len(images)), dtype=np.uint8)
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                kp1 = keypoints[i]
                kp2 = keypoints[j]
                des1 = descriptors[i]
                des2 = descriptors[j]
                matches = customised_matching(des1, des2)
                if len(matches) >= 4:
                    src_pts = np.float32([kp1[m[0]].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m[1]].pt for m in matches]).reshape(-1, 1, 2)
                    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    overlap_percentage = compute_overlap_percentage(images[i], images[j], H)
                    if overlap_percentage >= 20:
                        overlaps[i, j] = 1
                        overlaps[j, i] = 1
        return overlaps
    def compute_homography(images, keypoints, descriptors, overlap_arr):
        homography_matrices = {}
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                if overlap_arr[i, j] == 1:
                    kp1 = keypoints[i]
                    kp2 = keypoints[j]
                    des1 = descriptors[i]
                    des2 = descriptors[j]
                    matches = customised_matching(des1, des2)
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    homography_matrices[(i, j)] = H
        return homography_matrices
    def create_panorama(images, homography_matrices):
        panorama = images[0].copy()
        for i in range(1, len(images)):
            for j in range(i):
                if (j, i) in homography_matrices:
                    H = homography_matrices[(j, i)]
                    panorama = warpTwoImages(panorama, images[i], H)
        return panorama

    my_kp, my_descriptors = my_kp_des_detection(imgs)
    overlap_arr = compute_overlap(imgs, my_kp, my_descriptors)
    homography_matrices = compute_homography(imgs, my_kp, my_descriptors, overlap_arr)
    panorama = create_panorama(imgs, homography_matrices)
    cv2.imwrite(savepath, panorama)

    return overlap_arr
    
if __name__ == "__main__":
    #task2
    args = parse_args()
    overlap_arr = stitch(args.input_path, 't2', N=4, savepath=f'{args.output_panaroma}')
    with open(f'{args.output_overlap}', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
