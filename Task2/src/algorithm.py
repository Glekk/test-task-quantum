import os
import logging
from dotenv import load_dotenv
import torch
import cv2
import numpy as np
import kornia as K
import kornia.feature as KF
from utils import load_image, scale_points_to_original_image_size, equalize_img, visualize_matches, find_inliers
import matplotlib.pyplot as plt


# Load constants from .env file
load_dotenv()
LOGGING_LEVEL = os.getenv('LOGGING_LEVEL')
LOGGING_FORMAT = os.getenv('LOGGING_FORMAT')
LOGGING_DATE_FORMAT = os.getenv('LOGGING_DATE_FORMAT')
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT, datefmt=LOGGING_DATE_FORMAT)


class LoFTR:
    '''
    Class to perform matching using LoFTR

    Example usage:
    
    loftr = LoFTR()

    loftr("path/to/image0", "path/to/image1")

    Call args:
        img0_path(str): path to first image
        img1_path(str): path to second image
        img_shape(tuple): shape to resize images to
        equalize(bool): whether to use histogram equalization
        clahe(bool): whether to use clahe for equalization
        visualize(bool): whether to visualize the matches
        visualize_non_scaled_images(bool): whether to visualize the images without resizing
    '''
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.matcher = KF.LoFTR(pretrained="outdoor").to(self.device)
        
    def __call__(self, img0_path, img1_path, img_shape=(1024, 1024), equalize=True, clahe=False, visualize=True, visualize_non_scaled_images=False, save=False):
        img0, non_scaled_img0 = load_image(img0_path, img_shape, return_non_scaled=visualize_non_scaled_images, device=self.device)
        img1, non_scaled_img1 = load_image(img1_path, img_shape, return_non_scaled=visualize_non_scaled_images, device=self.device)
        original_img0 = img0.clone()
        original_img1 = img1.clone()

        if equalize:
            img0 = equalize_img(img0, clahe)
            img1 = equalize_img(img1, clahe)

        # loftr expects grayscale images
        img0 = K.color.rgb_to_grayscale(img0) 
        img1 = K.color.rgb_to_grayscale(img1)
        logging.info(f"Images loaded")

        with torch.inference_mode():
            input_dict = {"image0": img0, "image1": img1}
            correspondences_dict = self.matcher(input_dict)
        logging.info(f"Predictions made")

        mkpts0 = correspondences_dict["keypoints0"].cpu().numpy()
        mkpts1 = correspondences_dict["keypoints1"].cpu().numpy()

        if visualize_non_scaled_images:
            mkpts0 = scale_points_to_original_image_size(mkpts0, non_scaled_img0.shape[-2:], img_shape)
            mkpts1 = scale_points_to_original_image_size(mkpts1, non_scaled_img1.shape[-2:], img_shape)

        inliers = find_inliers(mkpts0, mkpts1)

        lafs0 = KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0)[None])
        lafs1 = KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1)[None])
        idxs = torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2)
        logging.info(f"Matches found")

        if visualize_non_scaled_images:
            visualize_matches(non_scaled_img0, non_scaled_img1, lafs0, lafs1, idxs, save, inliers)

        elif visualize:
            visualize_matches(original_img0, original_img1, lafs0, lafs1, idxs, save, inliers)


class DISK_LightGlue:
    '''
    Class to perform matching using DISK and LightGlue

    Example usage:
    
    disk_lg = DISK_LightGlue()

    disk_lg("path/to/image0", "path/to/image1")

    Call args:
        img0_path(str): path to first image
        img1_path(str): path to second image
        img_shape(tuple): shape to resize images to
        equalize(bool): whether to use histogram equalization
        clahe(bool): whether to use clahe for equalization
        visualize(bool): whether to visualize the matches
        visualize_non_scaled_images(bool): whether to visualize the images without resizing
    '''
    def __init__(self, disk_features=2048):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.disk_features = disk_features
        self.matcher = KF.DISK.from_pretrained("depth").to(self.device)
        self.lg_matcher = KF.LightGlueMatcher("disk").eval().to(self.device)

    def get_matching_keypoints(self, kpts0, kpts1, idxs):
            '''
            Get matching keypoints from two sets of keypoints and their indices
            Args:
                kpts0: keypoints from image 0
                kpts1: keypoints from image 1
                idxs: indices of matching keypoints
            Returns:
                mkpts0: matching keypoints from image 0
                mkpts1: matching keypoints from image 1
            '''
            mkpts1 = kpts0[idxs[:, 0]]
            mkpts2 = kpts1[idxs[:, 1]]
            return mkpts1.cpu().numpy(), mkpts2.cpu().numpy()
    
    def __call__(self, img0_path, img1_path, img_shape=(1024, 1024), equalize=True, clahe=False, visualize=True, visualize_non_scaled_images=False, save=False):
        img0, non_scaled_img0 = load_image(img0_path, img_shape, return_non_scaled=visualize_non_scaled_images, device=self.device)
        img1, non_scaled_img1 = load_image(img1_path, img_shape, return_non_scaled=visualize_non_scaled_images, device=self.device)
        original_img0 = img0.clone()
        original_img1 = img1.clone()

        if equalize:
            img0 = equalize_img(img0, clahe)
            img1 = equalize_img(img1, clahe)
        logging.info(f"Images loaded")

        # Get dimensions of images for lightglue
        hw1 = torch.tensor(img0.shape[2:], device=self.device)
        hw2 = torch.tensor(img1.shape[2:], device=self.device)

        with torch.inference_mode():
            inp = torch.cat([img0, img1], dim=0)
            features1, features2 = self.matcher(inp, self.disk_features, pad_if_not_divisible=True)
            kpts0, descs0 = features1.keypoints, features1.descriptors
            kpts1, descs1 = features2.keypoints, features2.descriptors
            lafs0 = KF.laf_from_center_scale_ori(kpts0[None])
            lafs1 = KF.laf_from_center_scale_ori(kpts1[None])
            dists, idxs = self.lg_matcher(descs0, descs1, lafs0, lafs1, hw1=hw1, hw2=hw2)
        logging.info(f"Predictions made")

        idxs = idxs.cpu()
        mkpts0, mkpts1 = self.get_matching_keypoints(kpts0, kpts1, idxs)

        if visualize_non_scaled_images:
            mkpts0 = scale_points_to_original_image_size(mkpts0, non_scaled_img0.shape[-2:], img_shape)
            mkpts1 = scale_points_to_original_image_size(mkpts1, non_scaled_img1.shape[-2:], img_shape)

        inliers = find_inliers(mkpts0, mkpts1)
        logging.info(f"Matches found")
        
        if visualize_non_scaled_images:
            visualize_matches(non_scaled_img0, non_scaled_img1, lafs0, lafs1, idxs, save, inliers)

        elif visualize:
            visualize_matches(original_img0, original_img1, lafs0, lafs1, idxs, save, inliers)


class SIFT:
    '''
    Class to perform matching using SIFT

    Example usage:

    sift = SIFT()

    sift("path/to/image0", "path/to/image1")

    Call args:
        img0_path(str): path to first image
        img1_path(str): path to second image
        img_shape(tuple): shape to resize images to
        equalize(bool): whether to use histogram equalization
        clahe(bool): whether to use clahe for equalization
        visualize(bool): whether to visualize the matches
        visualize_non_scaled_images(bool): whether to visualize the images without resizing
    '''
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()

    def __call__(self, img0_path, img1_path, img_shape=(1024, 1024), equalize=True, clahe=False, visualize=True, visualize_non_scaled_images=False, save=False):
        img0, non_scaled_img0 = load_image(img0_path, img_shape, return_non_scaled=visualize_non_scaled_images)
        img1, non_scaled_img1 = load_image(img1_path, img_shape, return_non_scaled=visualize_non_scaled_images)
        original_img0 = img0.clone()
        original_img1 = img1.clone()

        if equalize:
            img0 = equalize_img(img0, clahe)
            img1 = equalize_img(img1, clahe)

        # SIFT expects grayscale images
        img0 = K.color.rgb_to_grayscale(img0)
        img1 = K.color.rgb_to_grayscale(img1)

        img0 = img0.squeeze().cpu().numpy()
        img1 = img1.squeeze().cpu().numpy()

        #to 0-255
        img0 = (img0 * 255).astype(np.uint8)
        img1 = (img1 * 255).astype(np.uint8)
        logging.info(f"Images loaded")

        kpts0, descs0 = self.sift.detectAndCompute(img0, None)
        kpts1, descs1 = self.sift.detectAndCompute(img1, None)
        logging.info(f"Predictions made")

        matches = self.bf.knnMatch(descs0, descs1, k=2) # k:-finds k best matches for each descriptors from quary set
        
        # Perform Lowe's ratio test (if the two distances are sufficiently different)
        # col1 is 1st best match and col2 is 2nd best match for each descriptor
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        mkpts0 = np.array([kpts0[m[0].queryIdx].pt for m in good])
        mkpts1 = np.array([kpts1[m[0].trainIdx].pt for m in good])

        if visualize_non_scaled_images:
            mkpts0 = scale_points_to_original_image_size(mkpts0, non_scaled_img0.shape[-2:], img_shape)
            mkpts1 = scale_points_to_original_image_size(mkpts1, non_scaled_img1.shape[-2:], img_shape)

        inliers = find_inliers(mkpts0, mkpts1)

        lafs0 = KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0)[None])
        lafs1 = KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1)[None])
        idxs = torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2)
        logging.info(f"Matches found")

        if visualize_non_scaled_images:
            visualize_matches(non_scaled_img0, non_scaled_img1, lafs0, lafs1, idxs, save, inliers)

        elif visualize:
            visualize_matches(original_img0, original_img1, lafs0, lafs1, idxs, save, inliers)

