import cv2
import kornia as K
from kornia.enhance import equalize_clahe, equalize
from kornia_moons.feature import draw_LAF_matches
import numpy as np
import matplotlib.pyplot as plt

def scale_points_to_original_image_size(points, original_image_size, resized_image_size):
    '''
    Scale points to the original image size

    Args:
        points(np.array): points to scale
        original_image_size(tuple): original image size
        resized_image_size(tuple): resized image size

    Returns:
        np.array: scaled points
    '''
    scale_factor_x = original_image_size[1] / resized_image_size[1]
    scale_factor_y = original_image_size[0] / resized_image_size[0]
    scaled_points = points.copy()
    scaled_points[:, 0] = scaled_points[:, 0] * scale_factor_x
    scaled_points[:, 1] = scaled_points[:, 1] * scale_factor_y
    return scaled_points


def equalize_img(img, clahe=False):
    '''
    Apply histogram equalization to an image

    Args:
        img(torch.Tensor): image to equalize
        clahe(bool): whether to use clahe instead of simple equalization
    
    Returns:
        torch.Tensor: equalized image
    '''
    if not clahe:
        img = equalize(img)
    else:
        img = equalize_clahe(img)
    return img


def load_image(path, img_shape=None, return_non_scaled=False, device='cuda'):
    '''
    Load an image and convert it to a torch tensor

    Args:
        path(str): path to the image
        img_shape(tuple): shape of the image to resize to
        device(str): device to load the image to

    Returns:
        torch.Tensor: image tensor
    '''
    img = K.io.load_image(path, K.io.ImageLoadType.RGB32, device=device)
    img = img.unsqueeze(0)
    non_scaled = img.clone()
    if img_shape is not None:
        img = K.geometry.resize(img, img_shape)
    if return_non_scaled:
        return img, non_scaled
    return img, None


def find_inliers(mkpts0, mkpts1):
    '''
    Find inliers using RANSAC

    Args:
        mkpts0(np.array): keypoints of the first image
        mkpts1(np.array): keypoints of the second image

    Returns:
        np.array: inliers
    '''
    #cv2.findFundamentalMat could throw an exception if the number of points is too low 
    try:
        Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.7, 0.999, 100000)
        inliers = inliers > 0
    except:
        inliers = np.zeros(mkpts0.shape[0], dtype=bool)

    return inliers


def visualize_matches(img0, img1, lafs0, lafs1, idxs, save, inliers=None):
    '''
    Visualize matches between two images

    Args:
        img0(torch.Tensor): first image
        img1(torch.Tensor): second image
        lafs0(torch.Tensor): local affine frames of the first image
        lafs1(torch.Tensor): local affine frames of the second image
        idxs(torch.Tensor): indices of the matches
        save(bool): whether to save the visualization
        inliers(np.array): inliers to draw
    '''
    fig, ax = draw_LAF_matches(
        lafs0,
        lafs1,
        idxs,
        img0,
        img1,
        inliers,
        draw_dict={"inlier_color": (0.2, 1, 0.2, 0.6), "tentative_color": None, "feature_color": (0.2, 0.5, 1), "vertical": False},
        return_fig_ax=True
    )
    if save:
        dpi = max(img0.shape[-2:]) // 10
        fig.savefig('matches.png', dpi=dpi, bbox_inches='tight')
    plt.show()