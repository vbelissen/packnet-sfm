# Copyright 2020 Toyota Research Institute.  All rights reserved.

import cv2
import numpy as np
import random
import torchvision.transforms as transforms
from PIL import Image
import os

from packnet_sfm.utils.misc import filter_dict

########################################################################################################################

def resize_image(image, shape, interpolation=Image.ANTIALIAS):
    """
    Resizes input image.

    Parameters
    ----------
    image : Image.PIL
        Input image
    shape : tuple [H,W]
        Output shape
    interpolation : int
        Interpolation mode

    Returns
    -------
    image : Image.PIL
        Resized image
    """
    transform = transforms.Resize(shape, interpolation=interpolation)
    return transform(image)

def resize_depth(depth, shape):
    """
    Resizes depth map.

    Parameters
    ----------
    depth : np.array [h,w]
        Depth map
    shape : tuple (H,W)
        Output shape

    Returns
    -------
    depth : np.array [H,W]
        Resized depth map
    """
    depth = cv2.resize(depth, dsize=shape[::-1],
                       interpolation=cv2.INTER_NEAREST)
    return np.expand_dims(depth, axis=2)

def resize_sample_image_and_intrinsics_fisheye(sample, shape,
                                       image_interpolation=Image.ANTIALIAS):
    """
    Resizes the image and intrinsics of a sample

    Parameters
    ----------
    sample : dict
        Dictionary with sample values
    shape : tuple (H,W)
        Output shape
    image_interpolation : int
        Interpolation mode

    Returns
    -------
    sample : dict
        Resized sample
    """
    # Resize image and corresponding intrinsics
    image_transform = transforms.Resize(shape, interpolation=image_interpolation)
    (orig_w, orig_h) = sample['rgb'].size
    (out_h, out_w) = shape
    rescale_factor_h = out_h/orig_h
    assert rescale_factor_h == out_w/orig_w

    # Scale intrinsics
    for key in filter_dict(sample, [
        'intrinsics_poly_coeffs'
    ]):
        intrinsics_poly_coeffs = np.copy(sample[key])
        intrinsics_poly_coeffs *= rescale_factor_h
        sample[key] = intrinsics_poly_coeffs
    for key in filter_dict(sample, [
        'intrinsics_principal_point'
    ]):
        intrinsics_principal_point = np.copy(sample[key])
        intrinsics_principal_point *= rescale_factor_h
        sample[key] = intrinsics_principal_point
    for key in filter_dict(sample, [
        'path_to_theta_lut'
    ]):
        path_to_theta_lut = sample[key]

        path_to_theta_lut_clone = path_to_theta_lut
        splitted_path = path_to_theta_lut_clone.split('_')
        w_res_str = splitted_path[-2]
        splitted_path[-2] = str(int(rescale_factor_h * int(w_res_str)))
        h_res_str_with_ext = splitted_path[-1]
        h_res_str_splitted = h_res_str_with_ext.split('.')
        h_res_new = str(int(rescale_factor_h * int(h_res_str_splitted[0])))
        splitted_path[-1] = '.'.join([h_res_new, h_res_str_splitted[1]])
        sample[key] = '_'.join(splitted_path)

        # dir = os.path.dirname(path_to_theta_lut)
        # base_clone, ext = os.path.splitext(os.path.basename(path_to_theta_lut))
        # base_clone_splitted = base_clone.split('_')
        # base_clone_splitted[2] = str(int(rescale_factor_h * float(base_clone_splitted[2])))
        # base_clone_splitted[3] = str(int(rescale_factor_h * float(base_clone_splitted[3])))
        # path_to_theta_lut = os.path.join(dir, '_'.join(base_clone_splitted) + '.npy')
        # sample[key] = path_to_theta_lut
    # Scale images
    for key in filter_dict(sample, [
        'rgb', 'rgb_original',
    ]):
        sample[key] = image_transform(sample[key])
    # Scale context images
    for key in filter_dict(sample, [
        'rgb_context', 'rgb_context_original',
    ]):
        sample[key] = [image_transform(k) for k in sample[key]]
    # Return resized sample
    return sample

def resize_sample(sample, shape, image_interpolation=Image.ANTIALIAS):
    """
    Resizes a sample, including image, intrinsics and depth maps.

    Parameters
    ----------
    sample : dict
        Dictionary with sample values
    shape : tuple (H,W)
        Output shape
    image_interpolation : int
        Interpolation mode

    Returns
    -------
    sample : dict
        Resized sample
    """
    # Resize image and intrinsics
    sample = resize_sample_image_and_intrinsics_fisheye(sample, shape, image_interpolation)
    # Resize depth maps
    for key in filter_dict(sample, [
        'depth',
    ]):
        sample[key] = resize_depth(sample[key], shape)
    # Resize depth contexts
    for key in filter_dict(sample, [
        'depth_context',
    ]):
        sample[key] = [resize_depth(k, shape) for k in sample[key]]
    # Return resized sample
    return sample

########################################################################################################################

def to_tensor(image, tensor_type='torch.FloatTensor'):
    """Casts an image to a torch.Tensor"""
    transform = transforms.ToTensor()
    return transform(image).type(tensor_type)

def to_tensor_sample(sample, tensor_type='torch.FloatTensor'):
    """
    Casts the keys of sample to tensors.

    Parameters
    ----------
    sample : dict
        Input sample
    tensor_type : str
        Type of tensor we are casting to

    Returns
    -------
    sample : dict
        Sample with keys cast as tensors
    """
    transform = transforms.ToTensor()
    # Convert single items
    for key in filter_dict(sample, [
        'rgb', 'rgb_original', 'depth',
    ]):
        sample[key] = transform(sample[key]).type(tensor_type)
    # Convert lists
    for key in filter_dict(sample, [
        'rgb_context', 'rgb_context_original', 'depth_context'
    ]):
        sample[key] = [transform(k).type(tensor_type) for k in sample[key]]
    # Return converted sample
    return sample

########################################################################################################################

def duplicate_sample(sample):
    """
    Duplicates sample images and contexts to preserve their unaugmented versions.

    Parameters
    ----------
    sample : dict
        Input sample

    Returns
    -------
    sample : dict
        Sample including [+"_original"] keys with copies of images and contexts.
    """
    # Duplicate single items
    for key in filter_dict(sample, [
        'rgb'
    ]):
        sample['{}_original'.format(key)] = sample[key].copy()
    # Duplicate lists
    for key in filter_dict(sample, [
        'rgb_context'
    ]):
        sample['{}_original'.format(key)] = [k.copy() for k in sample[key]]
    # Return duplicated sample
    return sample

def colorjitter_sample(sample, parameters, prob=1.0):
    """
    Jitters input images as data augmentation.

    Parameters
    ----------
    sample : dict
        Input sample
    parameters : tuple (brightness, contrast, saturation, hue)
        Color jittering parameters
    prob : float
        Jittering probability

    Returns
    -------
    sample : dict
        Jittered sample
    """
    if random.random() < prob:
        # Prepare transformation
        color_augmentation = transforms.ColorJitter()
        brightness, contrast, saturation, hue = parameters
        augment_image = color_augmentation.get_params(
            brightness=[max(0, 1 - brightness), 1 + brightness],
            contrast=[max(0, 1 - contrast), 1 + contrast],
            saturation=[max(0, 1 - saturation), 1 + saturation],
            hue=[-hue, hue])
        # Jitter single items
        for key in filter_dict(sample, [
            'rgb'
        ]):
            sample[key] = augment_image(sample[key])
        # Jitter lists
        for key in filter_dict(sample, [
            'rgb_context'
        ]):
            sample[key] = [augment_image(k) for k in sample[key]]
    # Return jittered (?) sample
    return sample

########################################################################################################################


