# Copyright 2020 Toyota Research Institute.  All rights reserved.

import cv2
import numpy as np
import random
import torchvision.transforms as transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
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

def resize_sample_image_and_intrinsics_multifocal(sample, shape, image_interpolation=Image.ANTIALIAS):
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
    for key in filter_dict(sample, ['intrinsics_poly_coeffs']):
        intrinsics_poly_coeffs = np.copy(sample[key])
        intrinsics_poly_coeffs *= rescale_factor_h
        sample[key] = intrinsics_poly_coeffs
    for key in filter_dict(sample, ['intrinsics_principal_point']):
        intrinsics_principal_point = np.copy(sample[key])
        intrinsics_principal_point *= rescale_factor_h
        sample[key] = intrinsics_principal_point
    # Scale intrinsics
    for key in filter_dict(sample, ['intrinsics_poly_coeffs_geometric_context']):
        res = []
        for k in sample[key]:
            intrinsics_poly_coeffs = np.copy(k)
            intrinsics_poly_coeffs *= rescale_factor_h
            res.append(intrinsics_poly_coeffs)
        sample[key] = res
    for key in filter_dict(sample, ['intrinsics_principal_point_geometric_context']):
        res = []
        for k in sample[key]:
            intrinsics_principal_point = np.copy(k)
            intrinsics_principal_point *= rescale_factor_h
            res.append(intrinsics_principal_point)
        sample[key] = res
    for key in filter_dict(sample, ['intrinsics_K']):
        intrinsics = np.copy(sample[key])
        intrinsics[0] *= out_w / orig_w
        intrinsics[1] *= out_h / orig_h
        sample[key] = intrinsics
    for key in filter_dict(sample, ['intrinsics_K_geometric_context']):
        res = []
        for k in sample[key]:
            intrinsics = np.copy(k)
            intrinsics[0] *= out_w / orig_w
            intrinsics[1] *= out_h / orig_h
            res.append(intrinsics)
        sample[key] = res
    # Scale images
    for key in filter_dict(sample, [
        'rgb',
        'rgb_original'
    ]):
        sample[key] = image_transform(sample[key])
    # Scale context images
    for key in filter_dict(sample, [
        'rgb_temporal_context',
        'rgb_geometric_context',
        'rgb_geometric_context_temporal_context',
        'rgb_temporal_context_original',
        'rgb_geometric_context_original',
        'rgb_geometric_context_temporal_context_original'
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
    sample = resize_sample_image_and_intrinsics_multifocal(sample, shape, image_interpolation)
    # Resize depth maps
    for key in filter_dict(sample, [
        'depth',
    ]):
        sample[key] = resize_depth(sample[key], shape)
    # Resize depth contexts
    for key in filter_dict(sample, [
        'depth_temporal_context',
        'depth_geometric_context',
        'depth_temporal_context_geometric_context',
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
    print(sample)
    transform = transforms.ToTensor()
    # Convert single items
    for key in filter_dict(sample, [
        'rgb',
        'rgb_original',
        'depth',
    ]):
        print('tranform1')
        sample[key] = transform(sample[key]).type(tensor_type)
    # Convert lists
    for key in filter_dict(sample, [
        'rgb_temporal_context',
        'rgb_geometric_context',
        'rgb_geometric_context_temporal_context',
        'rgb_temporal_context_original',
        'rgb_geometric_context_original',
        'rgb_geometric_context_temporal_context_original'
        'depth_temporal_context',
        'depth_geometric_context',
        'depth_temporal_context_geometric_context',
    ]):
        print('tranform2')
        sample[key] = [transform(k).type(tensor_type) for k in sample[key]]
    print(sample)
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
        'rgb_temporal_context',
        'rgb_geometric_context',
        'rgb_geometric_context_temporal_context',
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
            'rgb_temporal_context',
            'rgb_geometric_context',
            'rgb_geometric_context_temporal_context',
        ]):
            sample[key] = [augment_image(k) for k in sample[key]]
    # Return jittered (?) sample
    return sample

########################################################################################################################


