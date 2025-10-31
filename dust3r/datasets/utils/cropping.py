# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# croppping utilities
# --------------------------------------------------------
import PIL.Image
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa
import numpy as np  # noqa
from dust3r.utils.geometry import colmap_to_opencv_intrinsics, opencv_to_colmap_intrinsics  # noqa
try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC


class ImageList:
    """ Convenience class to aply the same operation to a whole set of images.
    """

    def __init__(self, images):
        if not isinstance(images, (tuple, list, set)):
            images = [images]
        self.images = []
        for image in images:
            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def to_pil(self):
        return tuple(self.images) if len(self.images) > 1 else self.images[0]

    @property
    def size(self):
        sizes = [im.size for im in self.images]
        assert all(sizes[0] == s for s in sizes)
        return sizes[0]

    def resize(self, *args, **kwargs):
        return ImageList(self._dispatch('resize', *args, **kwargs))

    def crop(self, *args, **kwargs):
        return ImageList(self._dispatch('crop', *args, **kwargs))

    def _dispatch(self, func, *args, **kwargs):
        return [getattr(im, func)(*args, **kwargs) for im in self.images]


def rescale_image_depthmap(image, depthmap, camera_intrinsics, output_resolution, force=True):
    """ Jointly rescale a (image, depthmap) 
        so that (out_width, out_height) >= output_res
        VBR Finetuning: this code was simply for ensuring that a sparse lidar pc is not further subsampled
    """
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (W,H)
    output_resolution = np.array(output_resolution)
    if depthmap is not None:
        # can also use this with masks instead of depthmaps
        assert tuple(depthmap.shape[:2]) == image.size[::-1]

    # define output resolution
    assert output_resolution.shape == (2,)
    scale_final = max(output_resolution / image.size) + 1e-8
    if scale_final >= 1 and not force:  # image is already smaller than what is asked
        return (image.to_pil(), depthmap, camera_intrinsics)
    output_resolution = np.floor(input_resolution * scale_final).astype(int)
    # print("Target Resolution inside rescale_image_depthmap: ", output_resolution, scale_final)
    # first rescale the image so that it contains the crop
    image = image.resize(tuple(output_resolution), resample=lanczos if scale_final < 1 else bicubic)
    if depthmap is not None:
        # depthmap = cv2.resize(depthmap, output_resolution, fx=scale_final,
                            #  fy=scale_final, interpolation=cv2.INTER_NEAREST)
        depthmap = resize_sparse_depth_forward_scatter(depthmap, output_resolution)

    # no offset here; simple rescaling
    camera_intrinsics = camera_matrix_of_crop(
        camera_intrinsics, input_resolution, output_resolution, scaling=scale_final)

    return image.to_pil(), depthmap, camera_intrinsics

import numpy as np

def resize_sparse_depth_forward_scatter(depthmap, output_size):
    """
    Resize sparse depth map by scattering original valid pixels to the nearest
    integer pixel in the resized output, preserving all depth points.

    Args:
        depthmap (np.ndarray): 2D depth map with zeros for invalid pixels.
        output_size (tuple): (width, height) target size.

    Returns:
        np.ndarray: resized depth map with preserved points.
    """
    H_in, W_in = depthmap.shape
    W_out, H_out = output_size

    # Prepare output depth map initialized with zeros
    resized_depth = np.zeros((H_out, W_out), dtype=depthmap.dtype)

    # Get valid original pixel coordinates and values
    ys, xs = np.where(depthmap > 0)
    depths = depthmap[ys, xs]

    # Compute scaling factors
    scale_x = W_out / W_in
    scale_y = H_out / H_in

    # Map original pixel coords to resized pixel coords (float)
    xs_resized_float = xs * scale_x
    ys_resized_float = ys * scale_y

    # Round to nearest integer pixel coords in output
    xs_resized = np.clip(np.round(xs_resized_float).astype(int), 0, W_out - 1)
    ys_resized = np.clip(np.round(ys_resized_float).astype(int), 0, H_out - 1)

    # # Scatter depth values; if multiple points fall into one pixel, keep max depth
    # for x_t, y_t, d in zip(xs_resized, ys_resized, depths):
    #     if resized_depth[y_t, x_t] == 0 or d > resized_depth[y_t, x_t]:
    #         resized_depth[y_t, x_t] = d
            
    # Scatter depth values; if multiple points fall onto one pixel, keep the nearest one (smallest depth)
    for x_t, y_t, d in zip(xs_resized, ys_resized, depths):
        if resized_depth[y_t, x_t] == 0 or d < resized_depth[y_t, x_t]:
            resized_depth[y_t, x_t] = d
    return resized_depth


def camera_matrix_of_crop(input_camera_matrix, input_resolution, output_resolution, scaling=1, offset_factor=0.5, offset=None):
    # Margins to offset the origin
    margins = np.asarray(input_resolution) * scaling - output_resolution
    assert np.all(margins >= 0.0)
    if offset is None:
        offset = offset_factor * margins

    # Generate new camera parameters
    output_camera_matrix_colmap = opencv_to_colmap_intrinsics(input_camera_matrix)
    output_camera_matrix_colmap[:2, :] *= scaling
    output_camera_matrix_colmap[:2, 2] -= offset
    output_camera_matrix = colmap_to_opencv_intrinsics(output_camera_matrix_colmap)

    return output_camera_matrix


def crop_image_depthmap(image, depthmap, camera_intrinsics, crop_bbox):
    """
    Return a crop of the input view.
    """
    image = ImageList(image)
    l, t, r, b = crop_bbox

    image = image.crop((l, t, r, b))
    depthmap = depthmap[t:b, l:r]

    camera_intrinsics = camera_intrinsics.copy()
    camera_intrinsics[0, 2] -= l
    camera_intrinsics[1, 2] -= t

    return image.to_pil(), depthmap, camera_intrinsics


def bbox_from_intrinsics_in_out(input_camera_matrix, output_camera_matrix, output_resolution):
    out_width, out_height = output_resolution
    l, t = np.int32(np.round(input_camera_matrix[:2, 2] - output_camera_matrix[:2, 2]))
    crop_bbox = (l, t, l + out_width, t + out_height)
    return crop_bbox
