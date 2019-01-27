#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
It is downloaded from https://raw.githubusercontent.com/anishathalye/obfuscated-gradients/master/inputtransformations/defense.py
"""

from io import BytesIO

import PIL
import PIL.Image
import numpy as np
from torchvision.transforms import ToPILImage, ToTensor


def defend_jpeg(input_tensor, image_mode, quality):
    pil_image = ToPILImage(mode=image_mode)(input_tensor)
    fd = BytesIO()
    pil_image.save(fd, format='jpeg', quality=quality)  # quality level specified in paper
    jpeg_image = ToTensor()(PIL.Image.open(fd))
    return jpeg_image


# based on https://github.com/scikit-image/scikit-image/blob/master/skimage/restoration/_denoise_cy.pyx

# super slow since this is implemented in pure python :'(

def bregman(image, mask, weight, eps=1e-3, max_iter=100):
    rows, cols, dims = image.shape
    rows2 = rows + 2
    cols2 = cols + 2
    total = rows * cols * dims
    shape_ext = (rows2, cols2, dims)

    u = np.zeros(shape_ext)
    dx = np.zeros(shape_ext)
    dy = np.zeros(shape_ext)
    bx = np.zeros(shape_ext)
    by = np.zeros(shape_ext)

    u[1:-1, 1:-1] = image
    # reflect image
    u[0, 1:-1] = image[1, :]
    u[1:-1, 0] = image[:, 1]
    u[-1, 1:-1] = image[-2, :]
    u[1:-1, -1] = image[:, -2]

    i = 0
    rmse = np.inf
    lam = 2 * weight
    norm = (weight + 4 * lam)

    while i < max_iter and rmse > eps:
        rmse = 0

        for k in range(dims):
            for r in range(1, rows + 1):
                for c in range(1, cols + 1):
                    uprev = u[r, c, k]

                    # forward derivatives
                    ux = u[r, c + 1, k] - uprev
                    uy = u[r + 1, c, k] - uprev

                    # Gauss-Seidel method
                    if mask[r - 1, c - 1]:
                        unew = (lam * (u[r + 1, c, k] +
                                       u[r - 1, c, k] +
                                       u[r, c + 1, k] +
                                       u[r, c - 1, k] +
                                       dx[r, c - 1, k] -
                                       dx[r, c, k] +
                                       dy[r - 1, c, k] -
                                       dy[r, c, k] -
                                       bx[r, c - 1, k] +
                                       bx[r, c, k] -
                                       by[r - 1, c, k] +
                                       by[r, c, k]
                                       ) + weight * image[r - 1, c - 1, k]
                                ) / norm
                    else:
                        # similar to the update step above, except we take
                        # lim_{weight->0} of the update step, effectively
                        # ignoring the l2 loss
                        unew = (u[r + 1, c, k] +
                                u[r - 1, c, k] +
                                u[r, c + 1, k] +
                                u[r, c - 1, k] +
                                dx[r, c - 1, k] -
                                dx[r, c, k] +
                                dy[r - 1, c, k] -
                                dy[r, c, k] -
                                bx[r, c - 1, k] +
                                bx[r, c, k] -
                                by[r - 1, c, k] +
                                by[r, c, k]
                                ) / 4.0
                    u[r, c, k] = unew

                    # update rms error
                    rmse += (unew - uprev) ** 2

                    bxx = bx[r, c, k]
                    byy = by[r, c, k]

                    # d_subproblem
                    s = ux + bxx
                    if s > 1 / lam:
                        dxx = s - 1 / lam
                    elif s < -1 / lam:
                        dxx = s + 1 / lam
                    else:
                        dxx = 0
                    s = uy + byy
                    if s > 1 / lam:
                        dyy = s - 1 / lam
                    elif s < -1 / lam:
                        dyy = s + 1 / lam
                    else:
                        dyy = 0

                    dx[r, c, k] = dxx
                    dy[r, c, k] = dyy

                    bx[r, c, k] += ux - dxx
                    by[r, c, k] += uy - dyy

        rmse = np.sqrt(rmse / total)
        i += 1

    return np.asarray(u[1:-1, 1:-1])


def defend_tv(input_array, keep_prob=0.5, lambda_tv=0.03):
    mask = np.random.uniform(size=input_array.shape[:2])
    mask = mask < keep_prob
    return bregman(input_array, mask, weight=2.0 / lambda_tv)
