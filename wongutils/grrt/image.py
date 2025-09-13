__copyright__ = """Copyright (C) 2025 George N. Wong"""
__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np


def blur_image_butterworth(image, image_fov_uas, butterworth_scale_glambda,
                           butterworth_order=2, pad=4):
    """
    Blur an image with a Butterworth filter in Fourier space.

    :arg image: 2D numpy array containing the image to be blurred

    :arg image_fov_uas: full image field of view (FOV) in microarcseconds

    :arg butterworth_scale_glambda: spatial scale (in Glambda) at which
        to apply the Butterworth filter

    :arg butterworth_order: order of the Butterworth filter

    :arg pad: integer padding factor to use when filtering the image to
        mitigate edge effects

    :returns: blurred image as a 2D numpy array
    """

    # compute Butterworth filter radius in pixels
    pixel_uv = compute_uv_pixel_spacing(image_fov_uas, padding=pad)
    butterworth_radius = butterworth_scale_glambda / pixel_uv

    # create Butterworth filter mask
    s = image.shape[0]
    n = s * pad
    u = np.fft.fftshift(np.fft.fftfreq(n))
    v = np.fft.fftshift(np.fft.fftfreq(n))
    uu, vv = np.meshgrid(u, v)
    rr = np.sqrt(uu**2 + vv**2) * n
    butterworth_mask = 1. / np.sqrt(1. + (rr/butterworth_radius)**(2*butterworth_order))

    # apply filter to image
    blurred_image = filter_image(image, butterworth_mask, pad=pad)

    return blurred_image


def compute_uv_pixel_spacing(FOV_in_uas, padding=1):
    """
    Compute the uv-plane pixel spacing for an image.

    The uv pixel spacing is simply the inverse of the image field of view (FOV).
    The FOV can be derived from the usual ipole image parameters:

        FOV = (FOV_M * (M_bh / M_sun)) / d_pc * 4.783e-14   [radians]

    where:
        FOV_M   = full image FOV in gravitational radii (GM/c^2)
        M_bh    = black hole mass [M_sun]
        d_pc    = source distance [pc]

    If we pad the image, the effective FOV increases by the same factor. The
    uv pixel spacing in Glambda is then:

        pixel_uv = 1 / (FOV * 1e9 * padding)

    For example, with FOV_M=64, M_bh=6.5e9 M_sun, d_pc=16.8e6 pc, and padding=4,
    the uv pixel spacing is ~0.21108 Glambda.

    Notice that the uv pixel spacing can also be easily computed from the FOV in
    microarcseconds:

        pixel_uv = 206.2648 / (FOV_uas * padding)   [Glambda]

    Practical notes:
      - It's likely you're using this to figure out what radius (e.g., for a
        Butterworth filter) corresponds to a given spatial scale in Glambda.
        In that case, it's just Butterworth_radius = target_scale / pixel_uv,
        so for example, (15 Glambda scale) / 0.21108 ~ 71.0631, which means
        we should use a Butterworth filter with radius ~71 pixels.

    :arg FOV_in_uas: full image field of view (FOV) in microarcseconds

    :arg padding: integer padding factor used when filtering the image to

    :returns: dictionary containing the image data
    """

    return 206.2648 / (FOV_in_uas * padding)


def filter_image(image, mask, pad=4):
    """
    Apply a filter to an image in Fourier space.

    :arg image: 2D numpy array containing the image to be filtered

    :arg mask: 2D numpy array containing the Fourier filter to be applied

    :arg pad: integer padding factor to use when filtering the image. The
        image will be padded to (pad * s, pad * s) where s is the size of the
        input image. Padding helps mitigate edge effects when filtering.

    :returns: filtered image as a 2D numpy array
    """

    # pad the image
    s = image.shape[0]
    padded_image = np.zeros((s*pad, s*pad), dtype=image.dtype)
    padded_image[s*(pad-1)//2:s*(pad+1)//2,
                 s*(pad-1)//2:s*(pad+1)//2] = image

    # apply filter in Fourier space
    padded_image = np.fft.fftshift(padded_image)
    image_ft = np.fft.fftshift(np.fft.fft2(padded_image))
    image_ft_filt = np.fft.ifftshift(image_ft * mask)
    padded_image = np.fft.fftshift(np.fft.ifft2(image_ft_filt))

    # return unpadded result
    return np.abs(padded_image[s*(pad-1)//2:s*(pad+1)//2,
                               s*(pad-1)//2:s*(pad+1)//2])
