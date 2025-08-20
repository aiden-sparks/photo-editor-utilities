import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from utils import apply_channel_lut, apply_disposable_camera_filter
from utils import (
                   apply_channel_lut,
                   apply_point_lut,
                   adjust_highlights,
                   adjust_shadows,
                   adjust_midtones,
                   adjust_blacks,
                   adjust_whites,
                   adjust_tint,
                   adjust_temp
)


if __name__ == "__main__":
    img = Image.open("./TEST_IMGS/IMG_7363.jpg")
    out = apply_disposable_camera_filter(img)
    out.show()

    arr = np.asarray(img).astype(np.float32) / 255.0
    out_tint = adjust_temp(arr, 0.6)
    tinted_img = Image.fromarray((out_tint * 255).astype(np.uint8))
    tinted_img.show()

    # im = Image.open("./TEST_IMGS/IMG_2164.jpg").convert("RGBA")
    
    # # For histogram analysis, we need a numpy array
    # im_np = cv2.imread("./TEST_IMGS/IMG_2164.jpg")
    # # calculate mean value from RGB channels and flatten to 1D array
    # vals = im_np.mean(axis=2).flatten()
    # # calculate histogram
    # counts, bins = np.histogram(vals, range(257))
    # # plot histogram centered on values 0..255
    # plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
    # plt.xlim([-0.5, 255.5])
    # plt.show()

    # pt_lut_x = [0, 225]
    # pt_lut_y = [30, 255]
    # im = apply_point_lut(im, pt_lut_x, pt_lut_y)

    # g_lut_x = [0, 245]
    # g_lut_y = [10, 255]
    # im = apply_channel_lut(im, g_lut_x, g_lut_y, "G")

    # b_lut_x = [10, 255]
    # b_lut_y = [0, 255]
    # im = apply_channel_lut(im, b_lut_x, b_lut_y, "B")


    # im = adjust_highlights_cursor(im, 1, True)

    # im.show()

    # # Example usage:
    # img = cv2.imread("./TEST_IMGS/IMG_2164.jpg")
    # # out = adjust_highlights(img, 0.4)
    # out = adjust_blacks(img, -0.2)
    # cv2.imwrite("adjusted.jpg", out)
