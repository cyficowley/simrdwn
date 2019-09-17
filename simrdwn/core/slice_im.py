from __future__ import print_function
import os
import cv2
import time
import numpy as np
import shutil

###############################################################################
def slice_im(img, out_dir ,sliceHeight=256, sliceWidth=256,
             zero_frac_thresh=0.2, overlap=0.2, slice_sep='|'):

    use_cv2 = True
    image0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    win_h, win_w = image0.shape[:2]

    # if slice sizes are large than image, pad the edges
    pad = 0
    if sliceHeight > win_h:
        pad = sliceHeight - win_h
    if sliceWidth > win_w:
        pad = max(pad, sliceWidth - win_w)
    # pad the edge of the image with black pixels
    if pad > 0:
        border_color = (0, 0, 0)
        image0 = cv2.copyMakeBorder(image0, pad, pad, pad, pad,
                                    cv2.BORDER_CONSTANT, value=border_color)

    win_size = sliceHeight*sliceWidth

    t0 = time.time()
    n_ims = 0
    n_ims_nonull = 0
    dx = int((1. - overlap) * sliceWidth)
    dy = int((1. - overlap) * sliceHeight)

    image_list = []
    if(os.path.isdir(out_dir)):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    for y0 in range(0, image0.shape[0], dy):  # sliceHeight):
        for x0 in range(0, image0.shape[1], dx):  # sliceWidth):
            n_ims += 1
            # make sure we don't have a tiny image on the edge
            if y0+sliceHeight > image0.shape[0]:
                y = image0.shape[0] - sliceHeight
            else:
                y = y0
            if x0+sliceWidth > image0.shape[1]:
                x = image0.shape[1] - sliceWidth
            else:
                x = x0

            # extract image
            window_c = image0[y:y + sliceHeight, x:x + sliceWidth]
            # get black and white image
            window = cv2.cvtColor(window_c, cv2.COLOR_BGR2GRAY)

            outpath = os.path.join(
                    out_dir,
                    str(n_ims) + slice_sep + str(y) + '_' + str(x) + '_'
                    + str(sliceHeight) + '_' + str(sliceWidth)
                    + '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h)
                    + ".png")

            cv2.imwrite(outpath, window_c)
            image_list.append(outpath)

    return image_list
