#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function
from osgeo import ogr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import math
import time
import csv
import cv2
import os
# import pickle
# import sys

# path_simrdwn_core = os.path.dirname(os.path.realpath(__file__))
# import slice_im, convert, post_process scripts
# sys.path.append(path_simrdwn_core)
# import add_geo_cooords


###############################################################################
def get_global_coords(row,
                      edge_buffer_test=0,
                      max_edge_aspect_ratio=4,
                      test_box_rescale_frac=1.0,
                      rotate_boxes=False):
    """
    Get global pixel coords of bounding box prediction from dataframe row.

    Arguments
    ---------
    row : pandas dataframe row
        Prediction row from SIMRDWN
        columns:Index([u'Loc_Tmp', u'Prob', u'Xmin', u'Ymin',
                        u'Xmax', u'Ymax', u'Category',
                        u'Image_Root_Plus_XY', u'Image_Root', u'Slice_XY',
                        u'Upper', u'Left', u'Height', u'Width', u'Pad',
                        u'Image_Path']
    edge_buffer_test : int
        If a bounding box is within this distance from the edge, discard.
        Set edge_buffer_test < 0 keep all boxes. Defaults to ``0``.
    max_edge_aspect_ratio : int
        Maximum aspect ratio for bounding box for boxes near the window edge.
        Defaults to ``4``.
    test_box_rescale_frac : float
        Value by which to rescale the output bounding box.  For example, if
        test_box_recale_frac=0.8, the height and width of the box would be
        rescaled by 0.8.  Defaults to ``1.0``.
    rotate_boxes : boolean
        Switch to attempt to rotate bounding boxes.  Defaults to ``False``.

    Returns
    -------
    bounds, coords : tuple
        bounds = [xmin, xmax, ymin, ymax]
        coords = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    """

    xmin0, xmax0 = row['Xmin'], row['Xmax']
    ymin0, ymax0 = row['Ymin'], row['Ymax']
    upper, left = row['Upper'], row['Left']
    sliceHeight, sliceWidth = row['Height'], row['Width']
    vis_w, vis_h = row['Im_Width'], row['Im_Height']
    pad = row['Pad']

    if edge_buffer_test > 0:
        # skip if near edge (set edge_buffer_test < 0 to skip)
        if ((float(xmin0) < edge_buffer_test) or
            (float(xmax0) > (sliceWidth - edge_buffer_test)) or
            (float(ymin0) < edge_buffer_test) or
                (float(ymax0) > (sliceHeight - edge_buffer_test))):
            # print ("Too close to edge, skipping", row, "...")
            return [], []
        # skip if near edge and high aspect ratio (set edge_buffer_test < 0 to skip)
        elif ((float(xmin0) < edge_buffer_test) or
                (float(xmax0) > (sliceWidth - edge_buffer_test)) or
                (float(ymin0) < edge_buffer_test) or
                (float(ymax0) > (sliceHeight - edge_buffer_test))):
            # compute aspect ratio
            dx = xmax0 - xmin0
            dy = ymax0 - ymin0
            if (1.*dx/dy > max_edge_aspect_ratio) \
                    or (1.*dy/dx > max_edge_aspect_ratio):
                # print ("Too close to edge, and high aspect ratio, skipping", row, "...")
                return [], []

    # set min, max x and y for each box, shifted for appropriate
    #   padding
    xmin = max(0, int(round(float(xmin0)))+left - pad)
    xmax = min(vis_w - 1, int(round(float(xmax0)))+left - pad)
    ymin = max(0, int(round(float(ymin0)))+upper - pad)
    ymax = min(vis_h - 1, int(round(float(ymax0)))+upper - pad)

    # rescale output box size if desired, might want to do this
    #    if the training boxes were the wrong size
    if test_box_rescale_frac != 1.0:
        dl = test_box_rescale_frac
        xmid, ymid = np.mean([xmin, xmax]), np.mean([ymin, ymax])
        dx = dl*(xmax - xmin) / 2
        dy = dl*(ymax - ymin) / 2
        x0 = xmid - dx
        x1 = xmid + dx
        y0 = ymid - dy
        y1 = ymid + dy
        xmin, xmax, ymin, ymax = x0, x1, y0, y1

    # rotate boxes, if desird
    if rotate_boxes:
        # import vis
        vis = cv2.imread(row['Image_Path'], 1)  # color
        #vis_h,vis_w = vis.shape[:2]
        gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
        canny_edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        #lines = cv2.HoughLines(edges,1,np.pi/180,50)
        coords = _rotate_box(xmin, xmax, ymin, ymax, canny_edges)

    # set bounds, coords
    bounds = [xmin, xmax, ymin, ymax]
    coords = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]

    # check that nothing is negative
    if np.min(bounds) < 0:
        print("part of bounds < 0:", bounds)
        print(" row:", row)
        return
    if (xmax > vis_w) or (ymax > vis_h):
        print("part of bounds > image size:", bounds)
        print(" row:", row)
        return

    return bounds, coords


###############################################################################
def augment_df(df,
               testims_dir_tot='',
               slice_sizes=[416],
               slice_sep='|',
               edge_buffer_test=0,
               max_edge_aspect_ratio=4,
               test_box_rescale_frac=1.0,
               rotate_boxes=False,
               verbose=False):
    """
    Add global location columns to dataframe.

    Arguments
    ---------
    df : pandas dataframe
        Prediction dataframe from SIMRDWN
        Input columns are:
            ['Loc_Tmp', 'Prob','Xmin', 'Ymin', 'Xmax', 'Ymax', 'Category']
    testims_dir_tot : str
        Full path to location of testing images
    slice_sizes : list
        Window sizes.  Set to [0] if no slicing occurred.
        Defaults to ``[416]``.
    slice_sep : str
        Character used to separate outname from coordinates in the saved
        windows.  Defaults to ``|``
    edge_buffer_test : int
        If a bounding box is within this distance from the edge, discard.
        Set edge_buffer_test < 0 keep all boxes. Defaults to ``0``.
    max_edge_aspect_ratio : int
        Maximum aspect ratio for bounding box for boxes near the window edge.
        Defaults to ``4``.
    test_box_rescale_frac : float
        Value by which to rescale the output bounding box.  For example, if
        test_box_recale_frac=0.8, the height and width of the box would be
        rescaled by 0.8.  Defaults to ``1.0``.
    rotate_boxes : boolean
        Switch to attempt to rotate bounding boxes.  Defaults to ``False``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``

    Returns
    -------
    df : pandas dataframe
        Updated dataframe with global coords
    """

    extension_list = ['.png', '.tif', '.TIF', '.TIFF', '.tiff', '.JPG',
                      '.jpg', '.JPEG', '.jpeg']
    t0 = time.time()
    print("Augmenting dataframe of initial length:", len(df), "...")

    # extract image root?
    # df['Image_Root_Plus_XY'] = [f.split('/')[-1] for f in df['Loc_Tmp']]

    # parse out image root and location
    im_roots, im_locs = [], []
    # for j, f in enumerate(df['Image_Root_Plus_XY'].values):
    for j, loc_tmp in enumerate(df['Loc_Tmp'].values):

        if (j % 10000) == 0:
            print(j)

        f = loc_tmp.split('/')[-1]
        ext = f.split('.')[-1]
        # get im_root, (if not slicing ignore '|')
        if slice_sizes[0] > 0:
            im_root_tmp = f.split(slice_sep)[0]
            xy_tmp = f.split(slice_sep)[-1]
        else:
            im_root_tmp, xy_tmp = f, '0_0_0_0_0_0_0'
        if im_root_tmp == xy_tmp:
            xy_tmp = '0_0_0_0_0_0_0'
        im_locs.append(xy_tmp)

        if '.' not in im_root_tmp:
            im_roots.append(im_root_tmp + '.' + ext)
        else:
            im_roots.append(im_root_tmp)

    if verbose:
        print("loc_tmp[:3]", df['Loc_Tmp'].values[:3])
        print("im_roots[:3]", im_roots[:3])
        print("im_locs[:3]", im_locs[:3])

    df['Image_Root'] = im_roots
    df['Slice_XY'] = im_locs
    # get positions
    df['Upper'] = [float(sl.split('_')[0]) for sl in df['Slice_XY'].values]
    df['Left'] = [float(sl.split('_')[1]) for sl in df['Slice_XY'].values]
    df['Height'] = [float(sl.split('_')[2]) for sl in df['Slice_XY'].values]
    df['Width'] = [float(sl.split('_')[3]) for sl in df['Slice_XY'].values]
    df['Pad'] = [float(sl.split('_')[4].split('.')[0])
                 for sl in df['Slice_XY'].values]
    df['Im_Width'] = [float(sl.split('_')[5].split('.')[0])
                      for sl in df['Slice_XY'].values]
    df['Im_Height'] = [float(sl.split('_')[6].split('.')[0])
                       for sl in df['Slice_XY'].values]

    print("  set image path, make sure the image exists...")
    im_paths_list = []
    im_roots_update = []
    for ftmp in df['Image_Root'].values:
        # get image path
        im_path = os.path.join(testims_dir_tot, ftmp.strip())
        if os.path.exists(im_path):
            im_roots_update.append(os.path.basename(im_path))
            im_paths_list.append(im_path)
        # if this path doesn't exist, see if other extensions might work
        else:
            found = False
            for ext in extension_list:
                im_path_tmp = im_path.split('.')[0] + ext
                if os.path.exists(im_path_tmp):
                    im_roots_update.append(os.path.basename(im_path_tmp))
                    im_paths_list.append(im_path_tmp)
                    found = True
                    break
            if not found:
                print("im_path not found with test extensions:", im_path)
                print("   im_path_tmp:", im_path_tmp)
    # update columns
    df['Image_Path'] = im_paths_list
    df['Image_Root'] = im_roots_update
    # df['Image_Path'] = [os.path.join(testims_dir_tot, f.strip()) for f
    #                    in df['Image_Root'].values]

    if verbose:
        print("  Add in global location of each row")
    # if slicing, get global location from filename
    if slice_sizes[0] > 0:
        x0l, x1l, y0l, y1l = [], [], [], []
        bad_idxs = []
        for index, row in df.iterrows():
            print(row)
            bounds, coords = get_global_coords(
                row,
                edge_buffer_test=edge_buffer_test,
                max_edge_aspect_ratio=max_edge_aspect_ratio,
                test_box_rescale_frac=test_box_rescale_frac,
                rotate_boxes=rotate_boxes)
            if len(bounds) == 0 and len(coords) == 0:
                bad_idxs.append(index)
                [xmin, xmax, ymin, ymax] = 0, 0, 0, 0
            else:
                [xmin, xmax, ymin, ymax] = bounds
            x0l.append(xmin)
            x1l.append(xmax)
            y0l.append(ymin)
            y1l.append(ymax)
        df['Xmin_Glob'] = x0l
        df['Xmax_Glob'] = x1l
        df['Ymin_Glob'] = y0l
        df['Ymax_Glob'] = y1l
    # if not slicing, global coords are equivalent to local coords
    else:
        df['Xmin_Glob'] = df['Xmin'].values
        df['Xmax_Glob'] = df['Xmax'].values
        df['Ymin_Glob'] = df['Ymin'].values
        df['Ymax_Glob'] = df['Ymax'].values
        bad_idxs = []

    # remove bad_idxs
    if len(bad_idxs) > 0:
        print("removing bad idxs near junctions:", bad_idxs)
        df = df.drop(df.index[bad_idxs])

    print("Time to augment dataframe of length:", len(df), "=",
          time.time() - t0, "seconds")
    return df


###############################################################################
def post_process_yolt_test_create_df(yolt_test_classes_files, log_file,
                                     testims_dir_tot='',
                                     slice_sizes=[416],
                                     slice_sep='|',
                                     edge_buffer_test=0,
                                     max_edge_aspect_ratio=4,
                                     test_box_rescale_frac=1.0,
                                     rotate_boxes=False,
                                     verbose=False):
    """
    Create dataframe from yolt output text files.

    Arguments
    ---------
    yolt_test_classes_files : list
        List of output files (e.g. [boat.txt, car.txt]).
    log_file : str
        File for logging.
    testims_dir_tot : str
        Full path to location of testing images. Defaults to ``''``.
    slice_sizes : list
        Window sizes.  Set to [0] if no slicing occurred.
        Defaults to ``[416]``.
    slice_sep : str
        Character used to separate outname from coordinates in the saved
        windows.  Defaults to ``|``
    edge_buffer_test : int
        If a bounding box is within this distance from the edge, discard.
        Set edge_buffer_test < 0 keep all boxes. Defaults to ``0``.
    max_edge_aspect_ratio : int
        Maximum aspect ratio for bounding box for boxes near the window edge.
        Defaults to ``4``.
    test_box_rescale_frac : float
        Value by which to rescale the output bounding box.  For example, if
        test_box_recale_frac=0.8, the height and width of the box would be
        rescaled by 0.8.  Defaults to ``1.0``.
    rotate_boxes : boolean
        Switch to attempt to rotate bounding boxes.  Defaults to ``False``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``

    Returns
    -------
    df : pandas dataframe
        Output dataframe
        df.columns:
        Index([u'Loc_Tmp', u'Prob', u'Xmin', u'Ymin', u'Xmax', u'Ymax', u'Category',
        u'Image_Root_Plus_XY', u'Image_Root', u'Slice_XY', u'Upper', u'Left',
        u'Height', u'Width', u'Pad', u'Image_Path'],
        dtype='object')
    """

    # parse out files, create df
    df_tot = []

    for i, vfile in enumerate(yolt_test_classes_files):

        cat = vfile.split('/')[-1].split('.')[0]
        # load into dataframe
        df = pd.read_csv(vfile, sep=' ', names=['Loc_Tmp', 'Prob',
                                                'Xmin', 'Ymin', 'Xmax',
                                                'Ymax'])
        # set category
        df['Category'] = len(df) * [cat]

        # augment
        df = augment_df(df,
                        testims_dir_tot=testims_dir_tot,
                        slice_sizes=slice_sizes,
                        slice_sep=slice_sep,
                        edge_buffer_test=edge_buffer_test,
                        max_edge_aspect_ratio=max_edge_aspect_ratio,
                        test_box_rescale_frac=test_box_rescale_frac,
                        rotate_boxes=rotate_boxes)

        # append to total df
        if i == 0:
            df_tot = df
        else:
            df_tot = df_tot.append(df, ignore_index=True)

    return df_tot


###############################################################################
def refine_df(df, groupby='Image_Path',
              groupby_cat='Category',
              cats_to_ignore=[],
              use_weighted_nms=True,
              nms_overlap_thresh=0.5,  plot_thresh=0.5,
              verbose=True):
    """
    Remove elements below detection threshold, and apply non-max suppression.

    Arguments
    ---------
    df : pandas dataframe
        Augmented dataframe from augment_df()
    groupby : str
        Dataframe column indicating the image name or path.
        Defaults to ``'Image_Path'``
    groupby_cat : str
        Secondadary dataframe column to group by.  Can be used to group by
        category prior to NMS if one category is much larger than another
        (e.g. airplanes vs airports).  Set to '' to ignore.
        Defaults to ``'Category'``.
    cats_to_ignore : list
        List of categories to ignore.  Defaults to ``[]``.
    use_weighted_nms : boolean
        Switch to use weighted NMS. Defaults to ``True``.
    nms_overlap_thresh : float
        Overlap threshhold for non-max suppression. Defaults to ``0.5``.
    plot_thresh : float
        Minimum confidence to retain for plotting.  Defaults to ``0.5``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``True``.

    Returns
    -------
    df_tot : pandas dataframe
        Updated dataframe low confidence detections filtered out and NMS
        applied.
    """

    print("Running refine_df()...")
    t0 = time.time()

    # group by image, and plot
    group = df.groupby(groupby)
    count = 0
    # refine_dic = {}
    print_iter = 1
    df_idxs_tot = []
    for i, g in enumerate(group):

        img_loc_string = g[0]
        data_all_classes = g[1]

        # image_root = data_all_classes['Image_Root'].values[0]
        if (i % print_iter) == 0 and verbose:
            print(i+1, "/", len(group), "Processing image:", img_loc_string)
            print("  num boxes:", len(data_all_classes))

        # image = cv2.imread(img_loc_string, 1)
        # if verbose:
        #    print ("  image.shape:", image.shape)

        # apply a secondary filter
        # groupby category as well so that detections can be overlapping of
        # different categories (i.e.: a helicopter on a boat)
        if len(groupby_cat) > 0:
            group2 = data_all_classes.groupby(groupby_cat)
            for j, g2 in enumerate(group2):
                class_str = g2[0]

                # skip if class_str in cats_to_ignore
                if (len(cats_to_ignore) > 0) and (class_str in cats_to_ignore):
                    print("ignoring category:", class_str)
                    continue

                data = g2[1]
                df_idxs = data.index.values
                # classes_str = np.array(len(data) * [class_str])
                scores = data['Prob'].values

                if (i % print_iter) == 0 and verbose:
                    print("    Category:", class_str)
                    print("    num boxes:", len(data))
                    # print ("    scores:", scores)

                xmins = data['Xmin_Glob'].values
                ymins = data['Ymin_Glob'].values
                xmaxs = data['Xmax_Glob'].values
                ymaxs = data['Ymax_Glob'].values

                # set legend str?
                # if len(label_map_dict.keys()) > 0:
                #    classes_str = [label_map_dict[ztmp] for ztmp in classes_int]
                #    classes_legend_str = [str(ztmp) + ' = ' + label_map_dict[ztmp] for ztmp in classes_int]
                # else:
                #    classes_str = classes_int_str
                #    classes_legend_str = classes_str

                # filter out low probs
                high_prob_idxs = np.where(scores >= plot_thresh)
                scores = scores[high_prob_idxs]
                #classes_str = classes_str[high_prob_idxs]
                xmins = xmins[high_prob_idxs]
                xmaxs = xmaxs[high_prob_idxs]
                ymins = ymins[high_prob_idxs]
                ymaxs = ymaxs[high_prob_idxs]
                df_idxs = df_idxs[high_prob_idxs]

                boxes = np.stack((ymins, xmins, ymaxs, xmaxs), axis=1)

                if verbose:
                    print("len boxes:", len(boxes))

                ###########
                # NMS
                if nms_overlap_thresh > 0:

                    # try tf nms (always returns an empty list!)
                    # https://www.tensorflow.org/versions/r0.12/api_docs/python/image/working_with_bounding_boxes
                    #boxes_tf = tf.convert_to_tensor(boxes, np.float32)
                    #scores_tf = tf.convert_to_tensor(scores, np.float32)
                    # nms_idxs = tf.image.non_max_suppression(boxes_tf, scores_tf,
                    #                                        max_output_size=1000,
                    #                                        iou_threshold=0.5)
                    #selected_boxes = tf.gather(boxes_tf, nms_idxs)
                    #print ("  len boxes:", len(boxes))
                    #print ("  nms idxs:", nms_idxs)
                    #print ("  selected boxes:", selected_boxes)

                    # Try nms with pyimagesearch algorighthm
                    # assume boxes = [[xmin, ymin, xmax, ymax, ...
                    #   might want to split by class because we could have a car inside
                    #   the bounding box of a plane, for example
                    boxes_nms_input = np.stack(
                        (xmins, ymins, xmaxs, ymaxs), axis=1)
                    if use_weighted_nms:
                        probs = scores
                    else:
                        probs = []
                    # _, _, good_idxs = non_max_suppression(
                    good_idxs = non_max_suppression(
                        boxes_nms_input, probs=probs,
                        overlapThresh=nms_overlap_thresh)

                    if verbose:
                        print("num boxes_all:", len(xmins))
                        print("num good_idxs:", len(good_idxs))
                    if len(boxes) == 0:
                        print("Error, No boxes detected!")
                    boxes = boxes[good_idxs]
                    scores = scores[good_idxs]
                    df_idxs = df_idxs[good_idxs]
                    #classes = classes_str[good_idxs]

                df_idxs_tot.extend(df_idxs)
                count += len(df_idxs)

        # no secondary filter
        else:
            data = data_all_classes.copy()
            # filter out cats__to_ignore
            if len(cats_to_ignore) > 0:
                data = data[~data['Category'].isin(cats_to_ignore)]
            df_idxs = data.index.values
            #classes_str = np.array(len(data) * [class_str])
            scores = data['Prob'].values

            if (i % print_iter) == 0 and verbose:
                print("    num boxes:", len(data))
                # print ("    scores:", scores)

            xmins = data['Xmin_Glob'].values
            ymins = data['Ymin_Glob'].values
            xmaxs = data['Xmax_Glob'].values
            ymaxs = data['Ymax_Glob'].values

            # filter out low probs
            high_prob_idxs = np.where(scores >= plot_thresh)
            scores = scores[high_prob_idxs]
            #classes_str = classes_str[high_prob_idxs]
            xmins = xmins[high_prob_idxs]
            xmaxs = xmaxs[high_prob_idxs]
            ymins = ymins[high_prob_idxs]
            ymaxs = ymaxs[high_prob_idxs]
            df_idxs = df_idxs[high_prob_idxs]

            boxes = np.stack((ymins, xmins, ymaxs, xmaxs), axis=1)

            if verbose:
                print("len boxes:", len(boxes))

            ###########
            # NMS
            if nms_overlap_thresh > 0:
                # Try nms with pyimagesearch algorighthm
                # assume boxes = [[xmin, ymin, xmax, ymax, ...
                #   might want to split by class because we could have
                #   a car inside the bounding box of a plane, for example
                boxes_nms_input = np.stack(
                    (xmins, ymins, xmaxs, ymaxs), axis=1)
                if use_weighted_nms:
                    probs = scores
                else:
                    probs = []
                # _, _, good_idxs = non_max_suppression(
                good_idxs = non_max_suppression(
                    boxes_nms_input, probs=probs,
                    overlapThresh=nms_overlap_thresh)

                if verbose:
                    print("num boxes_all:", len(xmins))
                    print("num good_idxs:", len(good_idxs))
                boxes = boxes[good_idxs]
                scores = scores[good_idxs]
                df_idxs = df_idxs[good_idxs]
                # classes = classes_str[good_idxs]

            df_idxs_tot.extend(df_idxs)
            count += len(df_idxs)

    #print ("len df_idxs_tot:", len(df_idxs_tot))
    df_idxs_tot_final = np.unique(df_idxs_tot)
    #print ("len df_idxs_tot unique:", len(df_idxs_tot))

    # create dataframe
    if verbose:
        print("df idxs::", df.index)
        print("df_idxs_tot_final:", df_idxs_tot_final)
    df_out = df.loc[df_idxs_tot_final]

    t1 = time.time()
    print("Initial length:", len(df), "Final length:", len(df_out))
    print("Time to run refine_df():", t1-t0, "seconds")
    return df_out  # refine_dic


###############################################################################
def plot_refined_df(df, groupby='Image_Path', label_map_dict={}):

    print("Running plot_refined_df...")
    t0 = time.time()
    # get colormap, if plotting
    building_list = [["ImageId", "BuildingId", "PolygonWKT_Pix", "Confidence"]]

    group = df.groupby(groupby)
    # print_iter = 1
    for i, g in enumerate(group):

        img_loc_string = g[0]
        print("img_loc:", img_loc_string)

        # if '740351_3737289' not in img_loc_string:
        #    continue

        data_all_classes = g[1]
        # image = cv2.imread(img_loc_string, 1)
        # we want image as bgr (cv2 format)
        try:
            image = cv2.imread(img_loc_string, 1)
            # tst = image.shape
            print("  cv2: image.shape:", image.shape)
        except:
            img_sk = skimage.io.imread(img_loc_string)
            # make sure image is h,w,channels (assume less than 20 channels)
            if (len(img_sk.shape) == 3) and (img_sk.shape[0] < 20):
                img_mpl = np.moveaxis(img_sk, 0, -1)
            else:
                img_mpl = img_sk
            image = cv2.cvtColor(img_mpl, cv2.COLOR_RGB2BGR)
            print("  skimage: image.shape:", image.shape)

        # image_root = data_all_classes['Image_Root'].values[0]
        im_root = os.path.basename(img_loc_string)
        im_root_no_ext, ext = im_root.split('.')
        outfile = os.path.join(outdir, im_root_no_ext + '_thresh='
                               + str(plot_thresh) + '.' + ext)

        if (i % print_iter) == 0 and verbose:
            print(i+1, "/", len(group), "Processing image:", img_loc_string)
            print("  num boxes:", len(data_all_classes))
        # if verbose:
            print("  image.shape:", image.shape)

        xmins = data_all_classes['Xmin_Glob'].values
        ymins = data_all_classes['Ymin_Glob'].values
        xmaxs = data_all_classes['Xmax_Glob'].values
        ymaxs = data_all_classes['Ymax_Glob'].values
        classes = data_all_classes['Category']
        scores = data_all_classes['Prob']

        return np.stack((ymins, xmins, ymins - ymaxs, xmins - xmaxs, classes, scores), axis=1)

###############################################################################
def non_max_suppression(boxes, probs=[], overlapThresh=0.5):
    """
    Apply non-max suppression.
    Adapted from:
    http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    Malisiewicz et al.
    see modular_sliding_window.py, functions non_max_suppression, \
            non_max_supression_rot
    Another option:
        https://github.com/YunYang1994/tensorflow-yolov3/blob/master/core/utils.py

    Arguments
    ---------
    boxes : np.array
        Prediction boxes with the format: [[xmin, ymin, xmax, ymax], [...] ]
    probs : np.array
        Array of prediction scores or probabilities.  If [], ignore.  If not
        [], sort boxes by probability prior to applying non-max suppression.
        Defaults to ``[]``.
    overlapThresh : float
        minimum IOU overlap to retain.  Defaults to ``0.5``.

    Returns
    -------
    pick : np.array
        Array of indices to keep
    """

    print("Executing non-max suppression...")
    len_init = len(boxes)

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return [], [], []

    # boxes_tot = boxes  # np.asarray(boxes)
    boxes = np.asarray([b[:4] for b in boxes])
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # sort the boxes by the bottom-right y-coordinate of the bounding box
    if len(probs) == 0:
        idxs = np.argsort(y2)
    # sort boxes by the highest prob (descending order)
    else:
        idxs = np.argsort(probs)[::-1]

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs,
            np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    print("  non-max suppression init boxes:", len_init)
    print("  non-max suppression final boxes:", len(pick))
    return pick

#    # return only the bounding boxes that were picked using the
#    # integer data type
#    outboxes = boxes[pick].astype("int")
#    #outboxes_tot = boxes_tot[pick]
#    outboxes_tot = [boxes_tot[itmp] for itmp in pick]
#    return outboxes, outboxes_tot, pick


###############################################################################
def make_color_legend(outfile, label_map_dict, auto_assign_colors=True,
                      default_rgb_tuple=(255, 255, 0), verbose=False):
    """
    Create the color legend for each object category.

    Arguments
    ---------
    outfile : str
        Output path (e.g. /path/to/results/00_colormap_legend.png)
    label_map_dict : dict
        Dictionary matching category ints to category strings.
        Defaults to ``{}``.
    auto_assign_colors : boolean
        Switch to automatically assign colors. Defaults to ``True``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    None
    """

    if auto_assign_colors:
        # automatically assign colors?
        cmap = plt.cm.get_cmap('jet', len(list(label_map_dict.keys())))
        if verbose:
            print("len(list(label_map_dict.keys())):",
                  len(list(label_map_dict.keys())))
            print("cmap:", cmap)
            print("cmap.N:", cmap.N)
        # sometimes label_map_dict starts at 1, instead of 0
        if min(list(label_map_dict.keys())) == 1:
            idx_plus_val = 1
        else:
            idx_plus_val = 0
        colormap = []
        color_dict = {}

        # if just one object, use default color
        if len(label_map_dict.keys()) == 1:
            color_dict = {label_map_dict[1]:  default_rgb_tuple}
            colormap.append(default_rgb_tuple)

        else:
            for i in range(cmap.N):
                rgb = cmap(i)[:3]
                # hexa = matplotlib.colors.rgb2hex(rgb)
                # cmaplist.append(hexa)
                rgb_tuple = tuple([int(255*z) for z in rgb])
                colormap.append(rgb_tuple)
                color_dict[label_map_dict[i + idx_plus_val]] = rgb_tuple

        # for key in label_map_dict.keys():
        #    itmp = key
        #    color = colormap[itmp]
        #    color_dict[label_map_dict[key]] = color

        # # OPTIONAL, assign defaulr color as cyan
        # if list(label_map_dict.values()) == ['car']:
        #     color_dict = { 'car':  (255, 255, 0)}

    else:
        # assign colors
        colormap = [(255, 0,   0),
                    (0,   255, 0),
                    (0,   0,   255),
                    (255, 255, 0),
                    (0,   255, 255),
                    (255, 0,   255),
                    (0,   0,   255),
                    (127, 255, 212),
                    (72,  61,  139),
                    (255, 127, 80),
                    (199, 21,  133),
                    (255, 140, 0),
                    (0, 165, 255)]

        # manually assign colors?
        # cv2 colors are bgr not rgb:
        # https://www.webucator.com/blog/2015/03/python-color-constants-module
        color_dict = {
            'airplane':   (0,   255, 0),
            'boat':       (0,   0,   255),
            'car':        (255, 255, 0),
            'airport':    (255, 155,   0),
            'building':   (0, 0, 200),
        }

    h, w = 800, 400
    xpos = int(0.2*w)
    ydiff = int(0.05*h)
    # TEXT FONT
    # https://codeyarns.files.wordpress.com/2015/03/20150311_opencv_fonts.png
    font = cv2.FONT_HERSHEY_TRIPLEX  # FONT_HERSHEY_SIMPLEX
    font_size = 0.4
    label_font_width = 1

    # rescale height so that if we have a long list of categories it fits
    rescale_h = h * len(label_map_dict.keys()) / 18.
    hprime = max(h, int(rescale_h))
    img_mpl = 255*np.ones((hprime, w, 3))

    try:
        cv2.putText(img_mpl, 'Color Legend', (int(xpos), int(ydiff)), font,
                    1.5*font_size, (0, 0, 0), int(1.5*label_font_width),
                    # cv2.CV_AA)
                    cv2.LINE_AA)
    except:
        cv2.putText(img_mpl, 'Color Legend', (int(xpos), int(ydiff)), font,
                    1.5*font_size, (0, 0, 0), int(1.5*label_font_width),
                    cv2.LINE_AA)

    for key in label_map_dict.keys():
        itmp = key
        val = label_map_dict[key]
        color = color_dict[val]

        #color = colormap[itmp]
        #color_dict[label_map_dict[key]] = color

        text = '- ' + str(key) + ': ' + str(label_map_dict[key])
        ypos = 2 * ydiff + itmp * ydiff
        try:
            cv2.putText(img_mpl, text, (int(xpos), int(ypos)), font,
                        1.5*font_size, color, label_font_width,
                        cv2.CV_AA)
            # cv2.LINE_AA)
        except:
            cv2.putText(img_mpl, text, (int(xpos), int(ypos)), font,
                        1.5*font_size, color, label_font_width,
                        cv2.LINE_AA)

    cv2.imwrite(outfile, img_mpl)

    if verbose:
        print("post_process.py - make_color_legend() label_map_dict:", label_map_dict)
        print("post_process.py - make_color_legend() colormap:", colormap)
        print("post_process.py - make_color_legend() color_dict:", color_dict)

    return colormap, color_dict


###############################################################################
def _building_polys_to_csv(image_name, building_name, coords, conf=0,
                           asint=True, rotate_boxe=True, use2D=False):
    '''
    For spacenet data.
    coords should have format [[x0, y0], [x1, y1], ... ]
    Outfile should have format:
            ImageId,BuildingId,PolygonWKT_Pix,Confidence
    https://gis.stackexchange.com/questions/109327/convert-list-of-coordinates-to-ogrgeometry-or-wkt
    https://pcjericks.github.io/py-gdalogr-cookbook/geometry.html
    '''

    if asint:
        coords = np.array(coords).astype(int)

    ring = ogr.Geometry(ogr.wkbLinearRing)
    for coord in coords:
        ring.AddPoint(coord[0], coord[1])
    # add first point to close polygon
    ring.AddPoint(coords[0][0], coords[0][1])

    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    if use2D:
        # This doesn't work for some reason!!!!
        poly.flattenTo2D()
    wktpoly = poly.ExportToWkt()

    row = [image_name, building_name, wktpoly, conf]
    return row


###############################################################################
# Rotated boxes
###############################################################################
def _rotatePoint(centerPoint, point, angle):
    """Rotates a point around another centerPoint. Angle is in degrees.
    Rotation is counter-clockwise
    #http://stackoverflow.com/questions/20023209/python-function-for-rotating-2d-objects
    """
    angle = math.radians(angle)
    temp_point = point[0]-centerPoint[0], point[1]-centerPoint[1]
    temp_point = (temp_point[0]*math.cos(angle)-temp_point[1]*math.sin(angle),
                  temp_point[0]*math.sin(angle)+temp_point[1]*math.cos(angle))
    temp_point = temp_point[0]+centerPoint[0], temp_point[1]+centerPoint[1]
    # rotatePoint( (1,1), (2,2), 45)
    return temp_point


###############################################################################
def _rescale_angle(angle_rad):
    '''Transform theta to angle between -45 and 45 degrees
    expect input angle to be between 0 and pi radians'''
    angle_deg = round(180. * angle_rad / np.pi, 2)

    if angle_deg >= 0. and angle_deg <= 45.:
        angle_out = angle_deg
    elif angle_deg > 45. and angle_deg < 90.:
        angle_out = angle_deg - 90.
    elif angle_deg >= 90. and angle_deg < 135:
        angle_out = angle_deg - 90.
    elif angle_deg >= 135 and angle_deg < 180:
        angle_out = angle_deg - 180.
    else:
        print("Unexpected angle in rescale_angle() ",
              "[should be from 0-pi radians]")
        return

    return angle_out


###############################################################################
def _rotate_box(xmin, xmax, ymin, ymax, canny_edges, verbose=False):
    '''Rotate bonding box'''

    coords = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    centerPoint = (np.mean([xmin, xmax]), np.mean([ymin, ymax]))

    # get canny edges in desired window
    win_edges = canny_edges[ymin:ymax, xmin:xmax]

    # create hough lines
    hough_lines = cv2.HoughLines(win_edges, 1, np.pi/180, 20)
    if hough_lines is not None:
        # print "hough_lines:", hough_lines
        # get primary angle
        line = hough_lines[0]
        if verbose:
            print(" hough_lines[0]",  line)
        if len(line) > 1:
            rho, theta = line[0].flatten()
        else:
            rho, theta = hough_lines[0].flatten()
    else:
        theta = 0.
    # rescale to between -45 and +45 degrees
    angle_deg = _rescale_angle(theta)
    if verbose:
        print("angle_deg_rot:", angle_deg)
    # rotated coords
    coords_rot = np.asarray([_rotatePoint(centerPoint, c, angle_deg) for c in
                             coords], dtype=np.int32)

    return coords_rot
