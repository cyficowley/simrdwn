#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:11:56 2016

@author: avanetten

"""

from __future__ import print_function
import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
import argparse
import shutil
import copy
# import logging
# import tensorflow as tf

import utils
import post_process
import add_geo_coords
import slice_im

sys.stdout.flush()

###############################################################################
def yolt_command(yolt_cfg_file_tot='',
                 weight_file_tot='',
                 results_dir='',
                 yolt_loss_file='',
                 mode='train',
                 yolt_object_labels_str='',
                 yolt_classnum=1,
                 nbands=3,
                 gpu=-1,
                 single_gpu_machine=0,
                 yolt_train_images_list_file_tot='',
                 test_list_loc='',
                 test_im_tot='',
                 test_thresh=0.2,
                 yolt_nms_thresh=0,
                 min_retain_prob=0.025):
    

    ##########################

    c_arg_list = [
        './yolt3/darknet',
        0,
        "yolt3",
        "valid",
        yolt_cfg_file_tot,
        weight_file_tot,
        'null',
        "0",
        str(yolt_nms_thresh),
        yolt_train_images_list_file_tot,
        results_dir,
        test_list_loc,
        yolt_object_labels_str,
        str(yolt_classnum),
        str(nbands),
        yolt_loss_file,
        str(min_retain_prob)
    ]

    cmd = ' '.join(c_arg_list)

    print("Command:\n", cmd)

    return cmd



###############################################################################
def split_test_im(img, slided_image_dir, slice_sizes=[416], slice_overlap=0.2, test_slice_sep='__', zero_frac_thresh=0.5):

    image_list = slice_im.slice_im(img, slided_image_dir,
                        slice_sizes[0], slice_sizes[0],
                        zero_frac_thresh=zero_frac_thresh,
                        overlap=slice_overlap,
                        slice_sep=test_slice_sep)
    print("sliced image into {} peices".format(len(image_slices)))
    return image_list


###############################################################################
def prep_test_files(img, slice_sizes=[416], slice_overlap=0.2, test_slice_sep='__', zero_frac_thresh=0.5):
    t0 = time.time()
    image_list = split_test_im(img,
                          slice_sizes=slice_sizes,
                          slice_overlap=slice_overlap,
                          test_slice_sep=test_slice_sep,
                          zero_frac_thresh=zero_frac_thresh)


    cmd_time_str = 'Length of time to split test files: {} seconds'.format(t1 - t0)
    print(cmd_time_str)

    return image_list


###############################################################################
def run_test(framework='YOLT3',
             infer_cmd='',
             slice_sizes=[416],
             image_slices = None,
             label_map_dict={},
             val_df_path_init='',
             test_slice_sep='__',
             edge_buffer_test=1,
             max_edge_aspect_ratio=4,
             test_box_rescale_frac=1.0,
             rotate_boxes=False,
             min_retain_prob=0.025,
             test_list_loc=""):
    """Evaluate multiple large images"""

    t0 = time.time()
    os.system(infer_cmd)  # run_cmd(outcmd)
    t1 = time.time()


    df_tot = post_process.post_process_yolt_test_create_df(
        [], "",
        testims_dir_tot=test_list_loc,
        slice_sizes=slice_sizes,
        slice_sep=test_slice_sep,
        edge_buffer_test=edge_buffer_test,
        max_edge_aspect_ratio=max_edge_aspect_ratio,
        test_box_rescale_frac=test_box_rescale_frac,
        rotate_boxes=rotate_boxes)

    return df_tot

###############################################################################
def execute(args):
    """
    Execute train or test

    Arguments
    ---------
    args : Namespace
        input arguments
    train_cmd1 : str
        Training command
    test_cmd_tot : str
        Testing command
    test_cmd_tot2 : str
        Testing command for second scale (optional)

    Returns
    -------
    None
    """
    
    image_slices = prep_test_files(img=args.image, slice_sizes=args.slice_sizes, slice_overlap=args.slice_overlap)


    yolt_cmd = yolt_command(
        args.framework, yolt_cfg_file_tot=args.yolt_cfg_file_tot,
        weight_file_tot=args.weight_file_tot,
        yolt_object_labels_str=args.yolt_object_labels_str,
        yolt_classnum=args.yolt_classnum,
        yolt_train_images_list_file_tot=args.yolt_train_images_list_file_tot,
        test_list_loc=args.test_list_loc)

    df_tot = run_test(infer_cmd=yolt_cmd,
                            framework=args.framework,
                            slice_sizes=args.slice_sizes,
                            test_list_loc=args.test_list_loc,
                            label_map_dict=args.label_map_dict,
                            edge_buffer_test=args.edge_buffer_test)

    if len(df_tot) == 0:
        print("No detections found!")
    else:
        # save to csv
        df_tot.to_csv(args.val_df_path_aug, index=False)

    # refine for each plot_thresh (if we have detections)
    if len(df_tot) > 0:
        for plot_thresh_tmp in args.plot_thresh:
            groupby = 'Image_Path'
            groupby_cat = 'Category'
            df_refine = post_process.refine_df(df_tot,
                                                groupby=groupby,
                                                groupby_cat=groupby_cat,
                                                nms_overlap_thresh=args.nms_overlap_thresh,
                                                plot_thresh=plot_thresh_tmp,
                                                verbose=False)
            
            return post_process.plot_refined_df(df_refine, groupby=groupby, label_map_dict=args.label_map_dict_tot)


def main():
    args = { 
        "image": None,
        "slice_sizes": [824],
        "framework": "yolt3",
        "yolt_cfg_file_tot": "/simrdwn/yolt3/cfg/yolov3.cfg",
        "weight_file_tot": "/simrdwn/final_weights/current_weights.weights",
        "mode": "test",
        "yolt_object_labels_str": "/simrdwn/new_data/class_labels.pbtxt",
        "yolt_classnum": 3,
        "test_list_loc": "/simrdwn/data/test_data",
        "label_map_dict": {0:"yeet", 1:"dab", 2:"dank"},
        "edge_buffer_test": 1,
        "slice_overlap": 0.2
    }
   

    
    execute(args)


if __name__ == "__main__":

    main()
