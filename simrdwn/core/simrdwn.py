# to start gunicorn -b 0.0.0.0:80 simrdwn:app

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:11:56 2016

@author: avanetten

"""

import falcon
import json
import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
import argparse
import shutil
import copy
import cv2
import falcon
import utils
import post_process
import add_geo_coords
import slice_im
import base64


###############################################################################
def yolt_command(yolt_cfg_file_tot='',
                 weight_file_tot='',
                 results_dir='',
                 yolt_loss_file='',
                 mode='train',
                 label_map_dict='',
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
        '/simrdwn/yolt3/darknet',
        "-i",
        "0",
        "yolt3",
        "valid",
        yolt_cfg_file_tot,
        weight_file_tot,
        'null',
        "0",
        str(yolt_nms_thresh),
        'null',
        results_dir,
        test_list_loc,
        ",".join(label_map_dict.values()),
        str(yolt_classnum),
        str(nbands),
        yolt_loss_file,
        str(min_retain_prob)
    ]

    cmd = ' '.join(c_arg_list)

    print("Command:\n", cmd)

    return cmd



###############################################################################
def split_test_im(img, temp_dir, test_list_loc="", slice_sizes=[416], slice_overlap=0.2, test_slice_sep='__', zero_frac_thresh=0.5):

    image_list = slice_im.slice_im(img, temp_dir,
                        slice_sizes[0], slice_sizes[0],
                        zero_frac_thresh=zero_frac_thresh,
                        overlap=slice_overlap,
                        slice_sep=test_slice_sep)
    print("sliced image into {} peices".format(len(image_list)))
    
    with open(test_list_loc, "w") as f:
        for path in image_list:
            f.write(path + "\n")



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
             test_list_loc="",
             results_dir=""):
    """Evaluate multiple large images"""

    t0 = time.time()
    os.system(infer_cmd)
    t1 = time.time()

    yolt_test_classes_files = []
    for classes in label_map_dict.values():
        yolt_test_classes_files.append(os.path.join(results_dir, classes + ".txt"))

    df_tot = post_process.post_process_yolt_test_create_df(
        yolt_test_classes_files, "",
        testims_dir_tot=results_dir,
        slice_sizes=slice_sizes,
        slice_sep=test_slice_sep,
        edge_buffer_test=edge_buffer_test,
        max_edge_aspect_ratio=max_edge_aspect_ratio,
        test_box_rescale_frac=test_box_rescale_frac,
        rotate_boxes=rotate_boxes)

    return df_tot

###############################################################################
def execute(args):
    
    split_test_im(img=args["image"], test_list_loc=args["test_list_loc"], temp_dir=args["temp_dir"], slice_sizes=args["slice_sizes"], slice_overlap=args["slice_overlap"])


    yolt_cmd = yolt_command(yolt_cfg_file_tot=args["yolt_cfg_file_tot"],
        weight_file_tot=args["weight_file_tot"],
        label_map_dict=args["label_map_dict"],
        yolt_classnum=args["yolt_classnum"],
        test_list_loc=args["test_list_loc"],
        results_dir=args["temp_dir"])

    df_tot = run_test(infer_cmd=yolt_cmd,
                            framework=args["framework"],
                            slice_sizes=args["slice_sizes"],
                            test_list_loc=args["test_list_loc"],
                            label_map_dict=args["label_map_dict"],
                            edge_buffer_test=args["edge_buffer_test"],
                            results_dir=args["temp_dir"])


    # refine for each plot_thresh (if we have detections)
    if len(df_tot) > 0:
        groupby = 'Loc_Tmp'
        groupby_cat = 'Category'
        df_refine = post_process.refine_df(df_tot,
                                            groupby=groupby,
                                            groupby_cat=groupby_cat,
                                            nms_overlap_thresh=0.5,
                                            plot_thresh=args["confidence_threshold"],
                                            verbose=False)

        return post_process.get_final_data(df_refine, groupby=groupby, label_map_dict=args["label_map_dict"])


class GetDefects(object):
    def on_post(self, req, resp):
        query = falcon.uri.decode(req.query_string)
        queries = query.split("&")
        body = req.stream.read()
        b64 = json.loads(body)
        nparr = np.fromstring(base64.b64decode(b64), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR + cv2.IMREAD_IGNORE_ORIENTATION)

        args = { 
            "image": img,
            "slice_sizes": [824],
            "framework": "yolt3",
            "yolt_cfg_file_tot": "/simrdwn/yolt3/cfg/yolov3.cfg",
            "weight_file_tot": "/simrdwn/yolt3/final_weights/yolov3_final.weights",
            "mode": "test",
            "yolt_classnum": 3,
            "test_list_loc": "/simrdwn/temp/test_splitims_input_files.txt",
            "label_map_dict": {0:"dank", 1:"yeet", 2:"yote"},
            "edge_buffer_test": 1,
            "slice_overlap": 0.2,
            "temp_dir":"/simrdwn/temp",
            "confidence_threshold":0.2
        }

        output_array = execute(args)

        resp.status = falcon.HTTP_200
        resp.body = (json.dumps(output_array.tolist()))


app = falcon.API()
get_defects = GetDefects()
app.add_route('/get_defects', get_defects)
