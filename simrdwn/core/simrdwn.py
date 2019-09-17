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
def yolt_command(framework='yolt2',
                 yolt_cfg_file_tot='',
                 weight_file_tot='',
                 results_dir='',
                 log_file='',
                 yolt_loss_file='',
                 mode='train',
                 yolt_object_labels_str='',
                 yolt_classnum=1,
                 nbands=3,
                 gpu=0,
                 single_gpu_machine=0,
                 yolt_train_images_list_file_tot='',
                 test_splitims_locs_file='',
                 test_im_tot='',
                 test_thresh=0.2,
                 yolt_nms_thresh=0,
                 min_retain_prob=0.025):
    """
    Define YOLT commands
    yolt.c expects the following inputs:
    // arg 0 = GPU number
    // arg 1 'yolt'
    // arg 2 = mode
    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *test_filename = (argc > 5) ? argv[5]: 0;
    float plot_thresh = (argc > 6) ? atof(argv[6]): 0.2;
    float nms_thresh = (argc > 7) ? atof(argv[7]): 0;
    char *train_images = (argc > 8) ? argv[8]: 0;
    char *results_dir = (argc > 9) ? argv[9]: 0;
    //char *test_image = (argc >10) ? argv[10]: 0;
    char *test_list_loc = (argc > 10) ? argv[10]: 0;
    char *names_str = (argc > 11) ? argv[11]: 0;
    int len_names = (argc > 12) ? atoi(argv[12]): 0;
    int nbands = (argc > 13) ? atoi(argv[13]): 0;
    char *loss_file = (argc > 14) ? argv[14]: 0;
    """

        test_list_loc = test_splitims_locs_file
    else:
        # test_image = 'null'
        test_list_loc = 'null'

    ##########################

    c_arg_list = [
        './yolt3/darknet',
        0,
        "yolt3",  # 'yolt2',
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
        str(min_retain_prob),
        suffix
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
             min_retain_prob=0.05):
    """Evaluate multiple large images"""

    t0 = time.time()
    print("Running", infer_cmd)
    os.system('echo ' + infer_cmd + ' >> ' + log_file)
    os.system(infer_cmd)  # run_cmd(outcmd)
    t1 = time.time()
    cmd_time_str = '"\nLength of time to run command: ' + infer_cmd \
        + ' for ' + str(n_files) + ' cutouts: ' \
        + str(t1 - t0) + ' seconds\n"'
    print(cmd_time_str[1:-1])
    os.system('echo ' + cmd_time_str + ' >> ' + log_file)

    if framework.upper() not in ['YOLT2', 'YOLT3']:

        # if we ran inference with a tfrecord, we must now parse that into
        #   a dataframe
        if len(test_tfrecord_out) > 0:
            df_init = parse_tfrecord.tf_to_df(
                test_tfrecord_out, max_iter=500000,
                label_map_dict=label_map_dict, tf_type='test',
                output_columns=['Loc_Tmp', u'Prob', u'Xmin', u'Ymin',
                                u'Xmax', u'Ymax', u'Category'],
                # replace_paths=()
                )
            # use numeric categories
            label_map_dict_rev = {v: k for k, v in label_map_dict.items()}
            # label_map_dict_rev = {v: k for k,v in label_map_dict.iteritems()}
            df_init['Category'] = [label_map_dict_rev[vtmp]
                                   for vtmp in df_init['Category'].values]
            # save to file
            df_init.to_csv(val_df_path_init)
        else:
            print("Read in val_df_path_init:", val_df_path_init)
            df_init = pd.read_csv(val_df_path_init
                                  # names=[u'Loc_Tmp', u'Prob', u'Xmin', u'Ymin',
                                  #       u'Xmax', u'Ymax', u'Category']
                                  )

        #########
        # post process
        print("len df_init:", len(df_init))
        df_init.index = np.arange(len(df_init))

        # clean out low probabilities
        print("minimum retained threshold:",  min_retain_prob)
        bad_idxs = df_init[df_init['Prob'] < min_retain_prob].index
        if len(bad_idxs) > 0:
            print("bad idxss:", bad_idxs)
            df_init.drop(df_init.index[bad_idxs], inplace=True)

        # clean out bad categories
        df_init['Category'] = df_init['Category'].values.astype(int)
        good_cats = list(label_map_dict.keys())
        print("Allowed categories:", good_cats)
        # print ("df_init0['Category'] > np.max(good_cats)", df_init['Category'] > np.max(good_cats))
        # print ("df_init0[df_init0['Category'] > np.max(good_cats)]", df_init[df_init['Category'] > np.max(good_cats)])
        bad_idxs2 = df_init[df_init['Category'] > np.max(good_cats)].index
        if len(bad_idxs2) > 0:
            print("label_map_dict:", label_map_dict)
            print("df_init['Category']:", df_init['Category'])
            print("bad idxs2:", bad_idxs2)
            df_init.drop(df_init.index[bad_idxs2], inplace=True)

        # set index as sequential
        df_init.index = np.arange(len(df_init))

        # df_init = df_init0[df_init0['Category'] <= np.max(good_cats)]
        # if (len(df_init) != len(df_init0)):
        #    print (len(df_init0) - len(df_init), "rows cleaned out")

        # tf_infer_cmd outputs integer categories, update to strings
        df_init['Category'] = [label_map_dict[ktmp]
                               for ktmp in df_init['Category'].values]

        print("len df_init after filtering:", len(df_init))

        # augment dataframe columns
        df_tot = post_process.augment_df(
            df_init,
            testims_dir_tot=testims_dir_tot,
            slice_sizes=slice_sizes,
            slice_sep=test_slice_sep,
            edge_buffer_test=edge_buffer_test,
            max_edge_aspect_ratio=max_edge_aspect_ratio,
            test_box_rescale_frac=test_box_rescale_frac,
            rotate_boxes=rotate_boxes,
            verbose=True)

    else:
        # post-process
        # df_tot = post_process_yolt_test_create_df(args)
        df_tot = post_process.post_process_yolt_test_create_df(
            yolt_test_classes_files, log_file,
            testims_dir_tot=testims_dir_tot,
            slice_sizes=slice_sizes,
            slice_sep=test_slice_sep,
            edge_buffer_test=edge_buffer_test,
            max_edge_aspect_ratio=max_edge_aspect_ratio,
            test_box_rescale_frac=test_box_rescale_frac,
            rotate_boxes=rotate_boxes)

    ###########################################
    # plot

    # add geo coords to eall boxes?
    if test_add_geo_coords and len(df_tot) > 0:
        ###########################################
        # !!!!! Skip?
        # json = None
        ###########################################
        df_tot, json = add_geo_coords.add_geo_coords_to_df(
            df_tot, create_geojson=False, inProj_str='epsg:4326',
            outProj_str='epsg:3857', verbose=verbose)
    else:
        json = None

    return df_tot, json


###############################################################################
def prep(args):
    
    train_cmd1, test_cmd_tot, test_cmd_tot2 = '', '', ''

    yolt_cmd = yolt_command(
        args.framework, yolt_cfg_file_tot=args.yolt_cfg_file_tot,
        weight_file_tot=args.weight_file_tot,
        results_dir=args.results_dir,
        log_file=args.log_file,
        yolt_loss_file=args.yolt_loss_file,
        mode=args.mode,
        yolt_object_labels_str=args.yolt_object_labels_str,
        yolt_classnum=args.yolt_classnum,
        nbands=args.nbands,
        gpu=args.gpu,
        single_gpu_machine=args.single_gpu_machine,
        yolt_train_images_list_file_tot=args.yolt_train_images_list_file_tot,
        test_splitims_locs_file=args.test_splitims_locs_file,
        yolt_nms_thresh=args.yolt_nms_thresh,
        min_retain_prob=args.min_retain_prob)
    

    return "", yolt_cmd, ""


###############################################################################
def execute(outname, label_map_path, weight_file, yolo_cfg_file, testimg, boxes_per_grid, slice_sizes, slice_overlap):
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

    
    # need to split file for test first, then run command

    print("Prepping test files")
    image_slices = prep_test_files(img=testimg, slice_sizes=slice_sizes, slice_overlap=slice_overlap)

    df_tot, json = run_test(infer_cmd=test_cmd_tot,
                            framework=args.framework,
                            results_dir=args.results_dir,
                            log_file=args.log_file,
                            # test_files_locs_list=test_files_locs_list,
                            # test_presliced_tfrecord_tot=args.test_presliced_tfrecord_tot,
                            test_tfrecord_out=args.test_tfrecord_out,
                            slice_sizes=args.slice_sizes,
                            testims_dir_tot=args.testims_dir_tot,
                            yolt_test_classes_files=args.yolt_test_classes_files,
                            label_map_dict=args.label_map_dict,
                            val_df_path_init=args.val_df_path_init,
                            # val_df_path_aug=args.val_df_path_aug,
                            min_retain_prob=args.min_retain_prob,
                            test_slice_sep=args.test_slice_sep,
                            edge_buffer_test=args.edge_buffer_test,
                            max_edge_aspect_ratio=args.max_edge_aspect_ratio,
                            test_box_rescale_frac=args.test_box_rescale_frac,
                            rotate_boxes=args.rotate_boxes,
                            test_add_geo_coords=args.test_add_geo_coords)

    if len(df_tot) == 0:
        print("No detections found!")
    else:
        # save to csv
        df_tot.to_csv(args.val_df_path_aug, index=False)
        # get number of files
        n_files = len(np.unique(df_tot['Loc_Tmp'].values))
        # n_files = str(len(test_files_locs_list)
        t4 = time.time()
        cmd_time_str = '"Length of time to run test for ' \
            + str(n_files) + ' files = ' \
            + str(t4 - t3) + ' seconds\n"'
        print(cmd_time_str[1:-1])
        os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)


    # refine and plot
    t8 = time.time()
    if len(np.append(args.slice_sizes, args.slice_sizes2)) > 0:
        sliced = True
    else:
        sliced = False
    print("test data sliced?", sliced)

    # refine for each plot_thresh (if we have detections)
    if len(df_tot) > 0:
        for plot_thresh_tmp in args.plot_thresh:
            print("Plotting at:", plot_thresh_tmp)
            groupby = 'Image_Path'
            groupby_cat = 'Category'
            df_refine = post_process.refine_df(df_tot,
                                                groupby=groupby,
                                                groupby_cat=groupby_cat,
                                                nms_overlap_thresh=args.nms_overlap_thresh,
                                                plot_thresh=plot_thresh_tmp,
                                                verbose=False)
            # make some output plots, if desired
            if len(args.building_csv_file) > 0:
                building_csv_file_tmp = args.building_csv_file.split('.')[0] \
                    + '_plot_thresh_' + str(plot_thresh_tmp).replace('.', 'p') \
                    + '.csv'
            else:
                building_csv_file_tmp = ''
            if args.n_test_output_plots > 0:
                post_process.plot_refined_df(df_refine, groupby=groupby,
                                                label_map_dict=args.label_map_dict_tot,
                                                outdir=args.results_dir,
                                                plot_thresh=plot_thresh_tmp,
                                                show_labels=bool(
                                                    args.show_labels),
                                                alpha_scaling=bool(
                                                    args.alpha_scaling),
                                                plot_line_thickness=args.plot_line_thickness,
                                                print_iter=5,
                                                n_plots=args.n_test_output_plots,
                                                building_csv_file=building_csv_file_tmp,
                                                shuffle_ims=bool(
                                                    args.shuffle_val_output_plot_ims),
                                                verbose=False)

            # geo coords?
            if bool(args.test_add_geo_coords):
                df_refine, json = add_geo_coords.add_geo_coords_to_df(
                    df_refine,
                    create_geojson=bool(args.save_json),
                    inProj_str='epsg:32737', outProj_str='epsg:3857',
                    # inProj_str='epsg:4326', outProj_str='epsg:3857',
                    verbose=False)

            # save df_refine
            outpath_tmp = os.path.join(args.results_dir,
                                        args.val_prediction_df_refine_tot_root_part +
                                        '_thresh=' + str(plot_thresh_tmp) + '.csv')
            # df_refine.to_csv(args.val_prediction_df_refine_tot)
            df_refine.to_csv(outpath_tmp)
            print("Num objects at thresh:", plot_thresh_tmp, "=",
                    len(df_refine))
            # save json
            if bool(args.save_json) and (len(json) > 0):
                output_json_path = os.path.join(args.results_dir,
                                                args.val_prediction_df_refine_tot_root_part +
                                                '_thresh=' + str(plot_thresh_tmp) + '.GeoJSON')
                json.to_file(output_json_path, driver="GeoJSON")

        cmd_time_str = '"Length of time to run refine_test()' + ' ' \
            + str(time.time() - t8) + ' seconds"'
        print(cmd_time_str[1:-1])
        os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)

    # remove or zip test_split_dirs to save space
    if len(test_split_dir_list) > 0:
        for test_split_dir_tmp in test_split_dir_list:
            if os.path.exists(test_split_dir_tmp):
                # compress image chip dirs if desired
                if args.keep_test_slices:
                    print("Compressing image chips...")
                    shutil.make_archive(test_split_dir_tmp, 'zip',
                                        test_split_dir_tmp)
                # remove unzipped folder
                print("Removing test_split_dir_tmp:", test_split_dir_tmp)
                # make sure that test_split_dir_tmp hasn't somehow been shortened
                #  (don't want to remove "/")
                if len(test_split_dir_tmp) < len(args.results_dir):
                    print("test_split_dir_tmp too short!!!!:",
                            test_split_dir_tmp)
                    return
                else:
                    print("Removing image chips...")

                    shutil.rmtree(test_split_dir_tmp, ignore_errors=True)

    cmd_time_str = '"Total Length of time to run test' + ' ' \
        + str(time.time() - t3) + ' seconds\n"'
    print(cmd_time_str[1:-1])
    os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)

    # print ("\nNo honeymoon. This is business.")
    print("\n\n\nWell, I'm glad we got that out of the way.\n\n\n\n")

    return


###############################################################################
def main():

    
    yolt_cmd = yolt_command(
        "yolt3", 
        yolt_cfg_file_tot=args.yolt_cfg_file_tot,
        weight_file_tot=args.weight_file_tot,
        results_dir=args.results_dir,
        log_file=args.log_file,
        yolt_loss_file=args.yolt_loss_file,
        mode=args.mode,
        yolt_object_labels_str=args.yolt_object_labels_str,
        yolt_classnum=args.yolt_classnum,
        nbands=args.nbands,
        gpu=args.gpu,
        single_gpu_machine=args.single_gpu_machine,
        yolt_train_images_list_file_tot=args.yolt_train_images_list_file_tot,
        test_splitims_locs_file=args.test_splitims_locs_file,
        yolt_nms_thresh=args.yolt_nms_thresh,
        min_retain_prob=args.min_retain_prob)
    execute(yolt_cmd)


###############################################################################
###############################################################################
if __name__ == "__main__":

    print("\n\n\nPermit me to introduce myself...\n")
    main()

###############################################################################
###############################################################################
