"""
This script displays open pose [2] results on a selected MMDB [1] Rapid-ABC
session. Given the session name, you can iterate over all the frames in the
session to visualize open pose results:

@author: Ahmad Humayun
@email: ahumayun@cc.gatech.edu
@date: Feb 2018


[1] MMDB - http://www.cbi.gatech.edu/mmdb/
@inproceedings{rehg2013decoding,
  title = {Decoding children's social behavior},
  author = {Rehg, James M and Abowd, Gregory D and Rozga, Agata and Romero,
            Mario and Clements, Mark A and Sclaroff, Stan and Essa, Irfan and
            Ousley, Opal Y and Li, Yin and Kim, Chanho and others},
  booktitle = {CVPR},
  year = {2013},
}

[2] Open pose - https://github.com/CMU-Perceptual-Computing-Lab/openpose
@inproceedings{wei2016cpm,
  title = {Convolutional pose machines},
  author = {Shih-En Wei and Varun Ramakrishna and Takeo Kanade and Yaser Sheikh},
  booktitle = {CVPR},
  year = {2016}
}
"""

import os
import sys
import re
import glob
import argparse
import json
import logging
import traceback
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

script_path = os.path.dirname(os.path.realpath(__file__))
default_op_dir = os.path.join(script_path, "openpose_results")
sess_meta_fp = os.path.join(script_path, "sess_vids_meta.npy")
cv2_wnd_name = "Openpose results"
sess_data_input = []				# nx126 array
sess_data_output = []				# nx1 array


# These are joints between open pose keypoints. See:
# https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
# joints on body keypoints
conn_pose_jnt = ((0,1), (0,14), (0,15), (14,16), (15,17), (1,2), (1,5), (2,3), (3,4), \
                 (5,6), (6,7), (1,8), (1,11), (8,9), (11,12), (9,10), (12,13))
# joints on hands
conn_hand_jnt = ((0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (0,9), (9,10), \
                 (10,11), (11,12), (0,13), (13,14), (14,15), (15,16), (0,17), (17,18), \
                 (18,19), (19,20))
# this is the list of all keypoints except the arms
not_arm_kp_idxs = (0, 1, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)


def _return_jet_clr(i):
    """Returns the jet color (from MATLAB colormaps) given a a value [0,1]
    """
    # red
    r1 = (float(i) - 3.0/8) / (5.0/8 - 3.0/8)
    r2 = (float(i) - 9.0/8) / (7.0/8 - 9.0/8)
    r = min([r1, r2, 1.0])
    r = max(r, 0.0)
    # green
    g1 = (float(i) - 1.0/8) / (3.0/8 - 1.0/8)
    g2 = (float(i) - 7.0/8) / (5.0/8 - 7.0/8)
    g = min([g1, g2, 1.0])
    g = max(g, 0.0)
    # green
    b1 = (float(i) + 1.0/8) / (1.0/8 + 1.0/8)
    b2 = (float(i) - 5.0/8) / (3.0/8 - 5.0/8)
    b = min([b1, b2, 1.0])
    b = max(b, 0.0)

    return int(r*255), int(g*255), int(b*255)


def create_blank_frame(sess_name, vid_meta):
    """This function generates a blank numpy (unit8) frame of the same size as
    the video frames of [sess_name].
    """
    frm_h, frm_w = vid_meta["frame_sz"]
    frame = np.zeros((frm_h, frm_w, 3), np.uint8)
    return frame


def draw_pose(pose_data, frame, conn_jnt, vis_conf_th=0.05, fltr_kp=(), \
              pnt_sz=7, line_sz=5):
    """
    @brief      This function draws the pose given some pose data. This could
                be pose data for the body or the hand. It draws circles at
                keypoints and lines between pairs of connected keypoints/
                body parts.
    
    @param      pose_data  An np array of size (x,3) containing all the pose
                           information that needs to be drawn.
    @param      frame      This is the np array frame on which all the pose
                           is drawn.
    @param      conn_jnt   A list of tuples, where each tuple gives two
                           keypoint indices which are connected by a joint.
    @param      vis_conf_th Only keypoints above this threshold are displayed.
    @param      fltr_kp    This is a list/tuple of indices which contain indices
                           of all keypoints that shouldn't be displayed.
    @param      pnt_sz     The point size to use for drawing keypoints.
    @param      line_sz    The line width to use for drawing joints between
                           keypoints.
    """
    pose_pnt_seen = [False] * pose_data.shape[0]

    # display all the body keypoint which has above a certain confidence
    for kpi in range(pose_data.shape[0]):
        x, y, conf = pose_data[kpi,0], pose_data[kpi,1], pose_data[kpi,2]
        x, y = int(round(x)), int(round(y))
        # only display point if above a certain threshold and is not in the
        # filter display list/tuple
        if conf > vis_conf_th and kpi not in fltr_kp:
            clr = _return_jet_clr(float(kpi) / pose_data.shape[0])
            cv2.circle(frame, (x,y), pnt_sz, clr, -1)
            pose_pnt_seen[kpi] = True

    # display a joint between keypoints if both keypoints are visible
    for j, kp_idxs in enumerate(conn_jnt):
        kpi1, kpi2 = kp_idxs[0], kp_idxs[1]
        if pose_pnt_seen[kpi1] and pose_pnt_seen[kpi2]:
            x1, y1 = pose_data[kpi1,0], pose_data[kpi1,1]
            x1, y1 = int(round(x1)), int(round(y1))
            x2, y2 = pose_data[kpi2,0], pose_data[kpi2,1]
            x2, y2 = int(round(x2)), int(round(y2))

            clr = _return_jet_clr(float(kpi2) / pose_data.shape[0])
            cv2.line(frame, (x1,y1), (x2,y2),clr, line_sz)


def disp_pose_for_sess(sess_name, sess_dir, vid_meta, opts, logger):
    """
    @brief      This function displays the open pose results for all the frames
                in an MMDB session.

    @param      sess_name  The session's name
    @param      sess_dir   The session directory storing the json files
    @param      vid_meta   This session's meta data dictionary
    @param      opts       Options passed from the arguments
    @param      logger     Logger to print info
    """
    # create an opencv window
    # cv2.namedWindow(cv2_wnd_name)

    # count the number of frames in this session
    json_files = glob.glob(os.path.join(sess_dir,"*.json"))
    n_frames = len(json_files)

    # figure out the file-pattern for the json files
    json_filepattern = json_files[0]
    mtch = re.findall(r"_(\d+)_keypoints.json", os.path.basename(json_filepattern))
    assert len(mtch) == 1, "Can't figure out the json filepattern: %s" % sess_name
    json_filepattern = json_filepattern.replace(mtch[0], "%0"+str(len(mtch[0]))+"d")

    logger.info("'%s' session has %d frames", sess_name, n_frames)

    # check if we have all the json files
    assert vid_meta["out_num_frames"] == n_frames, \
        "Can't find all open pose json files for session: " + sess_name
    # check if we have all child idxs
    assert vid_meta["out_num_frames"] == vid_meta["child_data_idxs"].size, \
        "Child idxs meta data doesn't match number of frames"

    # initialize the frame and other variables
    # disp_frame = create_blank_frame(sess_name, vid_meta)

    # the frame number on which we are currently [0, n_frames-1]
    frame_idx = 0

    # iterate over all the frames in this video (for each frame we have a json
    # file containing the open pose results)
    while frame_idx < n_frames:
        # get the json filepath for this frame
        curr_json_fp = json_filepattern % frame_idx
        assert os.path.exists(curr_json_fp), \
            "Openpose json file doesn't exist: %s" % curr_json_fp

        # read the open pose json file for this frame
        with open(curr_json_fp, 'r') as fd:
            json_data = json.load(fd)

        # iterate over all the people
        people_data = json_data[u"people"]

        # get the index for the child
        child_idx = vid_meta["child_data_idxs"][frame_idx]

        # find out if this is a pointing frame
        sess_frame_idx = vid_meta["start_keyframe_num"] + frame_idx
        is_pointing = False
        for stf, enf in vid_meta["pointing_frames"]:
            if stf <= sess_frame_idx and sess_frame_idx <= enf:
                is_pointing = True
                break

        # reset the frame so we can draw over it
        if not opts.nopointing and is_pointing:
            # disp_frame.fill(50)
            # display pointing txt
            pointing = [1]
            # cv2.putText(disp_frame, "+ Pointing", (6, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        else:
            # disp_frame.fill(0)
            pointing = [0]

        # display frame number
        lbl_txt = "Frame " + str(frame_idx)
        # cv2.putText(disp_frame, lbl_txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        # onlt display child pose data if there is a valid pose
        if child_idx != -1:
            # get the child pose data
            person_data = people_data[child_idx]

            # get the body pose keypoints
            pose_data = np.array(person_data[u"pose_keypoints"])
            pose_data = np.reshape(pose_data, (-1, 3))
            assert pose_data.shape[0] == 18, \
                "Pose data is not 18 pnts: " + curr_json_fp

            # get the left hand keypoints
            lh_data = person_data[u"hand_left_keypoints"]
            lh_data = np.reshape(lh_data, (-1, 3))
            assert lh_data.shape[0] == 21, \
                "Left hand data is not 21 pnts: " + curr_json_fp

            # get the right hand keypoints
            rh_data = person_data[u"hand_right_keypoints"]
            rh_data = np.reshape(rh_data, (-1, 3))
            assert rh_data.shape[0] == 21, \
                "Right hand data is not 21 pnts: " + curr_json_fp

            frame_data = np.concatenate((lh_data.flatten(),rh_data.flatten()),axis=0)
            # frame_data = np.concatenate((frame_data,pointing),axis=0)
            sess_data_input.append(frame_data)
            sess_data_output.append(pointing)

            # get not-arm body keypoints, if --onlyarms
            flt_body_kp = not_arm_kp_idxs if opts.onlyarms else ()

            # display body keypoints
            # draw_pose(pose_data, disp_frame, conn_pose_jnt, opts.openposevisth, \
            #           fltr_kp=flt_body_kp, pnt_sz=7, line_sz=5)
            # # display left hand keypoints
            # draw_pose(lh_data, disp_frame, conn_hand_jnt, opts.openposevisth, \
            #           pnt_sz=2, line_sz=1)
            # # display right hand keypoints
            # draw_pose(rh_data, disp_frame, conn_hand_jnt, opts.openposevisth, \
            #           pnt_sz=2, line_sz=1)

        # display helper key info at start and end
        if frame_idx == 0 or frame_idx == n_frames-1:
            logger.info("d/right arrow: +1 frame  |  a/left arrow: +1 frame  |  "
                        "w/up arrow: +10 frames  |  s/down arrow: -10 frames  |  "
                        "q/x: quit")

        # display frame and wait for keypress
        # cv2.imshow(cv2_wnd_name, disp_frame)
        # keypress = cv2.waitKey(0)
        # keypress = keypress & 255

        # # manage key presses
        # if keypress in [ord('q'), ord('x')]:
        #     # q/x: quit
        #     break
        # elif keypress in [ord('d'), 83]:
        #     # d, fwd arrow: go to next frame
        #     frame_idx = min(frame_idx + 1, n_frames-1)
        # elif keypress in [ord('a'), 81]:
        #     # a, bwd arrow: go to previous frame
        #     frame_idx = max(frame_idx - 1, 0)
        # elif keypress in [ord('w'), 82]:
        #     # w, up arrow: +10 frames
        #     frame_idx = min(frame_idx + 10, n_frames-1)
        # elif keypress in [ord('s'), 84]:
        #     # s, down arrow: -10 frames
        #     frame_idx = max(frame_idx - 10, 0)
        frame_idx = frame_idx + 1

    # close all open windows
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    # set the logger
    logger = logging.getLogger(os.path.basename(__file__))
    logger.setLevel(logging.DEBUG)
    _hdlr = logging.StreamHandler()
    _formatter = logging.Formatter('%(name)-12s: %(levelname)-8s - %(message)s')
    _hdlr.setFormatter(_formatter)
    _hdlr.setLevel(logging.DEBUG)
    logger.addHandler(_hdlr)

    # create teh argument parser
    parser = argparse.ArgumentParser(description=
"--------------------------- Starter Code for Rapid-ABC -----------------------"
"------------------------------------------------------------------------------"
"This script is starter code for reading/analyzing the openpose json "
"results.",)

    parser.add_argument("-v", "--visualize", type=str, metavar="INPUTSESSION",
                        help="Visualize the videos for session")
    parser.add_argument("-T", "--train", type=str, metavar="TRAINSESSION",
                        help="Train the videos for session")
    parser.add_argument("-d", "--openposedir", type=str, metavar="OPENPOSEDIR",
                        default=default_op_dir,
                        help="This is the directory where openpose is")
    parser.add_argument("-t", "--openposevisth", type=float, metavar="OPENPOSETHRESHOLD",
                        default=0.05,
                        help="This sets the openpose visualization threshold")
    parser.add_argument("-a", "--onlyarms", action="store_true",
                        help="This would force the visualizer only to display arms "
                             "and hands.")
    parser.add_argument("-n", "--nopointing", action="store_true",
                        help="This would force the visualizer not to indicate which "
                             "frames have pointing behavior.")

    # parse all the program arguments
    args = parser.parse_args()

    try:
    	for subdir, dirs, files in os.walk(args.openposedir):
    		for sub in dirs:
		        sess_name = sub
		        print(sub)
		        # get the session directory
		        sess_dir = os.path.join(subdir, sess_name)
		        # print(sess_dir)
		        assert os.path.exists(sess_dir), \
		            "Session directory %s doesn't exist" % sess_dir

		        # load the this session's meta information
		        assert os.path.exists(sess_meta_fp), \
		            "Session meta file missing %s" % sess_meta_fp
		        sess_vid_meta = np.load(sess_meta_fp)
		        sess_vid_meta = sess_vid_meta.item()
		        if sess_name not in sess_vid_meta:
		        	# print("absent")
		        	continue
		        assert sess_name in sess_vid_meta, \
		            "Can't find meta information for video " + sess_name
		        vid_meta = sess_vid_meta[sess_name]

		        # display the pose for all the frames
		        disp_pose_for_sess(sess_name, sess_dir, vid_meta, args, logger)
		        print(np.shape(sess_data_input))
		        # print(sess_data_input[1530])
		clf = RandomForestClassifier()
		print np.mean(cross_val_score(clf,sess_data_input,sess_data_output,cv=10))

    except Exception as _err:
        logger.error(_err)
        traceback.print_exc(file=sys.stderr)
        sys.exit(-2)
