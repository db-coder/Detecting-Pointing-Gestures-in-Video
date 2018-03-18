"""
This script can be used to train a classifier based on OpenPose [2] results
for inferring whether a sample frame(s) from MMDB [1] Rapid-ABC session is
"pointing" or not. The script contains all the code to do K-fold
cross-validation, for training and testing a single frame or multi-frame
(sliding window) classifier. You can specify which project part to work on
using the command line arguments. For instance, if you want to train a
classifier for part 3, you can run this script as
    python train_test_rabc.py -p 3

According to the project part specified, this script decides how many frames
would be used in a sample, and what would be the sampling stride. It also
collects all the OpenPose data for you for the frames that are contained in a
sample. For part 1 and 2, a sample's data only comes from OpenPose results
from a single frame. Once you have trained the classifier, the code would
generate the precision recall curves and average precision numbers over all
the K-folds in cross-validation. These curves and numbers need to be
there in your project submission.

Your job is to at least fill write code at three places (look for the ######
markers):
(1) Fill in the function get_sample_feat(), where you are given all the pose
    data from all the frames - your task is to take all the pose data, and
    generate a feature vector which you think will be appropriate for the
    classification task. Currently, the function just vectorizes all the pose
    data, and uses it as a feature vector. If you change the feature
    dimension/size, you would need to adjust the function get_n_feats().
(2) Fill in the space in the function train(), where you would be provided
    the features computed from get_sample_feat() for all the samples
    (train_X), as well as the label for each sample (train_Y). Your task is
    to use these tensors to train a classifier, and store it in
    [trained_model] dictionary.
(3) Fill in the function test_sample(), which takes a feature (generated
    using get_sample_feat()) for a some sample, and should use the
    [trained_model] to infer whether the sample is positive (pointing) or
    negative. Currently, it returns a random number in the range [0,1].

Send me an email for concerns/questions.

@author: Ahmad Humayun
@email: ahumayun@cc.gatech.edu
@date: March 2018


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
import math
import random
from tqdm import tqdm
import numpy as np
from sklearn import metrics as skmetrics
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

script_path = os.path.dirname(os.path.realpath(__file__))
default_op_dir = os.path.join(script_path, "openpose_results")
sess_meta_fp = os.path.join(script_path, "sess_vids_meta2.npy")

# set the logger. you can use this to log anything to the console
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)
_hdlr = logging.StreamHandler()
_formatter = logging.Formatter('%(levelname)-8s - %(message)s')
_hdlr.setFormatter(_formatter)
_hdlr.setLevel(logging.DEBUG)
logger.addHandler(_hdlr)

# this is the complete session list (DON'T CHANGE THIS)
all_sessions = ['RA076_complete', 'RA091_FU_complete', 'RA138_complete', \
                'RA079_FU_complete', 'RA181_ams', 'RA113_complete', \
                'RA038_FU_complete', 'RA025_complete', 'RA125_complete', \
                'RA160_complete', 'RA056_FU_complete', 'RA168_complete', \
                'RA097_FU_revised', 'RA137_complete', 'RA049_FU_complete', \
                'RA149_FU_complete', 'RA154_aw_revised', 'RA139_complete', \
                'RA030_complete', 'RA177_revised', 'RA048_FU_complete', \
                'RA147_complete', 'RA072_FU_complete', 'RA155_FU_complete', \
                'RA031_complete', 'RA110_complete', 'RA087_FU_complete', \
                'RA140_complete', 'RA101_complete', 'RA060_FU_complete', \
                'RA122_complete', 'RA081_complete', 'RA157_cp']

# this is the list of all keypoints on the body that belong to arms
body_arm_kp_idxs = (2, 3, 4, 5, 6, 7)
n_body_pnts = 18
n_hand_pnts = 21


def get_cv_grps_stats(sess_cv_grps, sess_vids_meta):
    """
    This function prints some statistics given cross-validation session groups.
    @param  sess_cv_grps    Is a list of tuples of session names. Each tuple
                            contains all the session names in a certain cross
                            validation group.
    """
    # iterate over all groups
    for i, cv_grp in enumerate(sess_cv_grps):
        # keep track of the number of frames in this cv group
        tot_frames = 0
        # keep track of the total number of pointing gestures
        tot_pnt_gest = 0
        # keep track of the total number of pointing frames
        tot_pnt_frames = 0
        for sess_name in cv_grp:
            # check whether the session directory exists
            sess_dir = os.path.join(args.openposedir, sess_name)
            assert os.path.exists(sess_dir), \
                "Session directory %s doesn't exist" % sess_dir

            # load this session's meta information
            assert sess_name in sess_vids_meta, \
                "Can't find meta information for video " + sess_name
            vid_meta = sess_vids_meta[sess_name]

            # keep track of total frames
            nframes = vid_meta["out_num_frames"]
            tot_frames += nframes
            # keep track of total gestures
            pnt_frames = vid_meta["pointing_frames"]
            tot_pnt_gest += len(pnt_frames)
            # keep track of total # of pointing frames (+ve frame samples)
            tot_pnt_frames += sum([en-st+1 for st,en in pnt_frames])

        logger.info("CV group %d has a total of %d frames", i, tot_frames)
        logger.info("\tThis group has total %d pointing gestures", tot_pnt_gest)
        logger.info("\tThat gives a total of %d pointing frames (+ve samples)", \
                    tot_pnt_frames)
        logger.info("\tand a total of %d non-pointing frames (-ve samples)\n", \
                    tot_frames-tot_pnt_frames)


def get_json_filepattern(sess_name, opts):
    """This function gets json file-pattern for a certain session
    @param  sess_name    String session name.
    @param  opts         Stores the OpenPose results directory
    @returns json_filepattern   The file pattern which can be used as
                                json_filepattern % frame_idx to get the full
                                filepath to OpenPose results for a certain frame
    @returns n_frames           Number of frame associated with this session
    """
    # get the session directory
    sess_dir = os.path.join(opts.openposedir, sess_name)
    assert os.path.exists(sess_dir), \
        "Session directory %s doesn't exist" % sess_dir

    # count the number of frames in this session
    json_files = glob.glob(os.path.join(sess_dir,"*.json"))
    n_frames = len(json_files)

    # figure out the file-pattern for the json files
    json_filepattern = json_files[0]
    mtch = re.findall(r"_(\d+)_keypoints.json", os.path.basename(json_filepattern))
    assert len(mtch) == 1, "Can't figure out the json filepattern: %s" % sess_name
    json_filepattern = json_filepattern.replace(mtch[0], "%0"+str(len(mtch[0]))+"d")

    return json_filepattern, n_frames


def is_pstv_pnt_smpl(start_fr_idx, end_fr_idx, vid_meta, opts):
    """
    This function decides whether a particular sample is a positive (pointing)
    or not. This function work for both single frames and sliding windows.
    A sliding window (multiple frames) is considered a positive sample if the
    fraction of frames which are positive/pointing is above the threshold
    given in opts.gtoverlap

    @param  start_fr_idx    This is the starting frame number for the sample
    @param  end_fr_idx      This is the ending frame number for the sample +1.
                            So if the first two arguments are 4, and 7, then
                            the sample is over the frames 4,5,6.
    @param  vid_meta        Is the dictionary containing meta data for this
                            video session. It is used to find out whether
                            a certain frame is pointing or not.

    @returns is_pstv        {0,1}. It returns 0 when this is a negative sample,
                            and 1 when it's a positive.
    """
    cnt_pointing = 0
    cnt_tap = 0
    cnt_pat = 0
    cnt_reach=0
    cnt_push= 0
    # count how many frames are pointing in this sample
    for frame_idx in range(start_fr_idx, end_fr_idx):
        # find out if this is a pointing frame
        frame_id = False
        sess_frame_idx = vid_meta["start_keyframe_num"] + frame_idx
        for stf, enf in vid_meta["pointing_frames"]:
            if stf <= sess_frame_idx and sess_frame_idx <= enf:
                cnt_pointing += 1
                frame_id = True
                break
        for stf, enf in vid_meta["tap_gesture"]:
            if stf <= sess_frame_idx and sess_frame_idx <= enf:
                cnt_tap += 1
                frame_id = True
                break
        for stf, enf in vid_meta["pat_table"]:
            if stf <= sess_frame_idx and sess_frame_idx <= enf:
                cnt_pat += 1
                frame_id = True
                break
        for stf, enf in vid_meta["reach_gesture"]:
            if stf <= sess_frame_idx and sess_frame_idx <= enf:
                cnt_reach += 1
                frame_id = True
                break
        for stf, enf in vid_meta["push_away"]:
            if stf <= sess_frame_idx and sess_frame_idx <= enf:
                cnt_push += 1
                break
    # get the fraction of frames which intersect with pointing
    cnts = [cnt_pointing,cnt_tap,cnt_pat,cnt_reach,cnt_push]
    is_pstv = float(max(cnts)) / (end_fr_idx-start_fr_idx)
    is_pstv =  np.argmax(cnts)+1 if is_pstv >= opts.gtoverlap else 0
    return is_pstv


def get_smpl_pose_data(start_fr_idx, end_fr_idx, json_filepattern, arms_only, vid_meta, opts):
    """
    This function generates the pose data tensor. This is done in accordance
    to whether this is an arms-only classifier or a full-body pose based
    classifier. It collects the pose information for all the frames in this
    sample.

    @param  start_fr_idx    This is the starting frame number for the sample
    @param  end_fr_idx      This is the ending frame number for the sample +1.
                            So if the first two arguments are 4, and 7, then
                            the sample is over the frames 4,5,6.
    @param  json_filepattern  This is the json filepattern which is used to
                            read the OpenPose data files. This is set
                            according to what session this sample belongs to.
    @param  arms_only       A boolean indicating whether this is an arms
                            only classifier.
    @param  vid_meta        Is the dictionary containing meta data for the
                            video session for this sample. It is used to find
                            out which pose data belongs to the child.

    @retuns smpl_pose_data  This is a np tensor of size
                            (#frames in sample, #features, 3).
                            The #frames in this sample are decided by the
                            starting and ending frame index given. The # of
                            features depends on whether you wanted an arm
                            based classifier or not. If full body, the
                            second dimension is organized as follows:
                                (0:17)  - data for body pose
                                (18:38) - data for left hand
                                (39:59) - data for right hand
                            If arms_only is true, the 2nd dim:
                                (0:5)   - data for arms pose
                                (6:26)  - data for left hand
                                (27:47) - data for right hand
                            The third dimension gives the x, y coordinates and
                            the confidence value given by OpenPose for each
                            pose point.

                            NOTE: this tensor would be set to 0s for all the
                            frames where the child pose was unavailable! See
                            below.
    """
    # get the number of frames in this sample
    n_frames = end_fr_idx - start_fr_idx
    # get # of pose points (according to whether this is only arms)
    if arms_only:
        n_pose_pnts = 2*n_hand_pnts + len(body_arm_kp_idxs)
    else:
        n_pose_pnts = 2*n_hand_pnts + n_body_pnts
    # create an array which will hold all the data. Note that for the frames
    # where there's no child data, the data would contain zeros. If you want
    # to do something else, you can change it in the loop below
    smpl_pose_data = np.zeros((n_frames, n_pose_pnts, 3))

    # get pose data for all the frames in this sample
    for frame_idx in range(start_fr_idx, end_fr_idx):
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

        # this is the frame index relative to this sample
        fi = frame_idx - start_fr_idx

        # only collect pose data if there is a valid child pose
        if child_idx != -1:
            # get the child pose data
            child_data = people_data[child_idx]

            # get the body pose keypoints
            pose_data = np.array(child_data[u"pose_keypoints"])
            pose_data = np.reshape(pose_data, (-1, 3))
            assert pose_data.shape[0] == n_body_pnts, \
                "Pose data is not %d pnts: %s" % (n_body_pnts, curr_json_fp)
            # filter the arm pose data
            if arms_only:
                pose_data = pose_data[body_arm_kp_idxs,:]
            # add body pose data to the array
            smpl_pose_data[fi,:pose_data.shape[0],:] = pose_data
            offest_i = pose_data.shape[0]

            # get the left hand keypoints
            lh_data = child_data[u"hand_left_keypoints"]
            lh_data = np.reshape(lh_data, (-1, 3))
            assert lh_data.shape[0] == n_hand_pnts, \
                "Left hand data is not %d pnts: %s" % (n_hand_pnts, curr_json_fp)
            # add left hand pose data to the array
            smpl_pose_data[fi,offest_i:offest_i+lh_data.shape[0],:] = lh_data
            offest_i += lh_data.shape[0]

            # get the right hand keypoints
            rh_data = child_data[u"hand_right_keypoints"]
            rh_data = np.reshape(rh_data, (-1, 3))
            assert rh_data.shape[0] == n_hand_pnts, \
                "Right hand data is not %d pnts: %s" % (n_hand_pnts, curr_json_fp)
            # add left hand pose data to the array
            smpl_pose_data[fi, offest_i:offest_i+rh_data.shape[0],:] = rh_data

    return smpl_pose_data


def get_n_feats(frames_per_smpl, arms_only):
    ############################################################################
    # SET THE LENGTH OF YOUR FEATURE VECTOR HERE. Currently I have set it to
    # the number of features/frame (incl openpose confidence) x number of
    # frames. You can change this to whatever according to what you want to
    # feed to your learning algo. Once you change this, you also need to
    # change how each sample is collected.
    ############################################################################
    nfeat = 48 if arms_only else 60
    nfeat *= frames_per_smpl * 2
    return nfeat


def get_sample_feat(smpl_pose_data, frames_per_smpl, arms_only):
    ############################################################################
    # this is where you can customize your features. [smpl_pose_data] is all
    # OpenPose data for all the frames in this sample (as retuned by
    # get_smpl_pose_data()). For single-frame classifier, [smpl_pose_data] would
    # have data for a single frame, and for sliding window classifiers, it would
    # have pose data for all the frames. If this was an arms_only classifier, 
    # [smpl_pose_data] would contain pose data only for the arms and the hands.
    #
    # Currently the code just takes the pose for all the frames, and uses the
    # raw values (both x,y coordinates and confidence) as features for the
    # classifier you will train. The structure of [get_smpl_pose_data] is
    # detailed in get_smpl_pose_data(). You can change the features to however
    # you think would be better for detecting the pointing task.
    ############################################################################
    # assert smpl_pose_data.size == get_n_feats(frames_per_smpl, arms_only), \
    #         "Inconsistent feature size from pose data"

    # smpl_pose_data = smpl_pose_data.flatten()
    # print smpl_pose_data
    # scalar = StandardScaler()
    # scalar.fit(smpl_pose_data)
    # smpl_pose_data = scalar.transform(smpl_pose_data)
    # print "done"
    return (smpl_pose_data[:,:,0:2]).flatten(),np.sum(smpl_pose_data[:,:,2])


def test_sample(feat_X, trained_model):
    ############################################################################
    # this is where you can use your model to test a sample. [feat_X] is the 
    # feature you generate in get_sample_feat(). Given the feature, the goal
    # of this function is to predict the probability that this sample is a
    # positive (pointing). The return value should be between [0,1]. A 1
    # would mean that the model is fairly certain that this is a positive
    # sample. A 0 would mean that the model is sure that this is not a pointing
    # sample.
    ############################################################################

    # scalar = StandardScaler()
    # scalar.fit(feat_X)
    # feat_X = scalar.transform(feat_X)
    #print trained_model.predict_proba([feat_X])
    return trained_model.predict_proba([feat_X])[-1][1]


def get_all_data(sess_names, frames_per_smpl, tmprl_stride, arms_only, opts):
    """
    This function gets all the data for training or testing from all the
    sessions specified in [sess_names]. It uses get_sample_feat() to get
    features for each sample. It also uses is_pstv_pnt_smpl() to find whether a
    particular sample is positive or negative.

    @param sess_names       A list of session names for which we want data.
    @param frames_per_smpl  Number of frames per sample. This would be 1 for
                            project part 1/2. Would be greater than for sliding
                            window classifier.
    @param tmprl_stride     Dictates what is the temporal stride used to sample
                            data from the video session. This is explained in
                            the project document.
    @param arms_only        This indicates whether the classifier being trained
                            is just arms based (part 1/3) or full body based
                            (part 2/4).

    @returns data_X         Is a matrix of size [#samples x #features]. This
                            would contain all the feature data that can be used
                            in train() to train a classifier, OR in test()
                            to test a classifier.
    @returns data_Y         Is a vector of size [#samples], where each location
                            corresponds to [data_X] and gives the ground-truth
                            value. Hence, all values in [data_Y] are {0,1}.
    """
    nfeat = get_n_feats(frames_per_smpl, arms_only)

    # this would store the features for all samples
    data_X = np.empty((0,nfeat))
    # this would store labels for all samples
    data_Y = np.empty((0))
    # this would store confidence weight for all samples
    data_C = np.empty((0))

    # collect features/labels for all samples in all sessions
    for sess_name in tqdm(sess_names, desc="                Collecting all data"):
        # get meta data for this video
        assert sess_name in sess_vids_meta, \
            "Can't find meta information for video " + sess_name
        vid_meta = sess_vids_meta[sess_name]

        # get number of frames
        n_frames = vid_meta["out_num_frames"]

        json_filepattern, nf = get_json_filepattern(sess_name, opts)
        # check if we have all the json files
        assert nf == n_frames, \
            "Can't find all open pose json files for session: " + sess_name

        # this is the frame index which always points to the starting frame
        # number for a sample
        start_fr_idx = 0
        # keeps a count of the number of samples collected
        smpl_i = 0

        # compute the expected number of samples from this session, so that
        # we can pre-allocate an array for features and labels
        expct_n_smpls = \
            int(math.floor((n_frames - frames_per_smpl) / tmprl_stride) + 1)
        sess_feat_data = np.empty((expct_n_smpls, nfeat))
        confidence_weight = np.empty((expct_n_smpls))
        sess_lbl_data = np.empty((expct_n_smpls))

        # iterate over all sliding windows (or single frames) in this video
        # session in order to collect all feature samples and labels
        while start_fr_idx + frames_per_smpl <= n_frames:
            # this is ending frame of sample +1
            end_fr_idx = start_fr_idx + frames_per_smpl

            # get pose data for all the frames in this sliding window
            # Note that this would be single frame data if doing part 1/2
            smpl_pose_data = \
                get_smpl_pose_data(start_fr_idx, end_fr_idx, json_filepattern, \
                                   arms_only, vid_meta, opts)

            assert smpl_i < expct_n_smpls, \
                "Somehow expected number of sample calculation wrong"

            # get the label for current sliding window sample (or single frame)
            sess_lbl_data[smpl_i] = \
                is_pstv_pnt_smpl(start_fr_idx, end_fr_idx, vid_meta, opts)
            # get the features for current sliding window sample (or single frame)
            # This function needs to be changed if you want to use different feats
            sess_feat_data[smpl_i], confidence_weight[smpl_i] = \
                get_sample_feat(smpl_pose_data, frames_per_smpl, arms_only)

            smpl_i += 1
            start_fr_idx += tmprl_stride

        assert smpl_i == expct_n_smpls, \
                "Somehow expected number of sample calculation wrong"

        # collect feature data from all the sessions
        data_X = np.concatenate((data_X, sess_feat_data), axis=0)
        # collect label data from all the sessions
        data_Y = np.concatenate((data_Y, sess_lbl_data), axis=0)
        # collect confidence weight data from all the sessions
        data_C = np.concatenate((data_C, confidence_weight), axis=0)
    #print data_X
    #print data_X.shape
    return data_X, data_Y, data_C


def train(sess_names, frames_per_smpl, testing_stride, arms_only, sess_vids_meta, opts):
    """
    This function is supposed to train a model on a set of sessions
    @param sess_names       A list of session names on which you would train
                            your model.
    @param frames_per_smpl  Number of frames used for the sliding window
                            classifier you are building. Note, this would be 1
                            for part 1/2.
    @param testing_stride   Dictates what is the temporal stride used to sample
                            data from testing sessions. This is explained in
                            the project document. You are free to use a different
                            sampling stride for training.
    @param arms_only        Indicates whether this is an arms only classifier
    @param sess_vids_meta   This is used to find what are the pointing frames
                            in each session; and what pose belongs to the child.

    @returns trained_model  Is the dictionary which should contain your trained
                            model. This would be passed to test_sample() so
                            you can used your trained model to infer whether
                            a sample is a pointing or not given its features.
    """

    # This is the #frames gap (stride) between each subsequent training sample
    # (this is explained in the project document). You are free to choose
    # whatever value that suits your training. Currently I have set it
    # to the same temporal stride that would be used for testing.
    training_stride = testing_stride

    logger.info("\tTraining on %d sessions, with %d frame(s) per sample", \
                len(sess_names), frames_per_smpl)

    # get both training features, and labels
    train_X, train_Y, train_C = \
        get_all_data(sess_names, frames_per_smpl, training_stride, arms_only, opts)

    logger.info("\tCollected %d samples (%d feats), out of which %d are positively labeled", \
                train_X.shape[0], train_X.shape[1], np.sum(train_Y))

    ############################################################################
    # now you have all the data in train_X, and the corresponding label vector
    # is in train_Y, you can use these to train a model. Store all the trained
    # model information in [trained_model]
    ############################################################################

    # you can put anything in this dictionary. It contains your trained model.
    # This dictionary would be passed to the test() function. So you can use
    # this to call any testing functions on your model via this dictionary.
    # trained_model = RandomForestClassifier(n_estimators = 50, oob_score = True)#MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (50, 50), random_state = 1)
    trained_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=50,learning_rate=1.5,algorithm="SAMME")
    trained_model.fit(train_X,train_Y,sample_weight=train_C)
    return trained_model


def test(trained_model, sess_names, frames_per_smpl, testing_stride, arms_only, sess_vids_meta, opts):
    """
    This function tests the trained model on the specified sessions
    @param trained_model    The trained model returned by train(). You need
                            to change this in the train() function so you can
                            use it here to test the model.
    @param sess_names       A list of session names on which your trained model
                            would be tested on.
    @param frames_per_smpl  Number of frames used for this sliding window
                            classifier. Note, this would be 1 for part 1/2.
    @param testing_stride   Dictates what is the temporal stride used to sample
                            data from testing sessions. This is explained in
                            the project document.
    @param arms_only        Indicates whether this was an arms only classifier
    @param sess_vids_meta   This is used to find what are the pointing frames
                            in each session; and what pose belongs to the child.

    @returns test_Y         Is a vector of size [#samples], where each location
                            gives the ground-truth value for a sample. Hence,
                            all values in [test_Y] are {0,1}.
    @returns pred_prob      Is a vector of size [#samples], where each location
                            should give the prediction probability of each
                            sample when you feed to your classifier. Hence,
                            all values in [pred_prob] are in the range [0,1].
    """
    logger.info("\tTesting on %d sessions, with %d frame(s) per sample", \
                len(sess_names), frames_per_smpl)

    # get both testing features, and labels
    test_X, test_Y, train_C = \
        get_all_data(sess_names, frames_per_smpl, testing_stride, arms_only, opts)

    logger.info("\tCollected %d samples, out of which %d are positively labeled", \
                test_X.shape[0], np.sum(test_Y))

    # this would store all the prediction probabilities
    pred_prob = np.empty((test_Y.size))

    # get predictions from the trained model for each sample
    for si in range(test_Y.size):
        # you need to change this function so that given features for a single
        # sample, you can use your trained model to make a prediction.
        # the prediction is in term of probability of this being a positive
        # sample i.e. "pointing"
        pred_prob[si] = test_sample(test_X[si], trained_model)

    return test_Y, pred_prob


def cv_train_test(exp_name, frames_per_smpl, tmprl_stride, arms_only, sess_cv_grps, sess_vids_meta, opts):
    """
    Does K-fold cross-validation given all the sessions. Calls train() / test()
    to train, test for each cross-validation fold. At the end, it takes all the
    results from all cross-validation runs and computes average precision, and
    plots the precision-recall graph.
    """
    logger.info("> Running train/test with %d frame(s)/smpl, temporal stride of %d, %s", \
                frames_per_smpl, tmprl_stride, "arms-only" if arms_only else "full-body")

    # this would store all the labels across K-fold cross validation
    all_test_Y = np.empty((0))
    # this would store all the predictions across K-fold cross validation
    all_pred_prob = np.empty((0))

    # do K-fold cross-validation
    for k in range(len(sess_cv_grps)):
        # get the training and validation split for this cross-validation group
        train_grps = range(len(sess_cv_grps))
        train_grps.remove(k)
        val_grps = [k]
        train_sess = [sess_name for gi in train_grps for sess_name in sess_cv_grps[gi]]
        val_sess = [sess_name for gi in val_grps for sess_name in sess_cv_grps[gi]]
        # check whether there's no overlap between train / val
        assert len([1 for sess_name in train_sess if sess_name in val_sess]) == 0, \
            "There's somehow overlap of sessions between train/val"
        # check whether all sessions are covered
        assert len(set(train_sess)) == len(train_sess), "Some session duplicated in train"
        assert len(set(val_sess)) == len(val_sess), "Some session duplicated in val"
        assert len(all_sessions) == len(train_sess)+len(val_sess), \
            "Some session missing from the union of train/val sessions"

        # Train for this K-fold cross-val split
        model = train(train_sess, frames_per_smpl, tmprl_stride, arms_only, \
                      sess_vids_meta, opts)

        # Test the model trained above for this K-fold cross-val split
        test_Y, pred_prob = test(model, val_sess, frames_per_smpl, tmprl_stride, \
                                 arms_only, sess_vids_meta, opts)

        # collect feature data from all the sessions
        all_test_Y = np.concatenate((all_test_Y, test_Y), axis=0)
        # collect label data from all the sessions
        all_pred_prob = np.concatenate((all_pred_prob, pred_prob), axis=0)

    all_test_Y[all_test_Y != 1] = 0

    # compute the average precision metric over all K-fold cross validation runs
    ap = skmetrics.average_precision_score(all_test_Y, all_pred_prob)

    ap_desc_str = "{0} ({1},{2}): AP score: {3:0.3f}" \
                     .format(exp_name, frames_per_smpl, tmprl_stride, ap)
    logger.info("\t" + ap_desc_str)

    # compute the predicion recall at all confidence threshold points
    prec, rcll, thrsh = \
        skmetrics.precision_recall_curve(all_test_Y, all_pred_prob, pos_label=1)

    pickle.dump([rcll, prec, ap_desc_str], open("data5_new.pkl", "ab"))

    # plot precision recall curve
    # plt.step(rcll, prec, color='b', alpha=0.2, where='post')
    # plt.fill_between(rcll, prec, step='post', alpha=0.2, color='b')
    # plt.xlabel('Recall'); plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05]); plt.xlim([0.0, 1.0])
    # plt.title(ap_desc_str)
    # plt.ion()
    # plt.show()


if __name__ == "__main__":
    # create the argument parser
    parser = argparse.ArgumentParser(description=
"------------------------- Training/Testing for Rapid-ABC ---------------------"
"------------------------------------------------------------------------------"
"This script is starter code for reading/analyzing the openpose json "
"results.",)

    parser.add_argument("-p", "--part", type=int, metavar="PROJECTPART",
                        required=True, choices=xrange(1, 5),
                        help="What project part do you want to train/test for")
    parser.add_argument("-k", "--crossvalk", type=int, metavar="CROSSVALK",
                        default=3,
                        help="Number of cross-val groups - DON'T CHANGE.")
    parser.add_argument("-o", "--gtoverlap", type=float, metavar="GTOVERLAP",
                        default=0.5,
                        help="The overlap threshold between the temporal "
                             "window and the pointing frames to decide whether "
                             "a window is +ve or -ve - DON'T CHANGE.")
    parser.add_argument("-s", "--crossvalstats", action="store_true",
                        help="Print stats about cross-validation groups.")
    parser.add_argument("-d", "--openposedir", type=str, metavar="OPENPOSEDIR",
                        default=default_op_dir,
                        help="This is the directory where openpose is")

    # parse all the program arguments
    args = parser.parse_args()

    try:
        # number of cross validation groups
        K = args.crossvalk
        sess_cv_grps = []
        # split the sessions into K groups
        for g in range(K):
            start_i = int(round(g * float(len(all_sessions))/K))
            end_i = int(round((g+1) * float(len(all_sessions))/K))
            if g == K-1:
                end_i = len(all_sessions)
            sess_cv_grps.append(tuple(all_sessions[start_i:end_i]))

        # test whether the cross-val grouping was done correctly
        tmp_all_sess = [sess for grp in sess_cv_grps for sess in grp]
        assert len(tmp_all_sess) == len(set(tmp_all_sess)), \
            "We got repeats while creating session groups"
        assert len(tmp_all_sess) == len(all_sessions), \
            "All sessions were not present in the grouping"

        # load the this session's meta information
        assert os.path.exists(sess_meta_fp), \
            "Session meta file missing %s" % sess_meta_fp
        sess_vids_meta = np.load(sess_meta_fp)
        sess_vids_meta = sess_vids_meta.item()

        # get cross-validation statistics
        if args.crossvalstats:
            get_cv_grps_stats(sess_cv_grps, sess_vids_meta)

        # project part infos -   name                        , #frames/smpl, strides,  arm-only
        proj_parts_infos = [("Single-frame Arm-based Detector",    (1,),      (1,),     True), \
                            ("Single-frame Body-based Detector",   (1,),      (1,),     False), \
                            ("Sliding window Arm-based Detector",  (5,10,20), (3,5,10), True), \
                            ("Sliding window Body-based Detector", (5,10,20), (3,5,10), False)]
        proj_parts_info = proj_parts_infos[args.part-1]
        logger.info("Training/Testing (%d-fold cross-val) for Part %d: %s", \
                    args.crossvalk, args.part, proj_parts_info[0])

        frms_per_smpl = proj_parts_info[1]
        # run train/tests over all setting for frames/smpl / strides
        for i in range(len(frms_per_smpl)):
            cv_train_test(proj_parts_info[0], proj_parts_info[1][i], \
                          proj_parts_info[2][i], proj_parts_info[3], \
                          sess_cv_grps, sess_vids_meta, args)

    except Exception as _err:
        logger.error(_err)
        traceback.print_exc(file=sys.stderr)
        sys.exit(-2)
