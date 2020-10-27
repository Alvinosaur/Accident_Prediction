import cv2
import argparse
import numpy as np
import os
import pdb
import time
import matplotlib.pyplot as plt
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder

############### Global Parameters ###############
# path
train_path = './dataset/features/training/'
test_path = './dataset/features/testing/'
demo_path = './dataset/features/testing/'
default_model_path = './model/demo_model'
save_path = './model/'
video_path = './dataset/videos/testing/positive/'
# batch_number
train_num = 126
test_num = 46


############## Train Parameters #################

# Parameters
learning_rate = 0.0001
n_epochs = 30
batch_size = 10
display_step = 10

# Network Parameters
n_input = 4096  # fc6 or fc7(1*4096)
n_detection = 20  # number of object of each image (include image features)
n_hidden = 512  # hidden layer num of LSTM
n_img_hidden = 256  # embedding image features
n_att_hidden = 256  # embedding object features
n_classes = 2  # has accident or not
n_frames = 100  # number of frame in each video
##################################################


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='accident_LSTM')
    parser.add_argument('--mode', dest='mode',
                        help='train or test', default='demo')
    parser.add_argument('--model', dest='model', default=default_model_path)
    parser.add_argument('--gpu', dest='gpu', default='0')
    args = parser.parse_args()

    return args


"""
Possible architecture changes:
- process k frames at a time, not just 1 frame
- bidirectional 
- apply MLP's to features, not just affine transformation
- change to not have preset number of objects, handle any N objects
- play with LSTM params, or variants (ex: GRU)

Implementation Details:
- objects can be diff size, so how to transform them into features? pass
    entire image, but mask object so only its pixels are non-zero. this
    way, same weights of same size can be applied to each object
    - but in this case, weights depend on spatial location of an object,
        so this is how transforming object into a feature encodes its
        position, not just the object pixels, which are uninformative
- paper uses hyperparam for number of objects in image. What if there
    are less than num_objs in a given image? those extra remaining images
    are completely blacked out, and after applying linear transform, we
    just black out the entire feature so has no effect
    - but if some objects are not visible and their pixels are all
        blacked out, then why do we need to use the post-transformation
        mask to zero everything out?

TODO:
- look at what the input features X look like: just imshow them
- finish loss function
"""


class Fattn(nn.Module):
    def __init__(self, lstm_hidden_dim, obj_feat_dim):
        """Calculates alpha = softmax(f_attn(h_t-1, a_t)), which 
        assigns 

        Args:
            lstm_hidden_dim ([type]): [description]
            obj_feat_dim ([type]): [description]

        Returns:
            [type]: [description]
        """
        # linear transform of previous hidden state
        self.hidden_lin = nn.Linear(
            self.lstm_hidden_dim, self.obj_feat_dim, bias=False)
        self.joint_lin = nn.Linear(
            self.obj_feat_dim, 1, bias=False)

    def forward(self, a, hprev, mask):
        """[summary]

        Args:
            a (Tensor): (B x K-1 x obj_feat_dim)
            hprev (Tensor): (B x lstm_hidden_dim)
            mask (Tensor): (B x K-1 x 1)
        """
        # (B x K-1 x obj_feat_dim)
        e = torch.tanh(self.hidden_lin(hprev) + a)
        # (B x K-1 x obj_feat_dim) -> (B x K-1 x 1)
        alphas = self.joint_lin(e)
        # calculate probability/importance of each K-1 object
        # mask out any features that are non-existent
        alphas = torch.softmax(torch.multiply(alphas, mask), dim=-1)
        return alphas


class AccidentDetection(nn.Module):
    def __init__(self, img_dim, n_hidden_layers, img_feat_dim, obj_feat_dim, lstm_hidden_dim, lstm_dropout=0):
        """Main module encapsulating accident detection pipeline. Given a 
        video sequence of images, processes one frame at a time. 

        Args:
            img_dim (int): size of flattened image
            n_hidden_layers (int): number of hidden layers in LSTM
            img_feat_dim (int): size of processed image feature
            obj_feat_dim (int): size of processed object feature
        """
        self._img_dim = img_dim
        self._obj_dim = img_dim  # entire image masked out except for object
        self._n_hidden_layers = n_hidden_layers
        self._img_feat_dim = img_feat_dim
        self._obj_feat_dim = obj_feat_dim
        self._obj_feat_dim2 = obj_feat_dim  # after 2nd linear
        self._lstm_hidden_dim = lstm_hidden_dim

        self._img_to_feat = nn.Linear(self._img_dim, self._img_feat_dim)
        self._obj_to_feat = nn.Linear(self._obj_dim, self._obj_feat_dim)
        self._obj_to_feat2 = nn.Linear(self._obj_feat_dim, self._obj_feat_dim2)
        # probability of accident
        self._out_to_pred = nn.Linear(self._n_hidden_layers, 1)
        self._lstm = nn.LSTM(
            input_size=self._img_feat_dim + self._obj_feat_dim,
            hidden_size=self._lstm_hidden_dim,
            num_layers=self._n_hidden_layers,
            batch_first=True,
            dropout=lstm_dropout)
        self._fattn = Fattn(self._lstm_hidden_dim, self._obj_to_feat2)

    def forward(self, x):
        """Forward pass

        Args:
            x (Tensor): B x N x K x D
                B = Batch size
                N = num image frames per entry
                K = 1 + num objects to focus on
                D = input feature dimension (self._img_dim)
                D_i = image feature dim
                D_o = obj feature dim
                D_o2 = 2nd obj feature dim
        """
        B, N, K, D = x.shape
        # all zeros for an obj index in a frame of a specific batch if that obj
        # isn't present
        # no mask for first of K since that represents entire image, not an obj
        # (B x N x K-1 x 1)
        obj_mask = torch.sum(x[:, :, 1:], dim=-1, keepdim=True)

        # transform full image input vec into img feature
        # (B x N x 1 x D_i)
        img_feat = self._img_to_feat(x[:, :, 0, :])

        # transform each obj input vec into obj feature
        # (B x N x K-1 x D_o)
        obj_feat = self._obj_to_feat(x[:, :, 1:, :])

        # mask out any obj features where obj isn't present
        # (B x N x K-1, D_o) = (B x N x K-1 x D_o) * (B x N x K-1 x 1) < brdcst
        obj_feat = torch.multiply(obj_feat, obj_mask)
        # 2nd affine transform
        # (B x N x K-1, D_o) -> (B x N x K-1, D_o2)
        obj_feat = self.__obj_to_feat2(obj_feat)

        # intialize LSTM hidden state and
        hidden_state = torch.zeros((B, self._lstm_hidden_dim))
        prev_output = torch.zeros((B, self._lstm_hidden_dim))

        # track all info
        all_alphas = []
        all_predictions = []

        for fi in range(N):
            # (B)
            cur_obj_feat = obj_feat[:, fi, :, :]
            cur_img_feat = img_feat[:, fi, :]
            cur_obj_mask = obj_mask[:, fi, :]

            alphas = self._fattn(cur_obj_feat, prev_output, cur_obj_mask)
            # weighted each object feature by its attention alphas
            w_obj_feat = torch.multiply(cur_obj_feat, alphas)
            # sum up all K-1 features to produce weighted sum
            w_obj_feat = torch.sum(w_obj_feat, dim=2)

            fusion = torch.cat([cur_img_feat, w_obj_feat], dim=0)
            prev_output, hidden_state = self._lstm(fusion, hidden_state)
            logits = self._out_to_pred(prev_output)
            predictions = torch.softmax(logits, dim=1)

            # save all outputs
            all_alphas.append(alphas)
            all_predictions.append(predictions)

        return all_alphas, all_predictions


def calc_loss(predictions, labels):
    # TODO
    return
