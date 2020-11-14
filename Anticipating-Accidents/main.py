import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import sys
import yaml

import torch
import torch.optim

from accident_detection import AccidentDetection, AccidentLoss
import model_utils
import utils

if __name__ == "__main__":
    args = utils.parse_arguments()
    # args = Arguments()
    with open("params.yaml") as f:
        params = yaml.load(f)

    timestamp = int(time.time())
    progress_dir = os.path.join(
        params["root"], params["save_path"], "%d" % timestamp)
    if not os.path.exists(progress_dir):
        os.mkdir(progress_dir)
    print(args, flush=True)
    if args.print_to_file:
        sys.stdout = open(os.path.join(progress_dir, "output.txt"), "w")
    print("Params: %s" % params, flush=True)
    print("Args: %s" % args, flush=True)

    train_dir = os.path.join(params["root"], params["train_path"])
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    eval_dir = os.path.join(params["root"], params["test_path"])
    eval_files = [os.path.join(eval_dir, f) for f in os.listdir(eval_dir)]
    video_dir = os.path.join(params["root"], params["video_path"])
    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir)]

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    num_workers = 8 if cuda else 0
    if cuda:
        print("CUDA GPU!")
    else:
        print("CPU!")

    # define model, optimizer, lr scheduler
    model = AccidentDetection(
        img_dim=params["img_dim"], n_hidden_layers=params["n_hidden_layers"],
        img_feat_dim=params["img_feat_dim"],
        obj_feat_dim=params["obj_feat_dim"],
        lstm_hidden_dim=params["hidden_feat_dim"],
        lstm_dropout=params["lstm_dropout"],
        device=device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    n_frames = params["n_frames"]
    loss_fn = AccidentLoss(n_frames, device)

    # optionally load from some previous checkpoint
    if args.model_path != "":
        print("Loading Existing Model: %s" % args.model_path)
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        model.apply(model_utils.init_weights)

    if args.train:
        model_utils.train_model(model, optimizer, scheduler, loss_fn,
                                progress_dir, train_files[:15], eval_files,
                                args.num_epochs, device)
    else:
        # run demo
        pass
