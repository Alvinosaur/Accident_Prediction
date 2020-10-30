import numpy as np
import os
import time

import torch
import torch.nn as nn


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def onehot_to_binary(batch_ys: np.ndarray):
    # (2 x 1)
    has_accident = np.array([[0], [1]])
    # (B x 2)(2 x 1) = (B x 1)
    return batch_ys @ has_accident


def train_epoch(model, loss_fn, files, batch_indices,
                optimizer, device):
    model.train()
    avg_loss = 0.0

    for batch_i in batch_indices:
        batch_data = np.load(files[batch_i])
        batch_xs = torch.Tensor(batch_data['data']).to(device)

        n_frames = batch_xs.shape[1]
        batch_ys = onehot_to_binary(batch_data['labels'])
        batch_ys = np.tile(batch_ys, (1, n_frames))
        batch_ys = torch.Tensor(batch_ys).to(device)

        optimizer.zero_grad()   # .backward() accumulates gradients

        alphas, logits = model(batch_xs)
        loss = loss_fn(logits, batch_ys)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()

    avg_loss /= len(files)
    return avg_loss


def eval_model(model, loss_fn, files, device):
    with torch.no_grad():
        model.eval()
        avg_loss = 0.0

        for fname in files:
            batch_data = np.load(fname)
            batch_xs = torch.Tensor(batch_data['data']).to(device)
            batch_ys = torch.Tensor(batch_data['labels']).to(device)

            alphas, logits = model(batch_xs)
            loss = loss_fn(logits, batch_ys).detach()
            avg_loss += loss.item()

    avg_loss /= len(files)
    return avg_loss


def train_model(model, optimizer, scheduler, loss_fn, progress_dir,
                train_files, eval_files, num_epochs, device):
    # detect anomalies in calculating gradient
    torch.autograd.set_detect_anomaly(True)

    if not os.path.exists(progress_dir):
        os.mkdir(progress_dir)

    num_batches = len(train_files)
    batch_indices = np.arange(num_batches)

    for epoch in range(num_epochs):
        tstart = time.time()
        print("\n========== Epoch {} ==========".format(epoch))

        # shuffle train dataset
        np.random.shuffle(batch_indices)

        # Train
        train_loss = train_epoch(model, loss_fn, train_files, batch_indices,
                                 optimizer, device)

        # Evaluate on validation set
        val_loss = eval_model(model, loss_fn, eval_files, device)

        # decrease learning rate with scheduler
        scheduler.step(metrics=val_loss)

        # save model weights for this epoch
        unique_name = "epoch_%d.h5" % (epoch)
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss}
        torch.save(checkpoint, os.path.join(progress_dir, unique_name))

        tend = time.time()
        print("Epoch %d Elapsed Time: %.2fs" % (epoch, tend - tstart))
