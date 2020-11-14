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


def train_epoch(model, loss_fn, files, batch_indices,
                optimizer, device):
    model.train()
    avg_loss = 0.0

    for batch_i in batch_indices:
        batch_data = np.load(files[batch_i])
        batch_xs = torch.Tensor(batch_data['data']).to(device)

        batch_size, n_frames = batch_xs.shape[0:2]
        # accident: [0, 1]  -->  class = 1
        # no accident: [1, 0]  --> class = 0
        # model output = [1-p(accident), p(accident)]
        batch_ys = torch.Tensor(batch_data['labels'][:, 1]).long().to(device)
        batch_ys = batch_ys.unsqueeze(0)
        # (N x B x 1)
        batch_ys = batch_ys.repeat(n_frames, 1, 1)

        optimizer.zero_grad()   # .backward() accumulates gradients

        alphas, predictions = model(batch_xs)
        # weights = exp(-(N-1, N-2, ...0)) = exp(1-N), exp(2-N), ... 1
        # so last frame has the most weight since data is set up so last frame
        # has accident
        loss = loss_fn(predictions, batch_ys)
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

            batch_size, n_frames = batch_xs.shape[0:2]
            batch_ys = torch.Tensor(
                batch_data['labels'][:, 1]).long().to(device)
            batch_ys = batch_ys.unsqueeze(0)
            # (N x B x 1)
            batch_ys = batch_ys.repeat(n_frames, 1, 1)

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
        # np.random.shuffle(batch_indices)

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
        print("Epoch: %d, Train Loss: %.3f, Val loss: %.3f, Elapsed Time: %.2fs" % (
            epoch, float(train_loss), float(val_loss), tend - tstart),
            flush=True)
