import numpy as np
import matplotlib.pyplot as plt
import re


def parse_file(fname):
    with open(fname, "r") as f:
        txt = f.read()

    runtimes = re.findall("Epoch Time Cost: (\d+.\d+) s", txt)
    train_loss_vals = re.findall("Loss: (\d+.\d+)", txt)
    eval_loss_vals = re.findall("Eval Loss: (\d+.\d+)", txt)

    runtimes = [float(v) for v in runtimes]
    train_loss_vals = [float(v) for v in train_loss_vals]
    eval_loss_vals = [float(v) for v in eval_loss_vals]

    return runtimes, train_loss_vals, eval_loss_vals


runtimes, train_loss_vals, eval_loss_vals = parse_file("baseline_loss.txt")
runtimes2, train_loss_vals2, eval_loss_vals2 = parse_file("new_arch_loss.txt")
time_axis = list(range(0, 19, 3))

plt.plot(time_axis, eval_loss_vals, label="Baseline Eval Loss")
plt.plot(time_axis, eval_loss_vals2, label="New Architecture Eval Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Accident Prediction: Eval Loss vs Epoch")
plt.legend()
plt.show()
