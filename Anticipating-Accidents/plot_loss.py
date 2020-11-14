import numpy as np
import matplotlib.pyplot as plt
import re


with open("loss.txt", "r") as f:
    txt = f.read()

matches = re.findall(
    "Epoch: (\d+), Train Loss: (\d+.\d+), Val loss: (\d+.\d+),.*", txt)

epochs = []
train_loss_series = []
val_loss_series = []

for (epoch, train_loss, val_loss) in matches:
    epochs.append(int(epoch))
    train_loss_series.append(float(train_loss))
    val_loss_series.append(float(val_loss))

plt.plot(epochs, train_loss_series, label="train loss")
plt.plot(epochs, val_loss_series, label="val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Accident Prediction: Loss vs Epoch")
plt.legend()
plt.show()
