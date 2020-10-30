import numpy as np
import cv2
import matplotlib.pyplot as plt


demo_path = './dataset/features/testing/'
file_name = '%03d' % 1
all_data = np.load(demo_path + 'batch_' + file_name + '.npz')
data = all_data['data']

x = np.reshape(data[0, 0, 2, :], [64, 64])
y = np.reshape(data[0, 0, 1, :], [64, 64])
print(np.allclose(x, y))
# for i in range(20):
# x = np.reshape(data[0, 0, 2, :], [64, 64])
# x = x / np.max(x)
# x = x.astype(float)
# # x = (x * 255).astype(int)
# plt.imshow(x, cmap="gray")
# plt.show()
