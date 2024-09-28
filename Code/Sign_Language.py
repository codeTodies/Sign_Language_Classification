
import numpy as np
import pandas as pd
# data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from sklearn.preprocessing import LabelBinarizer

for dirname, _, filenames in os.walk('./../Dataset/Input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.

train_df = pd.read_csv("./../Dataset/Input/sign_mnist_train/sign_mnist_train.csv")
test_df = pd.read_csv("./../Dataset/Input/sign_mnist_test/sign_mnist_test.csv")
# 5 dòng dữ liệu đầu tiên
print(train_df.head(5))

y_train = train_df['label']
y_test = test_df['label']
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)
# Các nhãn (Label)
print('Classes: ' + str(label_binarizer.classes_))

x_train = train_df.drop('label', axis=1).values
x_test = test_df.drop('label', axis=1).values
# print(x_test.shape)
# Chuẩn hóa dữ liệu về 0-1 thay vì 0-255
x_train = x_train / 255
x_test = x_test / 255

size_train = x_train.shape[0]
size_test = x_test.shape[0]
# Reshape mảng 1 chiều thành mảng 28 * 28
x_train = x_train.reshape(size_train, 28, 28)
x_test = x_test.reshape(size_test, 28, 28)
# Plot 6 images
f, ax = plt.subplots(2, 3)
f.set_size_inches(6, 6)
k = 0
for i in range(2):
    for j in range(3):
        ax[i, j].imshow(x_train[k].reshape(28, 28), cmap='gray')
        k += 1
    plt.tight_layout()

plt.show()





