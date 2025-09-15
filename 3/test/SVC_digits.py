import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression, Perceptron
# from sys import path
# from os import walk
# from PIL.Image import open, Image
# from PIL.ImageOps import expand, autocontrast
# import time

digits = load_digits()

model = SVC(gamma=0.001)
# model = LogisticRegression()
# model = Perceptron()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.4, shuffle=False)

model.fit(X_train, y_train)


# Попытка классифицировать свои изображения цифр
# path_data = Path(path[0] + "\\28")
# test_data_paths, test_imgs = [], []
# for root, dirs, name in walk(path_data):
#     for n in name:
#         test_data_paths.append(root + "\\" + n)
# border = (1, 1, 1, 1) 
# test_data = [open(path).convert('L').resize((8, 8)) for path in test_data_paths]
# # test_data = [expand(open(path).convert('L').resize((6, 6)), border=border) for path in test_data_paths]
# test_data = [autocontrast(img) for img in test_data]

# for img in test_data:
#     test_imgs.append(np.array(img).flatten())
# test_imgs = np.array(test_imgs)
# y_test = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 4, 8]



predicted = model.predict(X_test)

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")
plt.show()

print(
    f"Classification report for classifier {model}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()