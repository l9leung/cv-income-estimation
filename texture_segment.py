import os
import numpy as np
import cv2
from lm_filter import makeLMfilters
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from data.load_subset import load_subset

np.random.seed(123)

n_images = 100
images, incomes, geoids = load_subset(size=n_images)

frames = images["Los Angeles"]
incomes = incomes["Los Angeles"]

lm = makeLMfilters()

for i in range(frames.shape[0]):
    frame = frames[i, :, :].reshape((400, 600, 3))
    fig, axes = plt.subplots(2, 2)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    axes[0, 0].imshow(frame, cmap="gray")
    axes[0, 0].axis("off")
    axes[0, 0].set_title(f"${int(incomes[i])}")
    frame = cv2.GaussianBlur(frame, (7, 7), 0)

    filtered = np.zeros(shape=(400, 600, 48), dtype="uint8")
    for i in range(lm.shape[2]):
        filtered[:, :, i] = cv2.filter2D(frame, -1, lm[:, :, i])
        # plt.imshow(cv2.filter2D(frame, -1, lm[:, :, i]), cmap="gray")
        # plt.axis("off")
        # plt.show()

    bank = filtered.reshape(-1, filtered.shape[-1])

    for k, ax in zip(range(2, 5), axes.flatten()[1:]):
        kmeans = KMeans(n_clusters=k, random_state=123, verbose=1).fit(bank)
        ax.imshow(kmeans.predict(bank).reshape((400, 600)), cmap="gray")
        ax.axis("off")
        ax.set_title(f"$k={k}$")
    plt.show()
