import os
import pickle
import time
import random
import numpy as np
import cv2
from lm_filter import makeLMfilters
# from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

random.seed(123)


def load_subset_gray(n_images=5000, seed=123, frames_path="D:/frames.dat"):
    # Choose n random images to use
    filenames = os.listdir("data/street_view/Los Angeles/images")
    random.shuffle(filenames)
    filenames = filenames[:n_images]
    frames = np.memmap(frames_path, dtype="uint8", mode="w+",
                       shape=(n_images, 200, 300))
    # Write grayscale images to disk
    for i, filename in enumerate(filenames):
        frame = cv2.imread(f"data/street_view/Los Angeles/images/{filename}",
                           cv2.IMREAD_GRAYSCALE)
        frame = cv2.resize(frame, (300, 200))
        frames[i, :, :] = frame
    frames.flush()


def build_bank(lm, frames_path="D:/frames.dat"):
    np.random.seed(123)
    # Map to images on disk
    frames = np.memmap(frames_path, dtype="uint8", mode="r+",
                       shape=(n_images, 200, 300))
    # Create filter bank array on disk
    bank = np.memmap("D:/bank.dat", dtype="uint8", mode="w+",
                     shape=(n_images, 200, 300, 48))
    for n in range(frames.shape[0]):
        print(n + 1)
        # Gaussian blur
        frames[n, :, :] = cv2.GaussianBlur(frames[n, :, :], (7, 7), 0)
        for x in range(lm.shape[2]):
            # plt.imshow(lm[:, :, x], cmap="gray")
            # plt.axis("off")
            # plt.title(f"LM Filter {x}")
            # plt.show()
            filtered = cv2.filter2D(frames[n, :, :], -1, lm[:, :, x])
            # plt.imshow(filtered, cmap="gray")
            # plt.axis("off")
            # plt.title(f"LM Filter {x}")
            # plt.show()
            np.random.shuffle(filtered)
            bank[n, :, :, x] = filtered  # Filters
    bank = bank.reshape(-1, bank.shape[-1])
    np.random.shuffle(bank)
    bank.flush()


def build_dict(batch_size, n_images=5000, k=512, bankpath="D:/bank.dat",
               dictpath="D:/texton_dict.dat"):
    bank = np.memmap(bankpath, dtype="uint8", mode="r+",
                     shape=(n_images*200*300, 48))
    if batch_size is None:
        kmeans = KMeans(n_clusters=k, init="k-means++", verbose=1,
                        copy_x=False, random_state=123)
        kmeans.fit(bank)

    else:
        kmeans = MiniBatchKMeans(n_clusters=k, init="k-means++", verbose=1,
                                 compute_labels=False, random_state=123,
                                 batch_size=batch_size)
        n_batches = int(bank.shape[0] / batch_size)
        # indicies = random.sample(range(bank.shape[0]), bank.shape[0])
        for t in range(n_batches):
            print(f"BATCH {t+1} OF {n_batches}")
            # kmeans.fit(bank[indicies[batch_size*t:batch_size*(t+1)], :])
            kmeans.fit(bank[batch_size*t:batch_size*(t+1), :])
    # texton_dict = np.memmap(dictpath, dtype="float32", mode="w+",
    #                         shape=(k, 48))
    # texton_dict[:, :] = kmeans.cluster_centers_
    return kmeans


def build_hist(filename, lm, kmeans):
    start = time.time()
    # Read image
    frame = cv2.imread(f"data/street_view/Los Angeles/images/{filename}",
                       cv2.IMREAD_GRAYSCALE)
    frame = cv2.resize(frame, (300, 200))
    assert frame is not None
    # Apply Gaussian blur
    frame = cv2.GaussianBlur(frame, (7, 7), 0)
    # Array to store filter responses
    responses = np.zeros(shape=(200, 300, 48), dtype="uint8")
    for x in range(lm.shape[2]):
        responses[:, :, x] = cv2.filter2D(frame, 1, lm[:, :, x])
    # Flatten
    responses = responses.reshape(-1, responses.shape[-1])
    # Array to store histogram
    texton_hist = np.zeros(shape=(len(kmeans.cluster_centers_)))
    for i in range(responses.shape[0]):
        texton_hist[kmeans.predict(responses[i].reshape(1, -1))] += 1
    end = time.time()
    print(end - start)
    return texton_hist


if __name__ == "__main__":
    n_images = 1000
    batch_size = 500000
    k = 100
    load_subset_gray(n_images)  # Make greyscale images on disk
    lm = makeLMfilters()  # Create Leung Malik filters
    build_bank(lm)  # Apply filters to get bank
    kmeans = build_dict(batch_size, k)  # Cluster textons to get dict
    with open("kmeans.pkl", "wb") as filename:
        pickle.dump(kmeans, filename)
