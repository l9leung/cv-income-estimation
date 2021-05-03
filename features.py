from pspnet_segment import segmenter
import numpy as np
import os
import pickle
import pandas as pd
from data.acs_load import acs_load
import cv2
from colors import lab_color_hists
from lm_filter import makeLMfilters
from textures import build_hist
import random
import matplotlib.pyplot as plt

# Income data
acs = acs_load(cities=["Los Angeles"])
acs = acs["Los Angeles"]
# DataFrame indexed by image names
la = pd.DataFrame(index=os.listdir("data/street_view/Los Angeles/images"))
# Geoid column
la["geoid"] = la.index.str.split("_").str[0]
# Merge with log median household income, preserve image file index
la = la.merge(acs[["geoid", "log_B19013001"]], how="left").set_axis(la.index)
# Shuffle
files = os.listdir("data/street_view/Los Angeles/images")
random.seed(123)
random.shuffle(files)
la = la.reindex(index=files)
# Drop missing
la.dropna(inplace=True)
print(len(la))

# Columns to store features
n_bins = 100
for segment in ["flat", "building", "nature", "vehicle"]:
    for color in ["L", "a", "b"]:
        for n_bin in range(n_bins):
            la[f"{segment}_{color}_{n_bin+1}"] = np.zeros(shape=(len(la)),
                                                          dtype="uint32")

# Segmenter instance (pspnet)
pspnet = segmenter()
for i, file in enumerate(la.index[:120]):
    frame = cv2.imread(f"data/street_view/Los Angeles/images/{file}")
    plt.imshow(np.flip(frame, axis=2))
    plt.axis("off")
    plt.show()
    # Get masks
    fname = f"""{file.split(".")[0]}_seg.png"""
    masks = pspnet.get_segment_masks(f"data/street_view/Los Angeles/segmented/{fname}")

    fig, axes = plt.subplots(2, 2)
    for segment, ax in zip(masks.keys(), axes.flatten()):
        masked = frame.copy()
        for channel in range(3):
            np.putmask(masked[:, :, channel], masks[segment], 255)
        ax.imshow(np.flip(masked, axis=2))
        ax.axis("off")
        ax.set_title(segment)
        masked = masked[(masked[:, :, 0] != 255) &
                        (masked[:, :, 1] != 255) &
                        (masked[:, :, 2] != 255)].reshape((-1, 1, 3))
        if masked.size == 0:
            continue

        # Color histograms
        hist = lab_color_hists(masked, n_bins)
        for _, color in enumerate(["L", "a", "b"]):
            la.loc[file, f"{segment}_{color}_1":f"{segment}_{color}_100"] = hist[_*n_bins:(_+1)*n_bins]

    plt.show()
    print("#" + str(i + 1))

# Subset of texton histograms
with open("kmeans.pkl", "rb") as file:
    kmeans = pickle.load(file)
lm = makeLMfilters()
la_subset = la.iloc[:2250, :].copy()
textures_subset = la_subset.apply(lambda x: build_hist(x.name, lm, kmeans),
                                  result_type="expand", axis=1)
la_subset = pd.concat([la_subset, textures_subset],
                      axis=1)
la_subset.rename(columns={i: f"texture_{i}" for i in range(len(kmeans.cluster_centers_))},
                 inplace=True)
la_subset.to_csv("D:/la_tester.csv", sep=",")

# Texton histograms
with open("kmeans.pkl", "rb") as file:
    kmeans = pickle.load(file)
lm = makeLMfilters()
la = pd.concat([la, la.apply(lambda x: build_hist(x.name, lm, kmeans), axis=1)],
               result_type="expand", axis=1)
la.rename(columns={i: f"texture_{i}" for i in range(len(kmeans.cluster_centers_))},
          inplace=True)

# Save data
la.to_csv("D:/la.csv", sep=",")
