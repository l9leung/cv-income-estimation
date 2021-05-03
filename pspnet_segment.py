from keras_segmentation.pretrained import pspnet_101_cityscapes
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import random


class segmenter:
    def __init__(self):
        # Define Cityscapes labels and colors
        self.color_dict = {"road": (128, 64, 128), "sidewalk": (244, 35, 232),
                           "building": (70, 70, 70), "wall": (102, 102, 156),
                           "fence": (190, 153, 153), "pole": (153, 153, 153),
                           "traffic light": (250, 170, 30), "traffic sign": (220, 220, 0),
                           "vegetation": (107, 142, 35), "terrain": (152, 251, 152),
                           "sky": (70, 130, 180), "person": (220, 20, 60),
                           "rider": (255,  0,  0), "car": (0, 0, 142),
                           "truck": (0, 0, 70), "bus": (0, 60, 100),
                           "train": (0, 80, 100), "motorcycle": (0, 0, 230),
                           "bicycle": (119, 11, 32)}
        # Put labels into 4 categories
        self.label_dict = {"flat": np.array([self.color_dict[key] for key in ["road", "sidewalk"]],
                                        dtype="uint8"),
                           "building": np.array([self.color_dict[key] for key in ["building", "wall", "fence",
                                                                                  "pole", "traffic light",
                                                                                  "traffic sign"]],
                                                    dtype="uint8"),
                           "nature": np.array([self.color_dict[key] for key in ["vegetation", "terrain"]],
                                              dtype="uint8"),
                           "vehicle": np.array([self.color_dict[key] for key in ["car", "truck", "bus", "train",
                                                                                 "motorcycle", "bicycle"]],
                                               dtype="uint8")}
        # Lists of colors and labels
        self.colors = list(self.color_dict.values())
        # self.labels = list(self.label_dict.keys())
        self.labels = list(self.label_dict.keys())
        # Import pretrained PSPNet
        self.model = pspnet_101_cityscapes()

    def plotcolors(self):
        # Plot self defined colors
        fig, axes = plt.subplots(4, 5)
        for label, ax in zip(self.color_dict.keys(), axes.flatten()):
            ax.imshow(np.flip(np.array(self.color_dict[label]).reshape((1, 1, 3)),
                              axis=2))
            ax.set_title(label)
            ax.axis("off")
        fig.delaxes(axes.flatten()[19])
        plt.tight_layout()
        plt.show()

    def segment(self, frame, overlay_img=False, show_legends=False,
                resize=False, out_fname="segmented.png", plot=False):
        """Writes to a png file with the image's semantic labels."""
        if resize is True:
            frame = cv2.resize(frame, (300, 200))
        self.model.predict_segmentation(inp=frame,
                                        out_fname=out_fname,
                                        overlay_img=overlay_img,
                                        show_legends=show_legends,
                                        colors=self.colors,
                                        class_names=self.labels)
        if plot is True:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(np.flip(frame, axis=2))
            ax[0].axis("off")
            ax[1].imshow(np.flip(cv2.imread(out_fname), axis=2))
            ax[1].axis("off")
            plt.show()
        return

    def get_segment_masks(self, fname="segmented.png"):
        # Load image
        segmented = cv2.imread(fname)
        # Dictionary to hold masks indexed by category
        masks = {}
        for label in self.labels:
            b = np.isin(segmented[:, :, 0], self.label_dict[label][:, 0])
            g = np.isin(segmented[:, :, 1], self.label_dict[label][:, 1])
            r = np.isin(segmented[:, :, 2], self.label_dict[label][:, 2])
            mask = (b == True) & (g == True) & (r == True)
            masks[label] = ~mask
        return masks


if __name__ == "__main__":
    # Segmenter instance (pspnet)
    pspnet = segmenter()
    # Plot class colors
    pspnet.plotcolors()

    files = os.listdir("data/street_view/Los Angeles/images")
    random.seed(123)
    random.shuffle(files)

    for i, file in enumerate(files[120:]):
        start = time.time()
        frame = cv2.imread(f"data/street_view/Los Angeles/images/{file}")
        out_fname = f"""data/street_view/Los Angeles/segmented/{file.split(".")[0]}_seg.png"""
        pspnet.segment(frame, overlay_img=False, show_legends=False, resize=False,
                       out_fname=out_fname, plot=False)
        end = time.time()
        print("#" + str(i + 1 + 120), end - start)
