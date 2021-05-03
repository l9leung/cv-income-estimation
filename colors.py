import numpy as np
import cv2
import matplotlib.pyplot as plt


def color_hists(frame, bins=256, mask=None):
    """Returns a 1D numpy array of the color histograms for the blue, green,
    and red color channels."""
    hists = np.zeros(shape=(bins * 3,), dtype="uint32")
    for i in range(3):
        hists[0+bins*i:bins+bins*i] = cv2.calcHist([frame],
                                                           channels=[i],
                                                           mask=mask,  # mask image
                                                           histSize=[bins],  # bin count
                                                           ranges=[0, 256]).ravel()
    return hists


def plot_color_hists(frame, bins=256, colors=["b", "g", "r"], mask=None):
    """Plots a histogram for each specified color channel."""
    for i, color in enumerate(colors):
        hist = cv2.calcHist([frame],
                            channels=[i],
                            mask=mask,  # mask image
                            histSize=[bins],  # bin count
                            ranges=[0, 256]).ravel()
        plt.plot(hist, color=color, linewidth=0.5)
    plt.xlim([0, bins])
    plt.xlabel("bin")
    plt.ylabel("count of pixels")
    plt.show()


def lab_color_hists(frame, bins=256, mask=None):
    """Color histograms in CIE L*a*b* colorspace. OpenCV standardizes the
    values to lie in 0-256."""
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    hists = np.zeros(shape=(bins * 3,), dtype="uint32")
    for i in range(3):
        hists[0+bins*i:bins+bins*i] = cv2.calcHist([frame],
                                                    channels=[i],
                                                    mask=mask,  # mask image
                                                    histSize=[bins],  # bin count
                                                    ranges=[0, 256]).ravel()
    return hists
