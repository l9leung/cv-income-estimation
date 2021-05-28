import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from data.acs_load import acs_load

random.seed(123)

def rand_block_groups(city="Los Angeles"):
    """Plots 3 random block groups with their random coordinates."""
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    for j in range(3):
        index = random.randint(0, len(acs[city]))
        geoid = acs[city][index:index+1]["geoid"].values[0]
        acs[city][index:index+1].plot(ax=ax[j], color="#C3ECB2")
        coords[city][coords[city]["geoid"] == geoid].plot(ax=ax[j],
                                                          color="#4A89F3")
        ax[j].ticklabel_format(style="plain", useOffset=False)
        ax[j].set_title(geoid)
        ax[j].axis("off")
    plt.tight_layout()
    plt.savefig(f"data/visualizations/randblocks_{city}", dpi=720,
                bbox_inches="tight")
    plt.show()


def coords_map(city="Los Angeles", save=False):
    """Plots locations of all random coordinates."""
    geodf = coords["Los Angeles"].merge(acs["Los Angeles"][["geoid", "B19013001"]],
                                        left_on="geoid", right_on="geoid")
    fig, ax = plt.subplots(figsize=(10, 10))
    acs[city].plot(ax=ax, color="#C3ECB2")
    # coords[city].plot(ax=ax, markersize=1, color="#4A89F3")
    coords[city].plot(ax=ax, markersize=1, c=geodf["B19013001"])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    # ax.set_title(f"Randomly Generated Coordinates for {city}")
    if save is True:
        plt.savefig(f"data/visualizations/coordinates_{city}.png",
                    dpi=720,
                    bbox_inches="tight")
    plt.show()


def plot_geoid(geoid, income, city="Los Angeles", save=False):
    """Plot up to 9 available images from a geoid."""
    files = [file for file in os.listdir(f"data/street_view/{city}/images") if geoid in file]
    if len(files) > 9:
        files = files[:9]
    fig, axes = plt.subplots(3, 3)
    for i, (file, ax) in enumerate(zip(files, axes.flatten())):
        frame = cv2.imread(f"data/street_view/{city}/images/{file}")
        ax.imshow(np.flip(frame, axis=2))
        ax.axis("off")
        if i == 1:
            ax.set_title(f"${int(income*1000):,d}")
    plt.tight_layout()
    if save is True:
        plt.savefig(f"data/visualizations/{geoid}.png", dpi=720)
    plt.show()


def income_histogram(city="Los Angeles", log=False, save=False):
    """Plots the distribution of block group level median household income."""
    plt.figure()
    if log is True:
        sd = round(acs[city]["log_B19013001"].std(), 2)
        plt.hist(acs[city]["log_B19013001"], color="tab:red",
                 edgecolor="k", label=f"Std Dev = {sd}")
        plt.xlabel("Log Dollars")
        plt.ylabel("Count")
    else:
        sd = round(acs[city]["B19013001_thousand"].std(), 2)
        plt.hist(acs[city]["B19013001_thousand"], color="#4A89F3",
                 edgecolor="k", label=f"Std Dev = {sd}")
        plt.xlabel("Thousands of Dollars")
        plt.ylabel("Count")
    plt.legend()
    plt.title(f"Household Income in {city}")
    if save is True:
        plt.savefig(f"data/visualizations/income_hist_{city}.png",
                    dpi=720,
                    bbox_inches="tight")
    plt.show()


def income_choropleth(city="Los Angeles", log=False, save=False):
    """Maps median household income by block group."""
    fig, ax = plt.subplots(figsize=(15, 10))
    # alternative cmap: "RdYlGn_r"
    if log is True:
        acs[city].plot(column="log_B19013001", ax=ax, cmap="plasma",
                       legend=True, missing_kwds={"color": "lightgrey"})
        ax.set_title(f"Household Income in {city} (Log Dollars)")

    else:
        acs[city].plot(column="B19013001_thousand", ax=ax, cmap="viridis",
                       legend=True, missing_kwds={"color": "lightgrey"})
        ax.set_title(f"Household Income in {city} (Thousands of Dollars)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if save is True:
        plt.savefig(f"data/visualizations/income_choro_{city}.png",
                    dpi=720,
                    bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    acs, coords = acs_load(ret_coords=True)
    rand_block_groups(city="Los Angeles")
    coords_map()
    income_histogram()
    income_choropleth(save=True)

    for i, row in acs["Los Angeles"].sample(10, random_state=123).iterrows():
        plot_geoid(row["geoid"], row["B19013001_thousand"], save=True)
