import numpy as np
import geojson
import geopandas as gpd
from shapely.geometry import Point
import os
import cv2
import matplotlib.pyplot as plt


def acs_load(cities=["Los Angeles"], ret_coords=False):
    """
    Returns
    -------
    acs : dict, City indexed GeoDataFrames of ACS income.
    coords: dict, City indexed GeoDataFrames of street view image locations.

    """
    ids = {"Los Angeles": "15000US060372732001", "Boston": "15000US250250108021",
           "New York": "15000US360810595004", "San Diego": "15000US060730078003",
           "San Francisco": "15000US060750123021"}
    acs = {}
    coords = {}
    for city in cities:
        # Load ACS
        acs[city] = gpd.read_file(f"data/acs/{city}/acs2019_5yr_B19013_{ids[city]}.geojson")
        # Get log of income
        acs[city]["log_B19013001"] = np.log(acs[city]["B19013001"])
        # Get income in thousands of dollars
        acs[city]["B19013001_thousand"] = acs[city]["B19013001"] / 1000

    if ret_coords is True:
        # Coordinates of images in each block group
        with open(f"data/street_view/{city}/coordinates.geojson") as f:
            coords_city = geojson.load(f)
        # Convert coordinates to GeoDataFrame
        coords_city = gpd.GeoDataFrame(coords_city).transpose()
        coords_city["geometry"] = coords_city["coordinates"].apply(Point)
        coords_city["geoid"] = coords_city.index
        coords_city["geoid"] = coords_city["geoid"].str.split("_", n=1, expand=True)[0]
        coords_city = coords_city.drop(columns=["type", "coordinates"])
        coords[city] = coords_city
        return acs, coords
    else:
        return acs


if __name__ == "__main__":
    acs = acs_load(cities=["Los Angeles"])
    acs = acs["Los Angeles"]
