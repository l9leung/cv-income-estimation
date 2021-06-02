import sys
import pickle
import geojson
import requests
import random
from data.generate_coordinates import make_polygons, random_coord

random.seed(123)


def get_images(blocks, width=600, height=400, heading=None, fov=90, pitch=0,
               source="default", city="Los Angeles"):
    """
    Retrives a Google Street View static image from the specified lat/long
    coordinates. Expects a valid Google Cloud API key to be stored in a txt
    file in the current directory.

    Parameters
    ----------
    blocks : dict, Geoid and MultiPolygon object pairs for each block group.
    width : int, Width of the image in pixels. The default is 600.
    height : int, Height of the image in pixels. The default is 400.
    heading : int, Compass heading of the camera in degrees.The default is 0,
        which indicates north. If no heading is specified, a value will be
        calculated that directs the camera towards the specified location, from
        the point at which the closest photograph was taken.
    fov: int, Horizontal field of view of the image in degrees. The default is
        90.
    pitch : int, The up or down angle of the camera in degrees. 0 is flat
        horizontal, 90 is straight up. The default is 0.
    source : str, Limits searches to images from certain sources. "outdoor"
        limits searches to outdoor images.
    Returns
    -------
    None.

    """

    # Read API key
    key = []
    with open("key.txt") as f:
        for line in f:
            key.append(line)
    key = f"key={key[0]}"

    # Dictionary to store image coordinates
    coords = {}
    # List to store blocks with no images found
    zero_results = []

    # Request parameters
    url = "https://maps.googleapis.com/maps/api/streetview"
    size = f"size={width}x{height}"
    if heading is not None:
        heading = f"heading={heading}"
    pitch = f"pitch={pitch}"
    source = "source=outdoor"

    for i, block in enumerate(blocks.keys()):
        count = 5
        tries = 0
        while count < 9 and tries < 200:  # retrieve up to 10 images per block group
            point = random_coord(blocks[block])
            coord = point.coords[0]  # note: this is in longitude, latitude
            location = f"location={coord[1]},{coord[0]}"  # location in latitude, longitude
            # Check if there is an image available for the location
            metadata_request = f"{url}/metadata?{location}&{source}&{key}"
            response = requests.get(metadata_request)
            tries += 1
            if response.ok is True:
                print(f"""Block #{i+1}, Try #{tries}: {response.json()["status"]}""")
                if response.json()["status"] == "OK":
                    # Save location in dictionary
                    coords[f"{block}_{count}"] = point
                    # Get image
                    if heading is None:
                        request = f"{url}?{size}&{location}&{pitch}&{source}&{key}"
                    else:
                        request = f"{url}?{size}&{location}&{heading}&{pitch}&{source}&{key}"
                    response = requests.get(request)
                    # Write image to a jpeg file
                    with open(f"data/street_view/{city}/images/{block}_{count}.jpeg", "wb") as f:
                        f.write(response.content)
                    count += 1
            if tries == 300:  # Cap at 300 requests
                zero_results.append(block)

    # Write coordinates of images to a geojson file
    with open(f"data/street_view/{city}/coordinates.geojson", "w") as f:
        geojson.dump(coords, f)
    # # Write geoids of blocks with no images found
    # with open(f"street_view/{city}/zero_results", "wb") as fp:
    #     pickle.dump(zero_results, fp)


if __name__ == "__main__":
    city = sys.arvg[1]

    # city="Los Angeles"
    # with open(f"data/acs/{city}/acs2019_5yr_B19013_15000US060372732001.geojson", "r") as f:
    #     blocks = geojson.load(f)
    # blocks = make_polygons(blocks)
    # get_images(blocks, heading=90, city="Los Angeles")  # set heading to east

    city = "New York"
    with open(f"data/acs/{city}/acs2019_5yr_B19013_15000US360810595004.geojson", "r") as f:
        blocks = geojson.load(f)
    blocks = make_polygons(blocks)
    get_images(blocks, city="New York")
