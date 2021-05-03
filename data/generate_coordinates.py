from shapely.geometry import shape, Point
import random


def make_polygons(geojs):
    """Returns a dictionary of MultiPolygon objects indexed by their geoid."""
    polygons = {}
    for block in range(len(geojs["features"])):
        geoid = geojs["features"][block]["properties"]["geoid"]
        polygon = shape(geojs["features"][block]["geometry"])
        polygons[geoid] = polygon
    return polygons


def random_coord(polygon):
    """Generates a Point object lying within a MultiPolygon."""
    min_x, min_y, max_x, max_y = polygon.bounds
    while True:
        coord = Point(random.uniform(min_x, max_x),
                      random.uniform(min_y, max_y))
        if polygon.contains(coord):
            return coord
