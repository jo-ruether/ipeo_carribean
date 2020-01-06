import rasterio
from rasterio.plot import show
from rasterio.mask import mask
from os import makedirs
from os.path import exists

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from os.path import join

from pyproj import Proj, transform

def preprocess_training(geojson_train,
                        image_fp,
                        profile,
                        roofs_train_dir):

    # Read the geojson file and etract feature data (coordinates, material...)
    with open(geojson_train) as geojson:
        geoms = json.loads(geojson.read())
        roofs = geoms['features']

    outProj = Proj(init=profile['crs']) # CRS format of image
    inProj = Proj(init='epsg:4326') # lat/lon coordinate format

    # Iterate over all roofs described in the geojson
    for roof in roofs:
        roof_id = roof['id']
        roof_geom = roof['geometry']
        roof_material = roof['properties']['roof_material']
        print(roof_id)

        # Transform oordinates from local to global format
        if roof_geom['type'] == 'MultiPolygon':
            print("MULTIPOLYGON")
            continue
        else:
            coord = roof_geom['coordinates'][0]
            for c in coord:
                c[0], c[1] = transform(inProj, outProj, c[0], c[1])

        # Cut out the roof
        with rasterio.open(image_fp) as image:
            roof_image, roof_transform = mask(image, [roof_geom],
                                             filled=True, crop=True)
        roof_meta = image.meta.copy()
        roof_meta.update({"driver": "GTiff",
            "dtype": rasterio.uint8,
            "height": roof_image.shape[1],
            "width": roof_image.shape[2],
            "transform": roof_transform,
            "tiled": True,
            "compress": 'lzw'})

        # Save the cut out roof image to disk
        roof_image_fp = join(roofs_train_dir, roof_material, str(roof_id)+".tif")
        with rasterio.open(roof_image_fp, "w", **roof_meta) as dest:
            dest.write(roof_image)
    print("Preprocessing training finished")


def preprocess_test(geojson_test,
                    image_fp,
                    profile,
                    roofs_test_dir):

    # Read the geojson file and etract feature data (coordinates, material...)
    with open(geojson_test) as geojson:
        geoms = json.loads(geojson.read())
        roofs = geoms['features']

    outProj = Proj(init=profile['crs']) # CRS format of image
    inProj = Proj(init='epsg:4326') # lat/lon coordinate format

    # Iterate over all roofs described in the geojson
    for roof in roofs:
        roof_id = roof['id']
        roof_geom = roof['geometry']
        print(roof_id)

        # Transform oordinates from local to global format
        if roof_geom['type'] == 'MultiPolygon':
            print("MULTIPOLYGON")
            continue
        else:
            coord = roof_geom['coordinates'][0]
            for c in coord:
                c[0], c[1] = transform(inProj, outProj, c[0], c[1])

        # Cut out the roof
        with rasterio.open(image_fp) as image:
            roof_image, roof_transform = mask(image, [roof_geom],
                                             filled=True, crop=True)
        roof_meta = image.meta.copy()
        roof_meta.update({"driver": "GTiff",
            "dtype": rasterio.uint8,
            "height": roof_image.shape[1],
            "width": roof_image.shape[2],
            "transform": roof_transform,
            "tiled": True,
            "compress": 'lzw'})

        # Save the cut out roof image to disk
        roof_image_fp = join(roofs_test_dir, str(roof_id)+".tif")
        with rasterio.open(roof_image_fp, "w", **roof_meta) as dest:
            dest.write(roof_image)

    print("Preprocessing test finished")

def preprocess_region(region):
    """Short summary.

    Parameters
    ----------
    region : type
        Description of parameter `region`.

    Returns
    -------
    type
        Description of returned object.

    """

    # Assemble the filepaths for geojson, full image and output directory
    geojson_train = join('..', '..', 'data', region, 'train-'+region+'.geojson')
    geojson_test = join('..', '..', 'data', region, 'test-'+region+'.geojson')
    image_fp = join('..', '..', 'data', region, region+'_ortho-cog.tif')
    roofs_train_dir = join('..', '..', 'data', region, 'roofs_train')
    roofs_test_dir = join('..', '..', 'data', region, 'roofs_test')

    # Retrieve images metadata (we need the CRS)
    with rasterio.open(image_fp) as src:
        profile = src.profile

    # Create the necessary subdirectories
    materials = ['healthy_metal', 'irregular_metal',
                 'concrete_cement', 'incomplete', 'other']
    for mat in materials:
        directory = join(roofs_train_dir, mat)
        if not exists(directory):
            makedirs(directory)

    if not exists(roofs_test_dir):
        makedirs(roofs_test_dir)

    preprocess_training(geojson_train, image_fp,
                        profile, roofs_train_dir)

    preprocess_test(geojson_test, image_fp,
                    profile, roofs_test_dir)
