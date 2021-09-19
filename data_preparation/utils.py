import ee
import os
import numpy as np
import argparse
import geojson, json
from tqdm import tqdm
from shapely.geometry import Polygon
import time
from datetime import datetime


ee.Initialize()


# utils/ functions

def get_collection(geometry, col_id , start_date , end_date, num_per_month=0, addNDVI=False, speckle_filter=False):

    """
    Args:
        geometry: feature to use as bounds
        col_id: mission ID. Sentinel-2 surface reflectance = 'COPERNICUS/S2_SR', for Sentinel-1 = COPERNICUS/S1_GRD 
        clip : clip all images in collection to geometry
        num_per_month : number of images to return per month. for S2, sorted by cloud cover%
        speckle_filter : applies a temporal filtering technique
    """

    if 'S2' in col_id: 
        collection = ee.ImageCollection(col_id).filterDate(start_date,end_date).filterBounds(geometry).filter(
                     ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE',80)).select(
                     ['B2','B3','B4','B5', 'B6','B7','B8','B8A','B11','B12', 'QA60'])

        # set normalisation statistics (placed prior to any parcel clipping operation)
        collection = collection.map(lambda img: img.set('stats', ee.Image(img).reduceRegion(reducer=ee.Reducer.percentile([2, 98]), bestEffort=True)))

        # compute NDVI
        if addNDVI == True:
            collection = collection.map(lambda img: img.addBands(img.normalizedDifference(['B8', 'B4']).rename('ndvi')))


    elif 'S1'  in col_id:
        collection = ee.ImageCollection(col_id).filter(ee.Filter.eq('instrumentMode', 'IW')).filterDate(
                     start_date, end_date).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).filter(
                     ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')).filterBounds(geometry).select(['VV','VH']).filter(
                     ee.Filter.eq('orbitProperties_pass', 'DESCENDING')).sort('system:time_start', True).filter( 
                     ee.Filter.eq('relativeOrbitNumber_start', 154))

        # set normalisation statistics (placed prior to any parcel clipping operation)
        collection = collection.map(lambda img: img.set('stats', ee.Image(img).reduceRegion(reducer=ee.Reducer.percentile([2, 98]), bestEffort=True)))

        if speckle_filter == True:
            # collection = collection.map(lambda img: img.clip(geometry.bounds().buffer(200)))
            collection = multitemporalDespeckle(collection)

        # sort by doa for ordered date sequence
        #  projection used here for co-registration
        collection = collection.map(lambda img: img.reproject(crs = 'EPSG:32630', crsTransform = [10, 0, 399960, 0, -10, 5400000]))


    # checks for incomplete, duplicate footprints
    collection = overlap_filter(collection, geometry)
        

    # return one image per month
    if  num_per_month > 0:
        collection = monthly_(col_id, collection, start_year = int(start_date[:4]), end_year = int(end_date[:4]), num_per_month=num_per_month)

    return collection



def monthly_(col_id, collection, start_year, end_year, num_per_month):
    """
    description:
        return n images per month for a given year sequence
    """    
    months = ee.List.sequence(1, 12)
    years = ee.List.sequence(start_year, end_year)

    try:
        if 'S2' in col_id: 
            collection = ee.ImageCollection.fromImages(years.map(lambda y: months.map(lambda m: collection.filter(
                        ee.Filter.calendarRange(y, y, 'year')).filter(ee.Filter.calendarRange(m, m, 'month')).sort(
                        'CLOUDY_PIXEL_PERCENTAGE').toList(num_per_month))).flatten())
            
            # sort by doa for ordered date sequence
            collection = collection.sort('system:time_start')

                
        elif 'S1' in col_id: 
            collection = ee.ImageCollection.fromImages(years.map(lambda y: months.map(lambda m: collection.filter(
                        ee.Filter.calendarRange(y, y, 'year')).filter(ee.Filter.calendarRange(m, m, 'month'))
                        .toList(num_per_month))).flatten())
            
            collection = collection.sort('system:time_start')
            
        return collection


    except:
        print("collection cannot be filtered")


def prepare_output(output_path):
    # creates output directory
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'DATA'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'META'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'QA'), exist_ok=True)


def parse_rpg(rpg_file, label_names=['CODE_GROUP']):
    """Reads rpg and returns a dict of pairs (ID_PARCEL : Polygon) and a dict of dict of labels
     {label_name1: {(ID_PARCEL : Label value)},
      label_name2: {(ID_PARCEL : Label value)}
     }
     """
    # Read rpg file
    print('Reading RPG . . .')
    with open(rpg_file) as f:
        data = json.load(f)

    # Get list of polygons
    polygons = {}
    lab_rpg = dict([(l, {}) for l in label_names])

    for f in tqdm(data['features']):
        # p = Polygon(f['geometry']['coordinates'][0][0])
        p = f["geometry"]["coordinates"][0]  
        polygons[f['properties']['ID_PARCEL']] = p
        for l in label_names:
            lab_rpg[l][f['properties']['ID_PARCEL']] = f['properties'][l]
    return polygons, lab_rpg


def shapely2ee(geometry):
    # converts geometry to GEE server object
    pt_list = list(zip(*geometry.exterior.coords.xy))
    return ee.Geometry.Polygon(pt_list)


def geom_features(geometry):
    # computes geometric info per parcel
    area  = geometry.area().getInfo()
    perimeter = geometry.perimeter().getInfo()
    bbox = geometry.bounds()
    return perimeter, perimeter/area, bbox


def overlap_filter(collection, geometry):

    # set masked/no data pixels to -9999
    collection = collection.filterBounds(geometry).map(lambda image: ee.Image(image).unmask(-9999).clip(geometry))
    
    #add image properties {doa, noData & overlap assertions}
    collection = collection.map(lambda image: image.set({
        'doa': ee.Date(image.get('system:time_start')).format('YYYYMMdd'),
        'noData': ee.Image(image).clip(geometry).reduceRegion(ee.Reducer.toList(), geometry).values().flatten().contains(-9999),
        'overlap': ee.Image(image).geometry().contains(geometry, 0.01)}))
    
    # remove tiles containing masked pixels, select one of many overlapping tiles over a parcel
    collection = collection.filter(ee.Filter.eq('noData', False)).filter(ee.Filter.eq('overlap',True)).distinct('doa')
    
    # !- set in prepare dataset function
    # collection = collection.map(lambda img: img.set(
    #     'temporal', ee.Image(img).reduceRegion(reducer = ee.Reducer.toList(), geometry= geometry, scale=10).values()))
    
    return collection


def parse_args():
    parser = argparse.ArgumentParser(description='Query GEE for reflectance data and return numpy array per feature')
    parser.add_argument('--rpg_file', type=str, help="path to json with attributes ID_PARCEL, CODE_GROUP")
    parser.add_argument('--output_dir', type=str, help='output directory')
    parser.add_argument('--col_id', type=str, default="COPERNICUS/S2_SR", help="GEE collection ID e.g. 'COPERNICUS/S2_SR' or 'COPERNICUS/S1_GRD'")
    parser.add_argument('--start_date', type=str,  default='2018-10-01', help='start date YYYY-MM-DD')
    parser.add_argument('--end_date', type=str,  default='2019-12-31', help='end date YYYY-MM-DD')
    parser.add_argument('--num_per_month', type=int, default=0, help='number of tiles per month')
    parser.add_argument('--addNDVI', type=bool, default=False, help='computes and append ndvi as additional band')  
    parser.add_argument('--speckle_filter', type=bool, default=False, help='computes and append ndvi as additional band')     
    parser.add_argument('--label_names', type=list, default=['CODE_GROUP'], help='label column name in json') 
    return parser.parse_args()


def normalize(img):
    img = ee.Image(img)
    def norm_band(name):
        name = ee.String(name)
        stats = ee.Dictionary(img.get('stats'))
        p2 = ee.Number(stats.get(name.cat('_p2')))
        p98 = ee.Number(stats.get(name.cat('_p98')))
        stats_img = img.select(name).subtract(p2).divide((p98.subtract(p2)))
        return stats_img
    
    new_img = img.addBands(srcImg = ee.ImageCollection.fromImages(img.bandNames().map(norm_band)).toBands().rename(img.bandNames()), overwrite=True)
    return new_img.toFloat()



def multitemporalDespeckle(images, radius = 70, units ='meters', opt_timeWindow={'before': -2, 'after': 2, 'units': 'month'}):

    bandNames = ee.Image(images.first()).bandNames()
    bandNamesMean = bandNames.map(lambda b: ee.String(b).cat('_mean'))
    bandNamesRatio = bandNames.map(lambda b: ee.String(b).cat('_ratio'))

    # compute space-average for all images
    def space_avg(image):
        mean = image.reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.square(radius, units)).rename(bandNamesMean)
        ratio = image.divide(mean).rename(bandNamesRatio)
        return image.addBands(mean).addBands(ratio)

    meanSpace = images.map(space_avg)

    def multitemporalDespeckleSingle(image):
        t = image.date()
        start = t.advance(ee.Number(opt_timeWindow['before']), opt_timeWindow['units'])
        end = t.advance(ee.Number(opt_timeWindow['after']), opt_timeWindow['units'])
        meanSpace2 = ee.ImageCollection(meanSpace).select(bandNamesRatio).filterDate(start, end)
        b = image.select(bandNamesMean)
        return b.multiply(meanSpace2.sum()).divide(meanSpace2.size()).rename(bandNames).copyProperties(image, ['system:time_start', 'stats']) 

    # denoise images
    return meanSpace.map(multitemporalDespeckleSingle).select(bandNames)


