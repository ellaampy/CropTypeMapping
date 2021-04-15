import ee
import os
import numpy as np
import geojson, json
from tqdm import tqdm
from shapely.geometry import Polygon
import time
from datetime import datetime
from utils import *


#ee.Authenticate()
ee.Initialize()

# --------------------------------------------------------------
def prepare_dataset(rpg_file, output_dir, col_id, start_date, end_date, num_per_month=0, addNDVI =False, speckle_filter=False, label_names=['CODE_GROUP']):

    # catch inconsistent shapes
    np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)

    start = datetime.now()

    # prepare output directory
    prepare_output(output_dir)

    # get farm geometries & labels
    polygons, lab_rpg = parse_rpg(rpg_file, label_names=label_names)

    # dict of global metadata
    #dates = {k:[] for k in list(polygons.keys())}
    labels = dict([(l, {}) for l in lab_rpg.keys()])

    # counter for ignored parcels
    ignored = 0

    # iterate parcels
    for parcel_id, geometry in tqdm(polygons.items()):
        
        # get collection
        s1 = datetime.now()
        geometry = ee.Geometry.Polygon(geometry)
        collection = get_collection(geometry, col_id, start_date, end_date, num_per_month, addNDVI,speckle_filter)
        # print('fetching collection elapsed', datetime.now()- s1)
    

        # global normalize using 2nd & 98th percentile
        collection = collection.map(normalize)

        # check for incomplete and overlapping footprints
        # s2 = datetime.now()
        collection = collection.map(lambda img: img.set('temporal', ee.Image(img).reduceRegion(reducer = ee.Reducer.toList(), geometry= geometry, scale=10).values()))
        # print('removing duplicates and distincting overlapping tiles ', datetime.now()-s2) 

        # iterate collection and return array of TxCxN
        # s4 = datetime.now()
        try:
            # query pre-selected collection & make numpy array            
            np_all_dates = np.array(collection.aggregate_array('temporal').getInfo())
            assert np_all_dates.shape[-1] > 0 
            

        # except:
        except:
            print('Error in parcel --------------------> {}'.format(parcel_id))
            with open(os.path.join(output_dir, 'META', 'ignored_parcels.json'), 'a+') as file:
                file.write(json.dumps(int(parcel_id))+'\n')
            ignored += 1
            

        else:
            # print('initial array created for',parcel_id, np_all_dates.shape,datetime.now()- start)

            # create date metadata
            # s3 = datetime.now()  
            #date_series = collection.aggregate_array('doa').getInfo()
            #dates[str(parcel_id)] = date_series
            # print('date series ', datetime.now() - s3)

            # s6 = datetime.now()
            # save lABELS
            for l in labels.keys():
                labels[l][parcel_id] = int(lab_rpg[l][parcel_id])
            # print(' append labels', datetime.now()- s6)
            
            
            # save .npy files
            s7 = datetime.now()
            if 'S2' in col_id:

                # save spectral bands
                np.save(os.path.join(output_dir, 'DATA', str(parcel_id)), np_all_dates[:,:10,:]) # slice spectral bands
                
                #save slc 
                np.save(os.path.join(output_dir, 'QA', str(parcel_id)+'_QA'), np_all_dates[:,-1:,:]) #slice QA info

            elif 'S1' in col_id:
                np.save(os.path.join(output_dir, 'DATA', str(parcel_id)), np_all_dates)
            
            # print('npy saved ', datetime.now() - s7)
            print('parcel {} complete all processes+ download in {}'.format(parcel_id, datetime.now() - s1))
            

        # save global metadata (labels)
        with open(os.path.join(output_dir, 'META', 'labels.json'), 'w') as file:
            file.write(json.dumps(labels, indent=4))

        #with open(os.path.join(output_dir, 'META', 'dates.json'), 'w') as file:
            #file.write(json.dumps(dates, indent=4))


    # get processing time
    end = datetime.now()
    print('total ignored pixels', ignored)
    print(f"\n processing time -> {end - start}")


if __name__ == '__main__':

    args = parse_args()

    # on cluster
    #rpg_file = '/share/projects/fuses1s2/rpg/sample_farm.geojson'
    #rpg_file = '/share/projects/fuses1s2/rpg/Morlaix_RPG.geojson'
    #output_dir = '/share/projects/fuses1s2/s2_data/Morlaix'
    #col_id = 'COPERNICUS/S2_SR' #or COPERNICUS/S1_GRD or COPERNICUS/S2_SR

    prepare_dataset(args.rpg_file, args.output_dir, args.col_id, args.start_date, args.end_date, args.num_per_month, args.addNDVI, args.speckle_filter, args.label_names)


