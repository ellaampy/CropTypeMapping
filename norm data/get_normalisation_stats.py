// stats generated for normalisation has been done using best effort to improve computational efficiency
// for precise statistics, remove bestEffort parameter and set scale :10



var department = ee.FeatureCollection('users/stella/cde/france_arrondissement')
department = department.filter(ee.Filter.inList('nom', ee.List(['Brest', 'Morlaix', 'Quimper', 'Châteaulin'])))

//convex hull
// var convex = department.geometry().convexHull()
// var convex_buffer = convex.buffer(100)
// department = sample
var start_date = '2018-10-01'
var end_date = '2019-12-31'

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterDate(start_date, end_date).filterBounds(department)
                    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                    .sort('system:time_start', true)
                    .filter(ee.Filter.eq('relativeOrbitNumber_start', 154))
                    .select(['VV','VH'])

var s2 =  ee.ImageCollection('COPERNICUS/S2_SR').filterDate(start_date,end_date).filterBounds(department).filter(
          ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE',80)).select(['B2','B3','B4','B5', 'B6','B7','B8','B8A','B11','B12'])


print('initial collection', s1)
print('initial collection', s2)

// Combine the mean and standard deviation reducers.
var reducers = ee.Reducer.mean().combine({
  reducer2: ee.Reducer.stdDev(),
  sharedInputs: true
});


// get stats per band : 2&98 percentile, mean, std
var norm = function(img){
  // return img.set('stats', img.reduceRegion({reducer:ee.Reducer.percentile([2, 98]), scale: 10, maxPixels:1e15}))})
  return img.set({'stats': img.reduceRegion({reducer:ee.Reducer.percentile([2, 98]), bestEffort:true}), 'doa':ee.Date(img.get('system:time_start')).format('YYYYMMDD'),
    'mean_std': img.reduceRegion({reducer:reducers, bestEffort:true})
  })}

var create_table = function(img){
  return ee.Feature(null).set({'doa': img.get('doa'), 'stats_percentile':img.get('stats'), 'mean_std':img.get('mean_std')})}


var s1_setNorm = s1.map(norm).map(create_table)
var s2_setNorm = s2.map(norm).map(create_table)


Export.table.toDrive({
  collection: s2_setNorm,
  folder: 'Earth Engine',
  description:'S2_norm_data',
  fileFormat: 'CSV',
});


//---------------------------------------------------------------------------------------------------------------





//---------------------------------------------------------------------------------------------------------------


