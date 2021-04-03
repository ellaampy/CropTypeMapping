// stats generated for normalisation has been done using best effort to improve computational efficiency
// for precise statistics, remove bestEffort parameter and set scale :10


var department = ee.FeatureCollection('users/stella/cde/france_arrondissement')

// department = department.filter(ee.Filter.eq('nom','Brest'))
department = department.filter(ee.Filter.inList('nom', ee.List(['Brest', 'Morlaix', 'Quimper', 'Châteaulin'])))

var parcel = table.filter(ee.Filter.eq('ID_PARCEL','2890459')) 
//convex hull
// var convex = department.geometry().convexHull()
// var convex_buffer = convex.buffer(100)
// department = sample
var s1 = imageCollection2.filterDate('2018-10-01','2019-12-31').filterBounds(department)
                    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                    .sort('system:time_start', true)
                    .filter(ee.Filter.eq('relativeOrbitNumber_start', 154))
                    // .first()
                    .select(['VV','VH'])

// Map.addLayer(s1.select('VV').mean(),{min: -25, max: 1}, 'Temporal average SAR image')
print('initial collection', s1)

// Combine the mean and standard deviation reducers.
var reducers = ee.Reducer.mean().combine({
  reducer2: ee.Reducer.stdDev(),
  sharedInputs: true
});


var s1_setNorm = s1.map(function(img){
  // return img.set('stats', img.reduceRegion({reducer:ee.Reducer.percentile([2, 98]), scale: 10, maxPixels:1e15}))})
  return img.set({'stats': img.reduceRegion({reducer:ee.Reducer.percentile([2, 98]), bestEffort:true}), 'doa':ee.Date(img.get('system:time_start')).format('YYYYMMDD'),
    'mean_std': img.reduceRegion({reducer:reducers, bestEffort:true})
  })})



print(s1_setNorm)

var featList = s1_setNorm.map(function(img){
  return ee.Feature(null).set({'doa': img.get('doa'), 'stats_percentile':img.get('stats'), 'mean_std':img.get('mean_std')})})


Export.table.toDrive({
  collection: featList,
  folder: 'Earth Engine',
  description:'S1_norm_data',
  fileFormat: 'CSV',
});


---------------------------------------------------------------------------------------------------------------


