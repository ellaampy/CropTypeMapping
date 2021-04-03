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
