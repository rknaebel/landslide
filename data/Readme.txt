All datasets have an extent of 4993x4986 pixels with 5m resolution. The dataformat is GeoTiff.

---
Satellite data:
  - [YYYYMMDD].tif
  - annual coverage of RapidEye images. (data gap in 2010)
  - each file has 5 spectral bands: blue, green, red, RedEdge, NIR
  - Data is in TOA-Reflectance [0-10000]
  - NoData Value: -99
  - signed int16

---
NDVI derivations of each file.
  - [YYYYMMDD]_NDVI.tif
  - NDVI: Normalized difference vegetation index
  - each file has 1 band.
  - NDVI is usually in [-1,1] -> converted to [0,200]
  - NoData Value: 255
  - unsigned int 8

---
True Positives. true landlsides as masks for each satellite scene.
  - [YYYYMMDD]_mask_ls.tif
  - mask [0,1] 1: true positive (landslide) 0: background

---
Relief information
  - DEM_altitude.tif (height (integer) in meter per pixel)
  - DEM_slope.tif  (slope/inclination (float) in degree per pixel)