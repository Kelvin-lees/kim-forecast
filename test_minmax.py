import xarray as xr
import scipy.ndimage as ndimage
import numpy as np

ds = xr.open_dataset('g576_v091_easia_etc.2byte.ft000.2026042700.nc')
lon_min, lon_max, lat_min, lat_max = 123.0, 132.0, 32.0, 43.0
ds = ds.sel(lats=slice(lat_max, lat_min), lons=slice(lon_min, lon_max))
mslp = np.squeeze(ds['psl'].values)
mslp_smooth = ndimage.gaussian_filter(mslp, sigma=1.5)

print("Sliced shape:", mslp.shape)
for size in [20, 30, 40, 50, 60, 80]:
    local_max = ndimage.maximum_filter(mslp_smooth, size=size) == mslp_smooth
    local_min = ndimage.minimum_filter(mslp_smooth, size=size) == mslp_smooth
    print(f"Size {size}: Max {np.sum(local_max)}, Min {np.sum(local_min)}")
