import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([123.0, 132.0, 32.0, 43.0])

provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='10m',
    facecolor='none')

ax.add_feature(provinces, edgecolor='gray', linewidth=0.5)
plt.savefig("test_admin.png")
