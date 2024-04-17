import matplotlib.pyplot as plt
import math

import _Utils.geographic_maths as GEO

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt


request_OSM = cimgt.OSM()

plot_in_progress = {}
def __save__(plot_name, fig, ax):
    """
    Save the current plot for continuing the plot later
    """
    global plot_in_progress

    if (plot_name not in plot_in_progress):
        plot_in_progress[plot_name] = {}

    plot_in_progress[plot_name]["fig"] = fig
    plot_in_progress[plot_name]["ax"] = ax


def __load__(plot_name):
    return plot_in_progress[plot_name]["fig"], plot_in_progress[plot_name]["ax"]



def attach_data(plot_name, data):
    """
    Save data related to your plot for later use
    """
    global plot_in_progress
    if (plot_name not in plot_in_progress):
        plot_in_progress[plot_name] = {}
    plot_in_progress[plot_name]["data"] = data


def get_data(plot_name):
    return plot_in_progress[plot_name]["data"]

def show(plot_name, png_path):
    fig, ax = __load__(plot_name)
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches='tight')
    plt.close(fig)

    del plot_in_progress[plot_name]


def figure(tag, min_lat, min_lon, max_lat, max_lon, figsize=(10, 10)):
    box_size = GEO.distance(min_lat, min_lon, max_lat, min_lon)
    circumference = 40075017
    zoom = math.ceil(math.log2(circumference / box_size))

    fig, ax = plt.subplots(figsize=figsize,
                    subplot_kw=dict(projection=request_OSM.crs))

    ax.set_extent([min_lon, max_lon, min_lat, max_lat])


    ax.add_image(request_OSM, zoom)

    __save__(tag, fig, ax)


def plot(tag, *args, **kwargs):
    fig, ax = __load__(tag)
    return ax.plot(*args, **kwargs, transform=ccrs.PlateCarree())

def scatter(tag, *args, **kwargs):
    fig, ax = __load__(tag)
    return ax.scatter(*args, **kwargs, transform=ccrs.PlateCarree())

def title(tag, title):
    fig, ax = __load__(tag)
    return ax.set_title(title)

