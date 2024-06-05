import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math

import _Utils.geographic_maths as GEO

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt


request_OSM = cimgt.OSM()

plot_in_progress = {}

class Plotter:

    def __init__(self, fig:plt.Figure, ax:plt.Axes) -> None:
        self.fig = fig
        self.ax = ax


    def plot(self, *args, **kwargs) -> None:
        self.ax.plot(*args, **kwargs, transform=ccrs.PlateCarree())

    def scatter(self,*args, **kwargs) -> None:
        self.ax.scatter(*args, **kwargs, transform=ccrs.PlateCarree())

    def title(self,title:str) -> None:
        self.ax.set_title(title)

    def legend(self) -> None:
        self.ax.legend()

    def set_aspect(self, aspect:str) -> None:
        self.ax.set_aspect(aspect)



class PLT:
    @staticmethod
    def figure(tag:str, min_lat:float, min_lon:float, max_lat:float, max_lon:float,
               figsize:"tuple[float, float]"=(10, 10), sub_plots:"tuple[int, int]"=(1, 1),
               display_map:"list[list[bool]]"=None, ratios:"list[list[bool]]"=None) -> None:

        circumference = 40075017

        fig, ax = plt.subplots(sub_plots[0], sub_plots[1], figsize=figsize,
                        subplot_kw=dict(projection=request_OSM.crs))

        if (sub_plots[0] == 1 and sub_plots[1] == 1):
            ax = [[ax]]
        elif (sub_plots[0] == 1):
            ax = [ax]
        elif (sub_plots[1] == 1):
            ax = [[ax[i]] for i in range(sub_plots[0])]


        if (display_map is None):
            display_map = [ [True] * sub_plots[1]] * sub_plots[0]
        if (ratios is None):
            ratios = [[1] * sub_plots[1]] * sub_plots[0]
        if (not(isinstance(min_lat, list))):
            min_lat = [[min_lat] * sub_plots[1]] * sub_plots[0]
            min_lon = [[min_lon] * sub_plots[1]] * sub_plots[0]
            max_lat = [[max_lat] * sub_plots[1]] * sub_plots[0]
            max_lon = [[max_lon] * sub_plots[1]] * sub_plots[0]


        for i in range(sub_plots[0]):
            for j in range(sub_plots[1]):
                if (display_map[i][j]):
                    box_size = GEO.distance(min_lat[i][j], min_lon[i][j], max_lat[i][j], min_lon[i][j])

                    zoom = math.ceil(math.log2(circumference / box_size))

                    ax[i][j].set_extent([min_lon[i][j], max_lon[i][j], min_lat[i][j], max_lat[i][j]])
                    ax[i][j].add_image(request_OSM, zoom)
                    ax[i][j].set_aspect('equal')
                else:

                    ax[i][j].set_aspect('auto')


        plotters:"list[list[Plotter]]" = []
        for i in range(sub_plots[0]):
            plotters.append([])
            for j in range(sub_plots[1]):
                plotters[i].append(Plotter(fig, ax[i][j]))

        PLT.__save__(tag, plotters)


    @staticmethod
    def subplot(tag:str, ax1:int=None, ax2:int=None) -> Plotter:
        if (ax1 is None):
            return PLT.__load__(tag)[0][0]
        return PLT.__load__(tag)[ax1][ax2]

    @staticmethod
    def show(tag:str, path:str=None, pdf:PdfPages=None) -> None:
        fig = PLT.__load__(tag)[0][0].fig

        fig.tight_layout()
        if (pdf is not None):
            pdf.savefig(fig, bbox_inches='tight')
        elif (path is not None):
            fig.savefig(path, bbox_inches='tight')
        else:
            fig.show()
        plt.close(fig)

        del plot_in_progress[tag]

    @staticmethod
    def savefig(tag:str, path:str=None, pdf:PdfPages=None) -> None:
        PLT.show(tag, path, pdf)

    @staticmethod
    def plot(tag:str, *args, **kwargs) -> None:
        plotter = PLT.__load__(tag)[0][0]
        plotter.plot(*args, **kwargs)

    @staticmethod
    def scatter(tag:str, *args, **kwargs) -> None:
        plotter = PLT.__load__(tag)[0][0]
        plotter.scatter(*args, **kwargs)

    @staticmethod
    def title(tag:str, title:str) -> None:
        plotter = PLT.__load__(tag)[0][0]
        plotter.title(title)

    @staticmethod
    def legend(tag:str) -> None:
        plotter = PLT.__load__(tag)[0][0]
        plotter.legend()




# |====================================================================================================================
# | SAVE AND LOADS
# |====================================================================================================================

    @staticmethod
    def __save__(tag:str, plotter: "list[list[Plotter]]") -> None:
        global plot_in_progress

        if (tag not in plot_in_progress):
            plot_in_progress[tag] = {}

        plot_in_progress[tag]["plotter"] = plotter


    @staticmethod
    def __load__(tag:str) -> "list[list[Plotter]]":
        return plot_in_progress[tag]["plotter"]


    @staticmethod
    def attach_data(tag:str, data:object) -> None:
        """
        Save data related to your plot for later use
        """
        global plot_in_progress
        if (tag not in plot_in_progress):
            plot_in_progress[tag] = {}
        plot_in_progress[tag]["data"] = data


    @staticmethod
    def get_data(tag:str) -> object:
        return plot_in_progress[tag]["data"]



class Color:
    TRAJECTORY = "tab:blue"
    PREDICTION = "tab:purple"
    
    TRAIN = "tab:blue"
    TEST = "tab:orange"
