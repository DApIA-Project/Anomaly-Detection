import matplotlib.pyplot as plt
import math

import _Utils.geographic_maths as GEO

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt


request_OSM = cimgt.OSM()

plot_in_progress = {}

class Plotter:

    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax


    def plot(self, *args, **kwargs):
        self.ax.plot(*args, **kwargs, transform=ccrs.PlateCarree())

    def scatter(self,*args, **kwargs):
        self.ax.scatter(*args, **kwargs, transform=ccrs.PlateCarree())

    def title(self,title):
        self.ax.set_title(title)





class PLT:
    @staticmethod
    def figure(tag:str, min_lat, min_lon, max_lat, max_lon, figsize=(10, 10), sub_plots=(1, 1), display_map=None, ratios=None):

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
                    # top_plot = None if (sub_plots[0] == 1) else (i - 1 if (i > 0) else i + 1)
                    # left_plot = None if (sub_plots[1] == 1) else (j - 1 if (j > 0) else j + 1)
                    # if (top_plot is None and left_plot is None):
                    #     pass

                    # ax_width = max_lat[top_plot ][j] - min_lat[top_plot ][j] if (top_plot is not None) else\
                    #             max_lat[i][left_plot] - min_lat[i][left_plot]
                    # ax_width *= ratios[i][j][0]

                    # ax_height = max_lon[i][left_plot] - min_lon[i][left_plot] if (left_plot is not None) else\
                    #             max_lon[top_plot][j] - min_lon[top_plot][j]
                    # ax_height *= ratios[i][j][1]

                    # # make figsize
                    # ax[i][j].set_extent([0, ax_width, 0, ax_height])


                    ax[i][j].set_aspect('auto')





        plotters = []
        for i in range(sub_plots[0]):
            plotters.append([])
            for j in range(sub_plots[1]):
                plotters[i].append(Plotter(fig, ax[i][j]))

        PLT.__save__(tag, plotters)


    @staticmethod
    def subplot(name, ax1:int=None, ax2:int=None) -> Plotter:
        if (ax1 is None):
            return PLT.__load__(name)[0][0]
        return PLT.__load__(name)[ax1][ax2]

    @staticmethod
    def show(tag:str, path=None, pdf=None):
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
    def plot(tag:str, *args, **kwargs):
        plotter = PLT.__load__(tag)[0][0]
        plotter.plot(*args, **kwargs)

    @staticmethod
    def scatter(tag:str, *args, **kwargs):
        plotter = PLT.__load__(tag)[0][0]
        plotter.scatter(*args, **kwargs)

    @staticmethod
    def title(tag:str, title):
        plotter = PLT.__load__(tag)[0][0]
        plotter.title(title)




# |====================================================================================================================
# | SAVE AND LOADS
# |====================================================================================================================

    @staticmethod
    def __save__(plot_name, plotter):
        global plot_in_progress

        if (plot_name not in plot_in_progress):
            plot_in_progress[plot_name] = {}

        plot_in_progress[plot_name]["plotter"] = plotter


    @staticmethod
    def __load__(plot_name):
        return plot_in_progress[plot_name]["plotter"]


    @staticmethod
    def attach_data(plot_name, data):
        """
        Save data related to your plot for later use
        """
        global plot_in_progress
        if (plot_name not in plot_in_progress):
            plot_in_progress[plot_name] = {}
        plot_in_progress[plot_name]["data"] = data


    @staticmethod
    def get_data(plot_name):
        return plot_in_progress[plot_name]["data"]