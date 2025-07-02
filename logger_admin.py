from _Utils.RunLogger import RunLogger
from _Utils import secrets_stuffs as S

LOGGER = RunLogger(host=S.IP, port=S.PORT)
loggers_per_problem = LOGGER.split_by("PROBLEM")

for i in range(len(loggers_per_problem)):
    if (loggers_per_problem[i].loc("PROBLEM", 0) == "AircraftClassification"):
        loggers_per_problem[i] = loggers_per_problem[i].where("ADD_MAP_CONTEXT", 0)
    loggers_per_problem[i] = loggers_per_problem[i].get_best_groupes_by("ACCURACY", "model", maximize=True)
loggers_per_problem:RunLogger = RunLogger.join(loggers_per_problem)

loggers_for_flooding_per_horizon = LOGGER.split_by("PROBLEM")
flood_i = 0
for i in range(len(loggers_for_flooding_per_horizon)):
    if (loggers_for_flooding_per_horizon[i].loc("PROBLEM", 0) == "FloodingSolver"):
        flood_i = i
        break
loggers_for_flooding_per_horizon = loggers_for_flooding_per_horizon[flood_i].get_best_groupes_by("ACCURACY", "HORIZON", maximize=True)


file = open("./_Artifacts/logs.txt", "w")
loggers_per_problem.group_by("PROBLEM").render(file, "Best models")
LOGGER.group_by("PROBLEM", inplace=False).render(file, "All runs")
# loggers_for_flooding_per_horizon.group_by("HORIZON").render(file, "FloodingSolver by horizon")
file.close()

# read and print the whole file
with open("./_Artifacts/logs.txt", "r") as file:
    print(file.read())
    
    
# loggers_for_flooding_per_horizon:RunLogger = loggers_for_flooding_per_horizon
    
# horizon = []
# accuracy = []
# error = []
# perfect = []

# for i in range(len(loggers_for_flooding_per_horizon)):
#     horizon.append(loggers_for_flooding_per_horizon.loc("HORIZON", i))
#     accuracy.append(loggers_for_flooding_per_horizon.loc("ACCURACY", i))
#     error.append(loggers_for_flooding_per_horizon.loc("ERROR", i))
#     perfect.append(loggers_for_flooding_per_horizon.loc("PERFECT", i))
    
# # make a plot with left y axis as accuracy and right y axis as error
# # accuracy is a bar plot and error is a line plot
# # 935329

# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.ticker as ticker

# fig, ax1 = plt.subplots(figsize=(10, 5))
# ax2 = ax1.twinx()

# # make a bar plot for accuracy
# # blue bar with alpha 0.5 and blue outline of 2px, and reduce bar width to 0.5
# ax1.bar(horizon, accuracy, color='tab:blue', alpha=0.7, edgecolor='tab:blue', linewidth=1, label='Accuracy', width=0.7)
# ax1.set_xlabel('Horizon')
# ax1.set_ylabel('Accuracy')
# # set the y axis to be between min-1 and 100
# ax1.set_ylim(bottom=np.min(accuracy)-1, top=100)
# # x limits are min(horizon) and max(horizon)
# ax1.set_xlim(left=np.min(horizon)-0.75, right=np.max(horizon) + 0.75)
# # set % and integer ticks
# ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0f} %'.format(x)))
# # x ticks are horizon [1, ..., 20

# # make a line plot for error dashed
# ax2.plot(horizon, error, color='tab:red', linestyle='--', linewidth=2, label='Error')
# ax2.set_ylabel('Error')
# # set m units
# ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0f} m'.format(x)))

# # make one legend for both plots
# lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# lines.extend(lines2)
# labels.extend(labels2)
# ax1.legend(lines, labels, loc=2)

# # show the grid
# ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# # set x axis interger ticks and only show ticks of existing horizon
# ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))

# # show
# plt.show()


# # save as pdf
# fig.savefig("./FloodingSolver_by_horizon.pdf", bbox_inches='tight')
    
    
    
    
# # aircraft classification timing:
# aircraft_classif = loggers_per_problem.where("PROBLEM", "AircraftClassification")
# for i in range(len(aircraft_classif)):
#     print(aircraft_classif.loc("model", i), aircraft_classif.loc("TIME", i) / 935329 * 1000.0)