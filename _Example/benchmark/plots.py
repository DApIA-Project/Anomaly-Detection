
import matplotlib.pyplot as plt
import numpy as np


# translate to french

x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
y = [5.169904708862305, 5.759028911590576, 6.791625499725342, 50.42728281021118, 64.31765651702881, 73.93254518508911, 90.11358308792114, 115.85217785835266, 161.1922812461853, 250.57810235023499]

plt.plot(x, y, marker="o", markersize=4)
plt.xlabel("Nombre d'avions en simultané (n)")
plt.ylabel("Temps (s)")
plt.title("Temps d'execution, pour le traitement en temps réel\nde n vols de 512 messages ADS-B")
plt.grid()
plt.savefig("time_per_simultaneous_aircraft_fr.png", dpi=300)
plt.close()

x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
y = [47.66769361495972, 27.110658645629883, 15.508031606674194, 9.392610311508179, 5.985950708389282, 4.123213052749634, 3.1290125846862793, 2.7381110191345215, 2.3267195224761963, 2.2480316162109375]
c = [511, 256, 128, 64, 32, 16, 8, 4, 2, 1]

plt.plot(x, y, marker="o", markersize=4)
plt.xlabel("Vitesse de simulation (x n)")
plt.ylabel("Temps (s)")
plt.title("Temps d'execution, pour le traitement de\n8 vols de 512 messages ADS-B en accéléré n fois")
plt.grid()
plt.savefig("time_per_acc_fr.png", dpi=300)
plt.close()


y = [y[i] / c[i] for i in range(len(x))]

plt.plot(x, y, marker="o", markersize=4)
# threshold = 1
plt.plot([x[0], x[-1]], [1, 1], color="red", linestyle="--", label="Limite du temps réel")
plt.xlabel("Vitesse de simulation (x n)")
plt.ylabel("Temps (s)")
plt.title("Temps d'execution d'un appel aux modèles de détection\nen fonction de la vitesse de simulation n, avec 8 vols en simultané")
plt.grid()
plt.savefig("time_per_predict_call_fr.png", dpi=300)
plt.close()







# english
x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
y = [5.169904708862305, 5.759028911590576, 6.791625499725342, 50.42728281021118, 64.31765651702881, 73.93254518508911, 90.11358308792114, 115.85217785835266, 161.1922812461853, 250.57810235023499]

y_factor = [512 / y[i] for i in range(len(x))]

# do the same code as below but plot y_factor on the same graph but with a right y axis with log scale 
# plt.plot(x, y, marker="o", markersize=4)
# plt.ylabel("Time (s)")
# plt.xlabel("Number of simultaneous aircraft (n)")
# plt.ylabel("Time (s)")
# plt.title("Execution time, to process in real time\n n flights of 512 ADS-B messages")
# plt.grid()
# plt.savefig("time_per_simultaneous_aircraft_en.png", dpi=300)
# plt.close()

fig, ax1 = plt.subplots()

# first plot
ax1.plot(x, y, marker="o", markersize=4, label="Execution time")
ax1.set_xlabel("Number of simultaneous aircraft (n)")
ax1.set_ylabel("Time (s)")
ax1.set_title("Execution time, to process in real time\n n flights of 512 ADS-B messages")
ax1.grid()

# second plot
ax2 = ax1.twinx()
ax2.plot(x, y_factor, marker="o", markersize=2, color="tab:red", linestyle="--", linewidth=1, label="Speedup factor")
ax2.set_ylabel("Speedup factor")

# set the y axis to log scale
ax2.set_yscale("log",base=2)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))

# show one legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc="upper left")
plt.savefig("time_per_simultaneous_aircraft_en.png", dpi=300)
plt.close()









x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
y = [47.66769361495972, 27.110658645629883, 15.508031606674194, 9.392610311508179, 5.985950708389282, 4.123213052749634, 3.1290125846862793, 2.7381110191345215, 2.3267195224761963, 2.2480316162109375]
c = [511, 256, 128, 64, 32, 16, 8, 4, 2, 1]


plt.plot(x, y, marker="o", markersize=4)
plt.xlabel("Simulation speed (x n)")
plt.ylabel("Time (s)")
plt.title("Execution time, for processing\n8 flights of 512 messages accelerated n times")
# show tick for max y and min y
plt.yticks(np.concatenate([np.arange(0, np.floor(max(y))+1, 10), [max(y), min(y)]]))
plt.grid()
plt.savefig("time_per_acc_en.png", dpi=300)
plt.close()

y = [y[i] / c[i] for i in range(len(x))]

plt.plot(x, y, marker="o", markersize=4)
# threshold = 1
plt.plot([x[0], x[-1]], [1, 1], color="red", linestyle="--", label="Real-time threshold")
plt.xlabel("Simulation speed (x n)")
plt.ylabel("Time (s)")
plt.title("Execution time of a call to detection models\ndepending on the simulation speed n, with 8 simultaneous flights")
plt.grid()
plt.savefig("time_per_predict_call_en.png", dpi=300)
plt.close()