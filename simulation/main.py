# Author: Giraudo Anthony                     
# 13 decembre 2018
# main.py

from functions import *



sdth = 4.65

list_clock_cycles = [[450]]

nb_pixel = 10000

nscan = 35



for clock_cycles in list_clock_cycles:
    folder = ""
    for clock_cycle in clock_cycles:
        folder += str(clock_cycle) + "_"
    folder += "sdth%.2f_%dpix" % (sdth, nb_pixel)

    print("[Status] Simulating noise distributions")
    runs(nb_pixel, clock_cycles, folder, nscan, sdth)

    print("[Status] Compiling measurements")
    addition(folder)

    print("[Status] Calculating brut mean and sd")
    brut_values(folder, nscan)

    print("[Status] Calculating fitted mean and sd")
    fitted_values(folder, nscan)

    print("[Status] Plotting")
    hists(folder, sdth)
    plot_pixels(1, folder, nb_pixel, nscan, sdth, condition="")

    plot_pixels(10, folder, nb_pixel, nscan, sdth, condition=("delta sd", (0.4, 10)), plot_folder="plots/sd")
