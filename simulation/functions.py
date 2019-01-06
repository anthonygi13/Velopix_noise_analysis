# Author: Giraudo Anthony                     
# 13 decembre 2018
# functions.py


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def clock_target(folder):
    clock_cycles = (np.loadtxt(folder + "/clock_cycles.txt", delimiter=',')).flatten()
    return np.sum(clock_cycles)


def high(clock_cycle):
    return 45 * (clock_cycle / 500.)


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.system("mkdir " + folder_name)


def relative_error(y):
    res = 1 / np.sqrt(np.array(y))
    return res


def absolute_error(y):
    res = y * relative_error(y)
    res[y < 1.] = 1. #TODO
    return res


def x_values(nscan):
    return np.arange(-int(nscan/2), int(nscan/2) + 1, 1)


def gaussian(p, x):
    """
    :param p: np.array([high, mean, sd])
    :param x: scalar or np.array
    :return:
    """
    return p[0] * np.exp(-(x-p[1])**2/(2*p[2]**2))


def residual(p, x, y, error):
    r = (y - gaussian(p, x)) / error
    return r


def fitting(x, y, p0, error):
    popt = least_squares(residual, p0, args=(x, y, error))
    return popt["x"]


def cut_gaussian(p, x):
    g = gaussian(p, x)
    g[g > 63] = 63
    return g


def base_files(nb_pixel, clock_cycle, folder, nscan, id, sdth):
    x = x_values(nscan)

    theoretical = np.array(nb_pixel * [gaussian(np.array([high(clock_cycle), 0, sdth]), x)])
    theoretical[theoretical <= 0.] = 0.

    simulated = np.array([np.random.poisson(theoretical_value, len(theoretical)) for theoretical_value in theoretical[0]])
    simulated = simulated.T
    simulated[simulated>63] = 63

    np.savetxt(folder + "/theoretical_values%d.txt"%id, theoretical, fmt='%.5f', delimiter=',')
    np.savetxt(folder + "/simulation_values%d.txt"%id, simulated, fmt='%d', delimiter=',')


def runs(nb_pixel, clock_cycles, folder, nscan, sdth):
    create_folder(folder)
    np.savetxt(folder + "/clock_cycles.txt", clock_cycles, fmt='%d')
    for id, clock_cycle in enumerate(clock_cycles):
        base_files(nb_pixel, clock_cycle, folder, nscan, id, sdth)


def addition(folder):
    clock_cycles = np.array([(np.loadtxt(folder + "/clock_cycles.txt", delimiter=','))]).flatten()

    h = np.loadtxt(folder + "/simulation_values0.txt", delimiter=',')
    h = h.reshape(1, h.shape[0], h.shape[1])
    h[h>=63.] = np.nan

    for id, clock_cycle in enumerate(clock_cycles[1:]):
        h_i = np.loadtxt(folder + "/simulation_values%d.txt"%(id+1), delimiter=',')
        h_i = h_i.reshape(1, h_i.shape[0], h_i.shape[1])
        h_i[h_i >= 63.] = np.nan
        h = np.append(h, h_i, axis=0)

    filtre = (np.isnan(h))
    h[filtre] = 0.

    new_clock_cycle = np.resize(clock_cycles, h.T.shape).T
    new_clock_cycle[filtre] = 0.
    new_clock_cycle = np.sum(new_clock_cycle, axis=0)

    h = np.einsum('kij->ij', h)
    error = absolute_error(h)

    h = h * clock_target(folder) / new_clock_cycle # nan if all saturated

    error = error * clock_target(folder) / new_clock_cycle
    error[np.isnan(h)] = np.nan

    filtre = np.logical_not(np.isnan(h)) # vire les point qui vallent 0, ils ont une erreur nulle dans le fitting ça fait des infinis
    # et vire les nan qui correspondent aux saturations

    np.savetxt(folder + "/compiled_measurements.txt", h, fmt='%.5f', delimiter=',')
    np.savetxt(folder + "/compiled_absolute_errors.txt", error, fmt='%.5f', delimiter=',')
    np.savetxt(folder + "/filtre.txt", filtre, fmt='%d', delimiter=',')



def brut_values(folder, nscan):
    filtre = np.loadtxt(folder + "/filtre.txt", dtype=bool, delimiter=',')
    h = np.loadtxt(folder + "/compiled_measurements.txt", delimiter=',')
    x_array = np.resize(x_values(nscan), h.shape)

    h[np.logical_not(filtre)] = 0.

    brut_mean = np.average(x_array, weights=h, axis=-1)
    brut_sd = np.sqrt(np.average(((x_array.T - brut_mean).T)**2, weights=h, axis=-1))

    np.savetxt(folder + "/brut_sd.txt", brut_sd, fmt='%.5f', delimiter=',')
    np.savetxt(folder + "/brut_mean.txt", brut_mean, fmt='%.5f', delimiter=',')



def fitted_values(folder, nscan):
    compiled_measurements = np.loadtxt(folder + "/compiled_measurements.txt", delimiter=',')
    compiled_absolute_errors = np.loadtxt(folder + "/compiled_absolute_errors.txt", delimiter=',')
    fitted_sd = np.zeros(compiled_measurements.shape[0])
    fitted_high = np.zeros(compiled_measurements.shape[0])
    fitted_mean = np.zeros(compiled_measurements.shape[0])
    brut_mean = np.loadtxt(folder + "/brut_mean.txt", delimiter=',')
    brut_sd = np.loadtxt(folder + "/brut_sd.txt", delimiter=',')
    filtre = np.loadtxt(folder + "/filtre.txt", dtype = bool, delimiter=',')

    x = x_values(nscan)

    for i in range(len(compiled_measurements)):
        if i % 2500 == 0:
            print("[Status] %d pourcents"%(int(i/compiled_measurements.shape[0]*100)))

        error = compiled_absolute_errors[i][filtre[i]]
        #error = np.ones(compiled_absolute_errors[i][filtre[i]].shape)

        p0 = [np.amax(compiled_measurements[i][filtre[i]]), brut_mean[i], brut_sd[i]]

        popt = fitting(x[filtre[i]], compiled_measurements[i][filtre[i]], p0, error)
        fitted_sd[i] = popt[2]
        fitted_high[i] = popt[0]
        fitted_mean[i] = popt[1]
    np.savetxt(folder + "/fitted_sd.txt", fitted_sd, fmt='%.5f', delimiter=',')
    np.savetxt(folder + "/fitted_high.txt", fitted_high, fmt='%.5f', delimiter=',')
    np.savetxt(folder + "/fitted_mean.txt", fitted_mean, fmt='%.5f', delimiter=',')



def plot_hist(data, output, title=""):
    bin_size = np.amax([0.1, abs(int((np.amin(data))) - int(np.amax(data))) / 35])
    edges = [i*bin_size for i in range(int((np.amin(data)-1)/bin_size), int((np.amax(data)+1)/bin_size))]
    plt.hist(data.flatten(), bins=edges, orientation="vertical")
    plt.axis([int(np.amin(data)-1), int(np.amax(data)+1),0.9,1000000])
    plt.yscale('log', nonposy='clip')       # set the y-axis to log scale
    plt.xticks(np.arange(int(np.amin(data)-1),int(np.amax(data)+1),abs((int(np.amin(data)-1)-int((np.amax(data)+1))))/10.))
    plt.title(title)
    mytext = "Mean : %.2f, SD : %.2f"%(np.mean(data), np.std(data))
    plt.text(0, 10**5, mytext, fontsize=12, horizontalalignment='left', verticalalignment='center', color='black')
    plt.legend()
    plt.savefig(output, bbox_inches='tight', format='png')
    plt.close()



def hists(folder, sdth):
    fitted_sd = np.loadtxt(folder + "/fitted_sd.txt", delimiter=',')
    fitted_high = np.loadtxt(folder + "/fitted_high.txt", delimiter=',')
    fitted_mean = np.loadtxt(folder + "/fitted_mean.txt", delimiter=',')
    brut_mean = np.loadtxt(folder + "/brut_mean.txt", delimiter=',')
    brut_sd = np.loadtxt(folder + "/brut_sd.txt", delimiter=',')

    plot_hist(fitted_sd - sdth, folder + "/delta_sd_fitted.png", title="Distribution of the difference between the theoretical sd and the fitted sd")
    plot_hist(fitted_high - high(clock_target(folder)), folder + "/delta_high_fitted.png",
              title="Distribution of the difference between the theoretical high and the fitted high")
    plot_hist(fitted_mean, folder + "/delta_mean_fitted.png",
              title="Distribution of the difference between the theoretical mean and the fitted mean")
    plot_hist(brut_mean, folder + "/delta_mean_brut.png", title="Distribution of the difference between the theoretical mean and the brut mean")
    plot_hist(brut_sd - sdth, folder + "/delta_sd_brut.png", title="Distribution of the difference between the theoretical sd and the brut sd")


def plot_noise_distribution(pixels, folder, nscan, sdth, plot_folder="plots"):
    h = np.loadtxt(folder + "/compiled_measurements.txt", delimiter=',')
    filtre = np.loadtxt(folder + "/filtre.txt", dtype=bool, delimiter=',')
    fitted_sd = np.loadtxt(folder + "/fitted_sd.txt", delimiter=',')
    fitted_high = np.loadtxt(folder + "/fitted_high.txt", delimiter=',')
    fitted_mean = np.loadtxt(folder + "/fitted_mean.txt", delimiter=',')
    brut_mean = np.loadtxt(folder + "/brut_mean.txt", delimiter=',')
    brut_sd = np.loadtxt(folder + "/brut_sd.txt", delimiter=',')

    x = x_values(nscan)

    create_folder(folder + "/" + plot_folder)

    for i, pixel in enumerate(pixels):
        print("[Status] Plotting advancement : %d/%d"%(i, len(pixels)))

        plt.plot(x[filtre[pixel]], h[pixel][filtre[pixel]], marker='o', markerfacecolor='blue', color='blue')

        x_cont = np.linspace(x[0], x[-1], 1000)

        plt.plot(x_cont, gaussian(np.array([fitted_high[pixel], fitted_mean[pixel], fitted_sd[pixel]]), x_cont), color='green')

        plt.plot(x_cont, gaussian(np.array([high(clock_target(folder)), 0, sdth]), x_cont),
                 color='red')


        ytext = -10


        mytext = "theoretical noise distribution, High = %.1f, Mean = %.2f, SD = %.2f"%(high(clock_target(folder)), 0., sdth)
        plt.text(x[0], (high(clock_target(folder)) / 70) * (ytext - 3 * 2), mytext, fontsize=12,
                 horizontalalignment='left', verticalalignment='center', color='red')

        mytext = "Simultated noise distribution, Mean = %.2f, SD = %.2f"%(brut_mean[pixel], brut_sd[pixel])
        plt.text(x[0], (high(clock_target(folder)) / 70) * (ytext - 3 * 3), mytext, fontsize=12,
                 horizontalalignment='left', verticalalignment='center', color='blue')

        mytext = "Fitted noise distribution, High = %.1f, Mean = %.2f, SD = %.2f"%(fitted_high[pixel], fitted_mean[pixel], fitted_sd[pixel])
        plt.text(x[0], (high(clock_target(folder)) / 70) * (ytext - 3 * 4), mytext, fontsize=12,
                 horizontalalignment='left', verticalalignment='center', color='green')

        mytext = "Pixel %d" %(pixel)
        plt.text(x[0], (high(clock_target(folder)) / 70) * (ytext - 3), mytext, fontsize=12, horizontalalignment='left',
                 verticalalignment='center', color='black')

        plt.axis([x[0], x[-1], 0, high(clock_target(folder)) * (1 + 1/4)])
        plt.xlabel("DAC Threshold", fontsize=20)
        plt.ylabel("Number of Hits", fontsize=20)
        plt.legend()

        plt.savefig(folder + "/" + plot_folder + "/plot_%d.png"%pixel, bbox_inches='tight', format='png')
        plt.close()


def plot_pixels(number, folder, nb_pixel, nscan, sdth, condition="", plot_folder="plots"):
    """
    :param number:
    :param folder:
    :param nb_pixel:
    :param condition: ("key word", bounds)
    :return:
    """

    if condition == "":
        filtre = np.ones(nb_pixel, dtype=bool)

    elif condition[0] == "delta sd":
        fitted_sd = np.loadtxt(folder + "/fitted_sd.txt", delimiter=',')
        filtre = (abs(fitted_sd - sdth) >= condition[1][0]) & (abs(fitted_sd - sdth) <= condition[1][1])

    elif condition[0] == "delta high":
        fitted_high = np.loadtxt(folder + "/fitted_high.txt", delimiter=',')
        filtre = (abs(fitted_high - high(clock_target(folder))) >= condition[1][0]) & (abs(fitted_high - high(clock_target(folder))) <= condition[1][1])

    elif condition[0] == "delta mean":
        fitted_mean = np.loadtxt(folder + "/fitted_mean.txt", delimiter=',')
        filtre = (abs(fitted_mean) >= condition[1][0]) & (abs(fitted_mean) <= condition[1][1])

    else:
        raise ValueError("condition ininterpretable")

    to_plot = np.argwhere(filtre)
    np.random.shuffle(to_plot)
    plot_noise_distribution(to_plot[:number].flatten(), folder, nscan, sdth, plot_folder)



def plot(x, y, output, title=""):
    plt.plot(x, y)
    plt.title(title)
    plt.legend()
    plt.savefig(output, bbox_inches='tight', format='png')
    plt.close()
