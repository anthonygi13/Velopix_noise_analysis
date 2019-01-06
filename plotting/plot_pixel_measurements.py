import numpy as np
import matplotlib.pyplot as plt
from analysis import create_basic_files
from analysis import fitting
import os


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.system("mkdir " + folder_name)


def clean(x_values, y_values): # remove zeros at the begining and the end of the data, for a plot easier to read
    x = np.copy(x_values)
    y = np.copy(y_values)
    for j, value in enumerate(y):
        if (value != 0):
            indice = j
            break
    else:
        return x, y
    y = y[indice:]
    x = x[indice:]
    for k in range(1, len(y) + 1):
        if (y[-k] != 0):
            indice = len(y) - k
            break
    if (indice != len(y) - 1):
        y = y[:indice + 1]
        x = x[:indice + 1]
    return x, y


def colors():
    return ['blue', 'red', 'green', 'darkorange', 'black', 'purple', 'maroon', 'fuchsia', 'gold']


def colors_fit():
    return ['dodgerblue', 'salmon', 'chartreuse', 'orange', 'grey', 'mediumorchid', 'firebrick', 'violet', 'yellow']


def plot_pixels(configs, folder, output_folder, coords, trim, first_fitting_type, param, plot_first_fit=True, plot_fit=True, title=""):
    create_folder(output_folder)

    paths = [config["output_directory"] + "/" for config in configs]
    asicID = configs[0]["asic"]
    nfile = len(paths)

    h = create_basic_files.load_distributions(paths, asicID, trim)

    std = create_basic_files.load(folder, "%s_brut_std_trim%s" %(asicID, trim), nfile)
    mean = create_basic_files.load(folder, "%s_brut_mean_trim%s" %(asicID, trim), nfile)

    if plot_fit:
        fitted_high = create_basic_files.load(folder, "%s_fitted_high_trim%s_%sFit_param%.1f"%(asicID, trim, first_fitting_type, param), nfile)
        fitted_mean = create_basic_files.load(folder, "%s_fitted_mean_trim%s_%sFit_param%.1f"%(asicID, trim, first_fitting_type, param), nfile)
        fitted_std = create_basic_files.load(folder, "%s_fitted_std_trim%s_%sFit_param%.1f"%(asicID, trim, first_fitting_type, param), nfile)

    if plot_first_fit:
        first_fitted_high = create_basic_files.load(folder, "%s_first_fitted_high_trim%s_%sFit"%(asicID, trim, first_fitting_type), nfile)
        first_fitted_mean = create_basic_files.load(folder, "%s_first_fitted_mean_trim%s_%sFit" % (asicID, trim, first_fitting_type), nfile)
        first_fitted_std = create_basic_files.load(folder, "%s_first_fitted_std_trim%s_%sFit" % (asicID, trim, first_fitting_type), nfile)

    ytext = -10

    x_values = create_basic_files.scan_points(configs[0], trim)

    compte = 0

    for xcoor, ycoor in coords:

        x_min_list = []
        x_max_list = []
        y_max_list = []

        compte += 1
        print("[Status] Avancement : %d/%d" % (compte, len(coords)))

        theFig = plt.figure(figsize=(6, 6), facecolor='white')

        for i in range(nfile):

            y_values = h[i, :, xcoor, ycoor]
            x, y = clean(x_values, y_values)

            x_min_list += [x[-1]]
            x_max_list += [x[0]]
            y_max_list.append(np.amax(y))

            #plot first fitting
            if plot_first_fit:
                x_fit = np.linspace(x[-1], x[0], 1000)
                if first_fitting_type == "cut":
                    plt.plot(x_fit, fitting.cutGaussian([first_fitted_high[i, xcoor, ycoor], first_fitted_mean[i, xcoor, ycoor], first_fitted_std[i, xcoor, ycoor]], x_fit), linestyle='--', color=colors_fit()[i])#, label="First fitting curve")
                elif first_fitting_type == "gaussian":
                    plt.plot(x_fit, fitting.gaussian([first_fitted_high[i, xcoor, ycoor], first_fitted_mean[i, xcoor, ycoor], first_fitted_std[i, xcoor, ycoor]], x_fit), linestyle='--', color=colors_fit()[i])#, label="First fitting curve")

            # plot fitting
            if plot_fit:
                x_fit = np.linspace(x[-1], x[0], 1000)
                plt.plot(x_fit, fitting.gaussian([fitted_high[i, xcoor, ycoor], fitted_mean[i, xcoor, ycoor], fitted_std[i, xcoor, ycoor]], x_fit), color=colors_fit()[i])

            # plot data
            plt.plot(x, y, marker='o', markerfacecolor=colors()[i], color=colors()[i])
            mytext = "Scan %d, Mean = %.1f, SD = %.2f"% (i, mean[i, xcoor, ycoor], std[i, xcoor, ycoor])
            if plot_fit: mytext += "\nFitted high = %.2f, fitted mean = %.2f, fitted SD = %.2f" % (fitted_high[i, xcoor, ycoor], fitted_mean[i, xcoor, ycoor], fitted_std[i, xcoor, ycoor])
            plt.text(np.amin(x_min_list), (np.amax(y_max_list) / 70.) * (ytext - 6 * (i + 1)), mytext, fontsize=12,
                     horizontalalignment='left', verticalalignment='center', color=colors()[i])

        # Visualise
        mytext = "Pixel [%d,%d]" % (xcoor, ycoor)
        plt.text(np.amin(x_min_list), (np.amax(y_max_list) / 70.) * ytext, mytext, fontsize=12, horizontalalignment='left',
                 verticalalignment='center', color='black')

        plt.axis([np.amin(x_min_list), np.amax(x_max_list), 0, np.amax(y_max_list) + 10])
        plt.xlabel("DAC Threshold", fontsize=20)
        plt.ylabel("Number of Hits", fontsize=20)
        plt.title(title)
        plt.legend()

        # Save
        savefile = "_plot_pixel_%d_%d" % (xcoor, ycoor)
        plt.savefig(output_folder + asicID + savefile + ".png", bbox_inches='tight', format='png')
        plt.close()





