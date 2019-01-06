import numpy as np
import matplotlib.pyplot as plt


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


def ponctual_differencies_hist(folder, asicID, trim):
    name = folder + asicID + "_ponctual_differencies_trim%s.txt"%trim
    data = np.loadtxt(name, delimiter=',')
    title = "Distribution of the hit count deviations between several runs"
    output = name[:-4] + "_hist.png"
    plot_hist(data, output, title)


def pixel_max_ponctual_differencies_hist(folder, asicID, trim):
    name = folder + asicID + "_pixels_max_ponctual_differencies_trim%s.txt"%trim
    data = np.loadtxt(name, delimiter=',')
    title = "Distribution of each pixel max hit count deviations between several runs"
    output = name[:-4] + "_hist.png"
    plot_hist(data, output, title)


def fitted_std_deviation_hist(folder, asicID, trim, first_fitting_type, param):
    name = folder + asicID + "_fitted_std_deviation_trim%s_%sFit_param%.1f.txt"%(trim, first_fitting_type, param)
    data = np.loadtxt(name, delimiter=',')
    title = "Distribution of each pixel fitted noise SD deviations between several runs"
    output = name[:-4] + "_hist.png"
    plot_hist(data, output, title)


def brut_std_deviation_hist(folder, asicID, trim):
    name = folder + asicID + "_brut_std_deviation_trim%s.txt"%trim
    data = np.loadtxt(name, delimiter=',')
    title = "Distribution of each pixel noise distribution rms deviations between several runs"
    output = name[:-4] + "_hist.png"
    plot_hist(data, output, title)


def brut_mean_deviation_hist(folder, asicID, trim):
    name = folder + asicID + "_brut_mean_deviation_trim%s.txt"%trim
    data = np.loadtxt(name, delimiter=',')
    title = "Distribution of each pixel noise distribution mean deviations between several runs"
    output = name[:-4] + "_hist.png"
    plot_hist(data, output, title)


def fitted_mean_deviation_hist(folder, asicID, trim, first_fitting_type, param):
    name = folder + asicID + "_fitted_mean_deviation_trim%s_%sFit_param%.1f.txt"%(trim, first_fitting_type, param)
    data = np.loadtxt(name, delimiter=',')
    title = "Distribution of each pixel noise distribution fitted mean deviations between two runs"
    output = name[:-4] + "_hist.png"
    plot_hist(data, output, title)


def brut_std_without_artifacts_deviation_hist(folder, asicID, trim, first_fitting_type, param):
    name = folder + asicID + "_std_without_artifacts_deviation_trim%s_%sFit_param%.1f.txt"%(trim, first_fitting_type, param)
    data = np.loadtxt(name, delimiter=',')
    title = "Distribution of each pixel noise distribution rms deviation between several runs\n after having removed the spike artifacts"
    output = name[:-4] + "_hist.png"
    plot_hist(data, output, title)











