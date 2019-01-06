# Author: Giraudo Anthony                     
# 18 decembre 2018
# create_basic_files.py


import numpy as np
from analysis import fitting


def read_paths(configs):
    return [config["output_directory"] + "/" for config in configs]


def load_distributions(paths, asicID, trim):
    print("[Status] Loading noise distributions data")
    h = np.loadtxt(paths[0] + 'Pixel_%s_Trim%s.txt' %(asicID, trim), dtype=float, delimiter=',').reshape(1, -1, 256, 256)
    for path in paths[1:]:
        h = np.append(h,
                      np.loadtxt(path + 'Pixel_%s_Trim%s.txt' % (asicID, trim), dtype=float, delimiter=',').reshape(1, -1, 256, 256), axis=0)
    print("[Status] Loaded")
    return h


def load(folder, keyname, nfile):
    data = np.loadtxt(folder + keyname + "_measurement0.txt", delimiter=',').reshape(1, 256, 256)
    for i in range(1, nfile):
        data = np.append(data, np.loadtxt(folder + keyname + "_measurement%d.txt" %i, delimiter=',').reshape(1, 256, 256), axis=0)
    return data


def load_popt(folder, asicID, trim, first_fitting_type, param, id):
    high = np.loadtxt(folder + "%s_fitted_high_trim%s_%sFit_param%.1f_measurement%d.txt"%(asicID, trim, first_fitting_type, param, id), delimiter=',')
    mean = np.loadtxt(folder + "%s_fitted_mean_trim%s_%sFit_param%.1f_measurement%d.txt" % (asicID, trim, first_fitting_type, param, id), delimiter=',')
    std = np.loadtxt(folder + "%s_fitted_std_trim%s_%sFit_param%.1f_measurement%d.txt" % (asicID, trim, first_fitting_type, param, id), delimiter=',')
    return high, mean, std


def load_first_popt(folder, asicID, trim, first_fitting_type, id):
    high = np.loadtxt(folder + "%s_first_fitted_high_trim%s_%sFit_measurement%d.txt"%(asicID, trim, first_fitting_type, id), delimiter=',')
    mean = np.loadtxt(folder + "%s_first_fitted_mean_trim%s_%sFit_measurement%d.txt" % (asicID, trim, first_fitting_type, id), delimiter=',')
    std = np.loadtxt(folder + "%s_first_fitted_std_trim%s_%sFit_measurement%d.txt" % (asicID, trim, first_fitting_type, id), delimiter=',')
    return high, mean, std


def load_brut_mean_and_std(folder, asicID, trim, i):
    mean = np.loadtxt(folder + asicID + "_brut_mean_trim%s_measurement%d.txt" %(trim, i), delimiter=',')
    std = np.loadtxt(folder + asicID + "_brut_std_trim%s_measurement%d.txt" %(trim, i), delimiter=',')
    return mean, std

def load_filtre(folder, asicID, nfile, trim, first_fitting_type, param):
    print("[Status] Loading filtre")
    filtre = np.loadtxt(folder + "%s_artifact_filtre_trim%s_%sFit_param%.1f_measurement%d.txt" % (asicID, trim, first_fitting_type, param, 0), dtype=bool, delimiter=',').reshape(1, -1, 256, 256)
    for i in range(1, nfile):
        filtre = np.append(filtre,
                      np.loadtxt(folder + "%s_artifact_filtre_trim%s_%sFit_param%.1f_measurement%d.txt" % (asicID, trim, first_fitting_type, param, 0), dtype=bool, delimiter=',').reshape(1, -1, 256, 256), axis=0)
    print("[Status] Loaded")
    return filtre


def scan_points(config, trim):
    if trim == "0":
        indice = 0
    elif trim == "F":
        indice = 1

    min_dac = config['min_global_threshold'][indice]  # obtain the minimum value from the yaml file
    max_dac = config['max_global_threshold'][indice]  # obtain the maximum value from the yaml file
    dac_step = config['global_threshold_step'][indice]  # obtain the step size from the yaml file
    return np.arange(max_dac, min_dac, -dac_step)


def mean_and_rms(h, config, trim):

    data = np.expand_dims(h, axis=0)
    data = np.swapaxes(data, 1, 2)
    data = np.swapaxes(data, 2, 3)

    # nan to the DAC counts of the dead pixels in order to don't get a division by 0 in the mean calculation
    _maskA = np.where(np.sum(data, axis=-1) < 1e-1)
    data[_maskA] = 1e-8

    dac = np.zeros(np.shape(data), dtype=int)

    dac[0, :, :, :] = scan_points(config, trim)
    mean = np.average(dac, weights=data, axis=-1)
    # clip the masked pixels, put their mean to 0
    mean[_maskA] = 0.

    std = np.sqrt(np.average((dac - mean.reshape(1, 256, 256, 1)) ** 2, weights=data, axis=-1))
    # clip the masked pixels, put their std to 0
    std[_maskA] = 0.

    return mean[0], std[0]


def brut_mean_and_std(output_folder, configs, trim):

    paths = read_paths(configs)
    asicID = configs[0]["asic"]
    h = load_distributions(paths, asicID, trim)

    for i in range(h.shape[0]):
        mean, std = mean_and_rms(h[i], configs[i], trim)
        np.savetxt(output_folder + asicID + "_brut_std_trim%s_measurement%d.txt" %(trim, i), std, fmt='%.2f', delimiter=",")
        np.savetxt(output_folder + asicID + "_brut_mean_trim%s_measurement%d.txt" % (trim, i), mean, fmt='%.2f',
                   delimiter=",")


def artifactMask_and_fitted_high_mean_std(configs, output_folder, trim, first_fitting_type, param):
    paths = read_paths(configs)
    asicID = configs[0]["asic"]
    x = scan_points(configs[0], trim) # same for all the measurements

    h = load_distributions(paths, asicID, trim)
    h = np.swapaxes(h, 1, 2)
    h = np.swapaxes(h, 2, 3)  # (x, 256, 256, nscanpoints)

    for i, h_i in enumerate(h): #copy ?

        filtre_i = np.ones(h_i.shape)

        print("[Status] Fitting")

        fitted_high = np.zeros((256, 256))
        fitted_mean = np.zeros((256, 256))
        fitted_std = np.zeros((256, 256))

        first_fitted_high = np.zeros((256, 256))
        first_fitted_mean = np.zeros((256, 256))
        first_fitted_std = np.zeros((256, 256))

        for j in range(256):
            print("[Status] Fitting row %d/256"%j)
            for k in range(256):
                popt, filtre, first_popt = fitting.double_fitting(x, h_i[j, k], first_fitting_type, param)
                fitted_high[j, k] = popt[0]
                fitted_mean[j, k] = popt[1]
                fitted_std[j, k] = popt[2]

                first_fitted_high[j, k] = first_popt[0]
                first_fitted_mean[j, k] = first_popt[1]
                first_fitted_std[j, k] = first_popt[2]

                filtre_i[j, k] = filtre

        print("[Status] Fitted")

        np.savetxt(output_folder + "%s_fitted_high_trim%s_%sFit_param%.1f_measurement%d.txt"%(asicID, trim, first_fitting_type, param, i),
                       fitted_high, fmt='%.2f', delimiter=',')
        np.savetxt(output_folder + "%s_fitted_mean_trim%s_%sFit_param%.1f_measurement%d.txt" % (asicID, trim, first_fitting_type, param, i),
                   fitted_mean, fmt='%.2f', delimiter=',')
        np.savetxt(output_folder + "%s_fitted_std_trim%s_%sFit_param%.1f_measurement%d.txt" % (asicID, trim, first_fitting_type, param, i),
                   fitted_std, fmt='%.2f', delimiter=',')

        np.savetxt(output_folder + "%s_first_fitted_high_trim%s_%sFit_measurement%d.txt" % (asicID, trim, first_fitting_type, i),
                   first_fitted_high, fmt='%.2f', delimiter=',')
        np.savetxt(output_folder + "%s_first_fitted_mean_trim%s_%sFit_measurement%d.txt" % (asicID, trim, first_fitting_type, i),
                   first_fitted_mean, fmt='%.2f', delimiter=',')
        np.savetxt(output_folder + "%s_first_fitted_std_trim%s_%sFit_measurement%d.txt" % (asicID, trim, first_fitting_type, i),
                   first_fitted_std, fmt='%.2f', delimiter=',')

        filtre_i = np.swapaxes(filtre_i, 1, 2)
        filtre_i = np.swapaxes(filtre_i, 0, 1)
        np.savetxt(output_folder + "%s_artifact_filtre_trim%s_%sFit_param%.1f_measurement%d.txt" % (asicID, trim, first_fitting_type, param, i),
                   filtre_i.reshape(256 * len(x), 256), fmt='%d', delimiter=',')


def fitted_values(configs, output_folder, trim, first_fitting_type, param):
    asicID = configs[0]["asic"]

    x = scan_points(configs[0], trim)

    for k in range(len(configs)):
        high, mean, std = load_popt(output_folder, asicID, trim, first_fitting_type, param, k)

        fitted_data = np.zeros((256, 256, len(x))) #(256, 256, nscanpoints)
        for i in range(256):
            for j in range(256):
                fitted_data[i, j] = fitting.gaussian([high[i, j], mean[i, j], std[i, j]], x)

        fitted_data = np.swapaxes(fitted_data, 1, 2)
        fitted_data = np.swapaxes(fitted_data, 0, 1)  # (nscanpoints, 256, 256)
        np.savetxt(output_folder + asicID + "_fitted_values_trim%s_%sFit_param%.1f_measurement%d.txt"%(trim, first_fitting_type, param, k),
                   fitted_data.reshape(256 * len(x), 256), fmt='%.2f', delimiter=',')


def mean_std_without_artifacts(configs, output_folder, trim, first_fitting_type, param):
    asicID = configs[0]["asic"]
    nfile = len(configs)
    paths = read_paths(configs)
    x = scan_points(configs[0], trim)
    h = load_distributions(paths, asicID, trim)
    filtre = load_filtre(output_folder, asicID, nfile, trim, first_fitting_type, param)
    h[np.logical_not(filtre)] = 0.

    for i in range(nfile):
        mean, std = mean_and_rms(h[i], configs[i], trim)
        np.savetxt(output_folder + "%s_mean_without_artifacts_trim%s_%sFit_param%.1f_measurement%d.txt"%(asicID, trim, first_fitting_type, param, i), mean, fmt='%.2f',
                   delimiter=",")
        np.savetxt(output_folder + "%s_std_without_artifacts_trim%s_%sFit_param%.1f_measurement%d.txt" % (asicID, trim, first_fitting_type, param, i), std, fmt='%.2f',
                   delimiter=",")

"""
def max_layers_removed(output_folder, asicID, nfile, param=16.):
    print("[Status] Loading masks")
    layers_count = np.loadtxt(output_folder + asicID + "_artifact_mask_param%d_measurement%d.txt" % (int(param), 0),
                              dtype=int, delimiter=',').reshape(-1, 256, 256)
    for i in range(1, nfile):
        layers_count = np.add(layers_count, np.loadtxt(
            output_folder + asicID + "_artifact_mask_param%d_measurement%d.txt" % (int(param), i), dtype=int,
            delimiter=',').reshape(-1, 256, 256))
    print("[Status] Loaded")
    max_layers_removed = np.amax(layers_count, axis=0)  # shape : (256, 256)
    np.savetxt(output_folder + asicID + "_max_layers_removed_param%d.txt" % (int(param)), max_layers_removed, fmt='%d',
               delimiter=',')
"""

