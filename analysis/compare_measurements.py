import numpy as np
from analysis import create_basic_files


def calculate_deviation(data):
    if data.shape[0] == 2:
        return data[0] - data[1]
    else:
        mean = np.average(data, axis=0)
        return np.sqrt(np.average((data - np.resize(mean, np.shape(data))) ** 2, axis=0))


def delta(file_1, file_2, output_file):
  data1 = np.loadtxt(file_1, delimiter=',')
  data2 = np.loadtxt(file_2, delimiter=',')
  delta = data1 - data2
  np.savetxt(output_file, delta, fmt='%.2f', delimiter=',')



def ponctual_differencies(folder, paths, asicID, trim):
    data = create_basic_files.load_distributions(paths, asicID, trim)
    deviation = calculate_deviation(data)
    np.savetxt(folder + asicID + "_ponctual_differencies_trim%s.txt"%trim, deviation.reshape(data.shape[1]*256, 256), fmt='%.2f', delimiter=',')


def pixel_max_ponctual_differencies(folder, paths, asicID, trim):
    data = create_basic_files.load_distributions(paths, asicID, trim)
    deviation = calculate_deviation(data)
    max_deviation = np.amax(abs(deviation), axis=0)
    np.savetxt(folder + asicID + "_pixels_max_ponctual_differencies_trim%s.txt"%trim, max_deviation, fmt='%.2f',
               delimiter=',')


def fitted_std_deviation(folder, asicID, nfile, trim, first_fitting_type, param):
    data = create_basic_files.load(folder, "%s_fitted_std_trim%s_%sFit_param%.1f"%(asicID, trim, first_fitting_type, param), nfile)
    deviation = calculate_deviation(data)
    np.savetxt(folder + asicID + "_fitted_std_deviation_trim%s_%sFit_param%.1f.txt"%(trim, first_fitting_type, param), deviation, fmt='%.2f', delimiter=',')


def brut_std_deviation(folder, asicID, nfile, trim):
    data = create_basic_files.load(folder, "%s_brut_std_trim%s"%(asicID, trim), nfile)
    deviation = calculate_deviation(data)
    np.savetxt(folder + asicID + "_brut_std_deviation_trim%s.txt"%trim, deviation, fmt='%.2f', delimiter=',')


def brut_std_without_artifacts_deviation(folder, asicID, nfile, trim, first_fitting_type, param):
    data = create_basic_files.load(folder, "%s_std_without_artifacts_trim%s_%sFit_param%.1f"%(asicID, trim, first_fitting_type, param), nfile)
    deviation = calculate_deviation(data)
    np.savetxt(folder + asicID + "_std_without_artifacts_deviation_trim%s_%sFit_param%.1f.txt"%(trim, first_fitting_type, param), deviation, fmt='%.2f', delimiter=',')


def brut_mean_without_artifacts_deviation(folder, asicID, nfile, trim, first_fitting_type, param):
    data = create_basic_files.load(folder, "%s_mean_without_artifacts_trim%s_%sFit_param%.1f" % (asicID, trim, first_fitting_type, param), nfile)
    deviation = calculate_deviation(data)
    np.savetxt(folder + asicID + "_mean_without_artifacts_deviation_trim%s_%sFit_param%.1f.txt" % (trim, first_fitting_type, param), deviation, fmt='%.2f', delimiter=',')


def brut_mean_deviation(folder, asicID, nfile, trim):
    data = create_basic_files.load(folder, "%s_brut_mean_trim%s"%(asicID, trim), nfile)
    deviation = calculate_deviation(data)
    np.savetxt(folder + asicID + "_brut_mean_deviation_trim%s.txt"%trim, deviation, fmt='%.2f', delimiter=',')


def fitted_mean_deviation(folder, asicID, nfile, trim, first_fitting_type, param):
    data = create_basic_files.load(folder, "%s_fitted_mean_trim%s_%sFit_param%.1f"%(asicID, trim, first_fitting_type, param), nfile)
    deviation = calculate_deviation(data)
    np.savetxt(folder + asicID + "_fitted_mean_deviation_trim%s_%sFit_param%.1f.txt"%(trim, first_fitting_type, param), deviation, fmt='%.2f', delimiter=',')


def pixels_max_fitted_brut_deviations(folder, paths, asicID, trim, first_fitting_type, param):
    for i, path in enumerate(paths):
        h = np.loadtxt(path + 'Pixel_%s_Trim%s.txt'%(asicID, trim), delimiter=',').reshape(-1, 256, 256)
        fitted_data = np.loadtxt(folder + asicID + "_fitted_values_trim%s_%sFit_param%.1f_measurement%d.txt"%(trim, first_fitting_type, param, i), delimiter=',').reshape(-1, 256, 256)
        deviation = h - fitted_data
        max_deviation = np.amax(abs(deviation), axis=0)
        np.savetxt(folder + asicID + "_pixels_max_fitted_brut_deviations_trim%s_%sFit_param%.1f_measurement%d.txt"%(trim, first_fitting_type, param, i), max_deviation, fmt='%.1f', delimiter=',')

