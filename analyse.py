# To be runned like "python measurement_comparison.py output_folder measurements1.yaml measurements2.yaml etc."

import sys
from utils.configurations import dict_yaml
from analysis import create_basic_files
from analysis import compare_measurements
from analysis import scripts
from plotting import plot_deviation_distribution
from plotting import plot_pixel_measurements

first_fitting_type = "gaussian"
trim = "0"
param = 40.

# Get the configuration
output_folder = sys.argv[1]+'/'
configs = [dict_yaml(sys.argv[i]) for i in range(2, len(sys.argv))]
paths = [config["output_directory"]+"/" for config in configs]
asicID = configs[0]["asic"]

nfile = len(paths)
folder = output_folder


# basic files
print("[Status] brut mean and std")
create_basic_files.brut_mean_and_std(output_folder, configs, trim)

print("[Status] fitting")
create_basic_files.artifactMask_and_fitted_high_mean_std(configs, output_folder, trim, first_fitting_type, param)

print("[Status] fitted values")
create_basic_files.fitted_values(configs, output_folder, trim, first_fitting_type, param)

print("[Status] mean std rm")
create_basic_files.mean_std_without_artifacts(configs, output_folder, trim, first_fitting_type, param)


# comparison files
print("[Status] ponctual differencies")
compare_measurements.ponctual_differencies(folder, paths, asicID, trim)

print("[Status] max ponctual differencies")
compare_measurements.pixel_max_ponctual_differencies(folder, paths, asicID, trim)

print("[Status] brut sd dev")
compare_measurements.brut_std_deviation(folder, asicID, nfile, trim)

print("[Status] brut mean dev")
compare_measurements.brut_mean_deviation(folder, asicID, nfile, trim)

print("[Status] fitted sd dev")
compare_measurements.fitted_std_deviation(folder, asicID, nfile, trim, first_fitting_type, param)

print("[Status] fitted mean dev")
compare_measurements.fitted_mean_deviation(folder, asicID, nfile, trim, first_fitting_type, param)

print("[Status] brut sd rm dev")
compare_measurements.brut_std_without_artifacts_deviation(folder, asicID, nfile, trim, first_fitting_type, param)

print("[Status] brut mean rm dev")
compare_measurements.brut_mean_without_artifacts_deviation(folder, asicID, nfile, trim, first_fitting_type, param)

print("[Status] max fitted-brut dev")
compare_measurements.pixels_max_fitted_brut_deviations(folder, paths, asicID, trim, first_fitting_type, param)


print("[Status] hist and plots")
# hist
plot_deviation_distribution.ponctual_differencies_hist(folder, asicID, trim)

plot_deviation_distribution.pixel_max_ponctual_differencies_hist(folder, asicID, trim)

plot_deviation_distribution.fitted_std_deviation_hist(folder, asicID, trim, first_fitting_type, param)

plot_deviation_distribution.brut_std_deviation_hist(folder, asicID, trim)

plot_deviation_distribution.brut_mean_deviation_hist(folder, asicID, trim)

plot_deviation_distribution.fitted_mean_deviation_hist(folder, asicID, trim, first_fitting_type, param)

plot_deviation_distribution.brut_std_without_artifacts_deviation_hist(folder, asicID, trim, first_fitting_type, param)


# plot pixels
coords = [(19, 216)]
plot_pixel_measurements.plot_pixels(configs, output_folder, output_folder + "plots/", coords, trim, first_fitting_type, param, plot_first_fit=True, plot_fit=False)


#scripts.count_and_plot_artifacts(configs, folder, trim, first_fitting_type, param, 0, 10)

