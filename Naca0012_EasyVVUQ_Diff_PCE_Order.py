# =============================================================
# Project: Uncertainty Quantification in CFD Simulations via PCE Methods
# File: Naca0012_EasyVVUQ_Diff_PCE_Order.py
# Author: Luca A. Mattiocco
# Email: luca.mattiocco@cranfield.ac.uk or luca.mattiocco@orange.fr
# Date: 06/08/24
# Version: 1.1
# =============================================================
# Description:
# This script facilitates uncertainty quantification for computational
# fluid dynamics (CFD) simulations using Polynomial Chaos Expansion (PCE)
# methods. It is designed to interface with STAR-CCM+ and can be adapted
# for different CFD scenarios. The script handles key parameters such as
# velocity, temperature, and pressure, incorporating probabilistic variations
# through normal distributions but can be tuned as user desires.
#
# Note:
# This script is part of a master's thesis project. While every effort has
# been made to ensure accuracy and functionality, it is provided "as is".
# Feel free to use, modify, and share, but please give credit where it's due!
# =============================================================

# Standard Library Imports
import os
import time
import pickle
import json
from shutil import rmtree

# Third-Party Library Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# EasyVVUQ & Chaospy Libraries
import easyvvuq as uq
import chaospy as cp

# Custom Project Libraries
from easyvvuq.actions import ExecuteLocal

# IPython Display Tools
from IPython.display import display


def remove_dir(path):
    try:
        if os.path.exists(path):
            rmtree(path)
    except PermissionError as e:
        print(f"PermissionError: {e}")
        print("Ensure no files are open and retry.")
        return

# Load configuration from JSON file
def load_config(json_file):
    with open(json_file, 'r') as file:
        config = json.load(file)
    return config

config = load_config('config.json')


# define parameters of the Starccm model
def define_params():
    return {
        "velocity": {"type": "float", "default": 51.04},
        "static_temp": {"type": "float", "default": 288.15},
        "ref_pressure": {"type": "float", "default": 101325},
        "Ru": {"type": "float", "default": 8.314},  # Universal gas constant
        "M_air": {"type": "float", "default": 0.0289647},  # Molar mass of air in kg/mol
        "R_air": {"type": "float", "default": 287.05}, #specific gas constant in  J/(kg·K)
        "gamma": {"type": "float", "default": 1.4},
        "power" : {"type": "float", "default": 0.5},
    }


# define varying quantities
def define_vary():
    vary_all = {
        "velocity": cp.Normal(51.04, 0.25),
        "static_temp": cp.Normal(288.15, 0.75 ),
        "ref_pressure": cp.Normal(101325, 50)
    }
    vary_2_1 =  {
        "velocity": cp.Normal(51.04, 2),
        "static_temp": cp.Normal(288.15, 2),
    }
    vary_2_2 =  {
        "static_temp": cp.Normal(288.15, 2),
        "ref_pressure": cp.Normal(101325, 1000)
    }
    vary_2_3 = {
        "velocity": cp.Normal(51.04, 2),
        "ref_pressure": cp.Normal(101325, 1000)

    }
    return vary_all


"""
Executes a Polynomial Chaos Expansion (PCE) campaign for a StarCCM+ model simulation.

This function sets up the campaign environment, runs the simulation, and analyzes the results.
It includes several phases: setting up the directory and model, defining parameters, running 
the simulation, and analyzing and saving the results.

Args:
----
pce_order (int): The order of the PCE expansion.
use_files (bool): Indicates whether files should be used for inputs/outputs. Default is True.

Returns:
-------
tuple: A tuple containing the following elements:
    results_df (pd.DataFrame): DataFrame containing model inputs and outputs.
    results (dict): Results of the PCE analysis.
    times (np.ndarray): Array containing the elapsed time for different computation phases.
    pce_order (int): The PCE order used.
    count (int): Number of PCE samples.
"""
def run_pce_campaign(pce_order, use_files=True):

    times = np.zeros(7)

    time_start = time.time()
    time_start_whole = time_start

    # Get the current working directory of the script, specify the input filename, and specify the output filename
    work_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the path to the template file
    template = os.path.join(work_dir, config['template_file'])

    # Read and display the content of the template file
    with open(template, 'r') as file:
        content = file.read()

    print("Contents of the template file:")
    print(content)

    campaign_work_dir = os.path.join(work_dir, f"{config['campaign_work_dir']}{pce_order}")

    # Clear the target campaign directory with error handling
    remove_dir(campaign_work_dir)
    os.makedirs(campaign_work_dir)

    # Set up a fresh campaign called 
    db_location = config['db_location_prefix'] + campaign_work_dir + f"/campaign_order_{pce_order}.db"
    campaign = uq.Campaign(
        name=f"{config['campaign_name']}_Order_{pce_order}",
        db_location=db_location,
        work_dir=campaign_work_dir
    )

    # Define parameter space
    params = define_params()

    # Create a decoder for PCE test app
    if use_files:
        class CustomDecoder:
            def __init__(self, target_filename=None, output_columns=None):
                self.target_filename = target_filename
                self.output_columns = output_columns

            def parse_sim_output(self, run_info={}):
                results = {}
                for key, file_path in self.target_filename.items():
                    try:
                        print(f"Reading file: {file_path}")
                        df = pd.read_csv(file_path)
                        if self.output_columns[key] in df.columns:
                            last_value = df[self.output_columns[key]].iloc[-1]
                            results[key] = last_value
                        else:
                            print(f"Column '{self.output_columns[key]}' not found in {file_path}")
                            results[key] = None
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        results[key] = None

                print(f"Results: {results}")
                return results

        # Define the target filenames and columns
        target_filename = config['output_files']
        output_columns = config['output_columns']

        encoder = uq.encoders.JinjaEncoder(template_fname=config['template_file'], target_filename=config['target_filename'])
        decoder = CustomDecoder(target_filename=target_filename, output_columns=output_columns)
        execute = ExecuteLocal(f"{config['starccm_executable']} -batch {config['target_filename']} {config['starccm_simulation_file']}")

        # Define the actions
        actions = uq.actions.Actions(
            uq.actions.CreateRunDirectory(root=os.getcwd()),  # Create run directories
            uq.actions.Encode(encoder),  # Encode the parameters
            execute,  # Execute the simulation
            uq.actions.Decode(decoder)  # Decode the results
        )
    else:
        print("Error in running STAR-CCM+")

    # Add the app to the campaign with the actions
    campaign.add_app(name=config['App_name'], params=params, actions=actions)

    time_end = time.time()
    times[1] = time_end-time_start
    print('Time for phase 1 = %.3f' % (times[1]))

    time_start = time.time()

    # Associate a sampler with the campaign
    campaign.set_sampler(uq.sampling.PCESampler(vary=define_vary(), polynomial_order=pce_order))
    campaign.draw_samples()
    print('Number of samples = %s' % campaign.get_active_sampler().count)

    time_end = time.time()
    times[2] = time_end-time_start
    print('Time for phase 2 = %.3f' % (times[2]))

    time_start = time.time()
    # Perform the actions
    print(f'Starting campaign for polynomial order {pce_order}...')
    try:
        campaign.execute(sequential=True).collate(progress_bar=True)
    except Exception as e:
        print(e)
    print(f'Finished campaign for polynomial order {pce_order}')

    time_end = time.time()
    times[3] = time_end-time_start
    print('Time for phase 3 = %.3f' % (times[3]))

    time_start = time.time()
    # Collate the results
    results_df = campaign.get_collation_result()

    time_end = time.time()
    times[4] = time_end-time_start
    print('Time for phase 4 = %.3f' % (times[4]))

    time_start = time.time()

    # Post-processing analysis
    pce_analysis = uq.analysis.PCEAnalysis(sampler=campaign.get_active_sampler(), qoi_cols=["Cl", "Cd", "L/D","L","D"])
    results = pce_analysis.analyse(results_df)

    time_end = time.time()
    times[5] = time_end-time_start
    print('Time for phase 5 = %.3f' % (times[5]))

    time_start = time.time()
    
    # Save the results
    pickle_file_path = config['pickle_file']
    pickle.dump(results, open(pickle_file_path, 'bw'))
    time_end = time.time()
    times[6] = time_end-time_start
    print('Time for phase 6 = %.3f' % (times[6]))

    times[0] = time_end - time_start_whole

    return results_df, results, times, pce_order, campaign.get_active_sampler().count

"""
Computes statistical moments (mean, variance, standard deviation) and Sobol indices 
for specified quantities of interest (QoI) using the provided results object.

This function iterates over each QoI provided in `qoi_cols`, calculates its statistical moments 
(mean, variance, standard deviation), and retrieves its first-order, second-order, and total Sobol 
indices. The results are then saved to a CSV file.

Args:
----
results (object): An object with a `describe` method for statistical analysis and methods for 
                    retrieving Sobol indices (`sobols_first`, `sobols_second`, `sobols_total`).
qoi_cols (list of str): A list of names of the quantities of interest for which statistics are computed.

Returns:
-------
None: The function saves the computed results to CSV files.
"""
def extract_results(results, qoi_cols):
    for qoi in qoi_cols:
        moments = {
            'Mean': results.describe(qoi=qoi, statistic="mean"),
            'Variance': results.describe(qoi=qoi, statistic="var"),
            'Standard Deviation': results.describe(qoi=qoi, statistic="std")
        }

        sobol_first = results.sobols_first()[qoi]
        sobol_second = results.sobols_second()[qoi]
        sobol_total = results.sobols_total()[qoi]

        sobol_indices = {
            f'First order Sobol index ({param})': sobol_first[param] for param in sobol_first
        }
        sobol_indices.update({
            f'Second order Sobol index ({param})': sobol_second[param] for param in sobol_second
        })
        sobol_indices.update({
            f'Total Sobol index ({param})': sobol_total[param] for param in sobol_total
        })

        save_results_to_csv(moments, sobol_indices, qoi, pce_order)

"""
Saves the statistical moments and Sobol indices to a CSV file.

Inputs:
    statistical_moments: A dictionary containing the mean, variance, and standard deviation.
    sobol_indices: A dictionary containing the Sobol indices.
    qoi_name: The name of the quantity of interest.
    order: The order of the analysis.

Outputs:
    Saves a CSV file containing the results.
"""
def save_results_to_csv(statistical_moments, sobol_indices, qoi_name, order):

    try:
        data = {
            'Metric': list(statistical_moments.keys()) + list(sobol_indices.keys()),
            'Value': list(statistical_moments.values()) + list(sobol_indices.values())
        }

        df = pd.DataFrame(data)
        safe_qoi_name = qoi_name.replace(":", "").replace("/", "").replace("\\", "")
        csv_filename = f"{safe_qoi_name}_Stats_Sobol_{order}.csv"
        csv_filepath = os.path.abspath(csv_filename)
        df.to_csv(csv_filepath, index=False)

    except Exception as e:
        print(f"An error occurred while saving data to CSV: {e}")

"""
    Plots the distribution of samples and Kernel Density Estimates (KDEs) for a specified field 
    and saves the plots as PNG and PDF files.

    This function extracts raw data samples from the provided DataFrame, computes the KDE for 
    both raw and distribution samples, and verifies the normalization of the PDFs. The plot 
    includes histograms of the samples and their respective KDEs.

    Args:
    ----
    results (object): Object containing methods for retrieving distributions and samples.
    results_df (pd.DataFrame): DataFrame containing the field data for plotting.
    field (str): The name of the field for which the distribution is plotted.
    x_label (str): Label for the x-axis of the plot.
    y_label (str, optional): Label for the y-axis of the plot. Default is 'Probability Density'.
    plot_title (str, optional): Title for the plot. Default is 'Distribution'.
    main_title (str, optional): Main title for the plot, displayed above the plot title. Default is None.
    file_name (str, optional): Base name for the output files (PNG and PDF). Default is 'distribution_plot'.

    Returns:
    -------
    None: The function saves the generated plots as PNG and PDF files.
    """
def plot_distribution(results, results_df, field, x_label, y_label='Probability Density', plot_title='Distribution', main_title=None, file_name='distribution_plot'):
    # Extract raw data samples
    raw_samples = np.asarray(results_df[field].values).flatten()

    # Extract the distribution for the specified QoI
    distribution = results.get_distribution(field)

    # Generate samples from the distribution
    dist_samples = np.asarray(distribution.sample(2500)).flatten()  # Adjust the number of samples as needed

    # Check the number of samples
    if len(raw_samples) < 100:
        print("Warning: The number of raw samples is very low. Consider increasing the order or the sample size for better results.")

    # Compute KDE for samples
    pdf_raw_samples = cp.GaussianKDE(raw_samples, estimator_rule='silverman')  # You can try 'scott' or a fixed bandwidth
    pdf_dist_samples = cp.GaussianKDE(dist_samples, estimator_rule='silverman')

    # Define the range for plotting
    x_min, x_max = min(raw_samples.min(), dist_samples.min()), max(raw_samples.max(), dist_samples.max())
    x_range = np.linspace(x_min, x_max, 1000)

    # Compute the PDF values for integration verification
    pdf_raw_samples_values = pdf_raw_samples.pdf(x_range)
    pdf_dist_samples_values = pdf_dist_samples.pdf(x_range)

    # Verify the normalization of PDFs with tolerance
    tolerance = 1e-2
    area_pdf_raw_samples = simpson(y=pdf_raw_samples_values, x=x_range)
    area_pdf_dist_samples = simpson(y=pdf_dist_samples_values, x=x_range)

    print(f"Area under PDF (raw samples): {area_pdf_raw_samples}")
    print(f"Area under PDF (distribution samples): {area_pdf_dist_samples}")

    if abs(1.0 - area_pdf_raw_samples) > tolerance:
        print("Warning: The PDF of raw samples is not properly normalized!")
    else:
        print(f"The PDF of raw samples is properly normalized within tolerance of {tolerance}.")

    if abs(1.0 - area_pdf_dist_samples) > tolerance:
        print("Warning: The PDF of distribution samples is not properly normalized!")
    else:
        print(f"The PDF of distribution samples is properly normalized within tolerance of {tolerance}.")

    # Set font properties
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14

    # Plot
    plt.figure(figsize=(12, 8))
    plt.hist(raw_samples, density=True, bins=40, alpha=0.5, label='Histogram of raw samples', color='blue', edgecolor='black')
    plt.plot(x_range, pdf_raw_samples_values, label='PDF (raw samples)', color='blue', linewidth=2)

    if dist_samples.size > 0:
        plt.hist(dist_samples, density=True, bins=40, alpha=0.5, label='Histogram of PCE distribution samples', color='red', edgecolor='black')
        plt.plot(x_range, pdf_dist_samples_values, label='PDF ( PCE distribution samples)', color='red', linewidth=2)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title, fontsize=16)
    if main_title:
        plt.suptitle(main_title, fontsize=18)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True)
    plt.subplots_adjust(top=0.85)  # Adjust top space to fit main title
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save and show plot
    plt.savefig(f'{file_name}.png', dpi=300)
    plt.savefig(f'{file_name}.pdf', dpi=300)
    plt.show()



# Calculate the polynomial chaos expansion for a range of orders
if __name__ == '__main__':

    R = {}
    max_pce_order = config.get('max_pce_order', 2)  # Default to 2 if not specified in config
    min_pce_order = config.get('min_pce_order', 1)  # Default to 1 if not specified in config
    for pce_order in range(min_pce_order, max_pce_order + 1):
        R[pce_order] = {}
        (R[pce_order]['results_df'],
         R[pce_order]['results'],
         R[pce_order]['times'],
         R[pce_order]['order'],
         R[pce_order]['number_of_samples']) = run_pce_campaign(pce_order=pce_order, use_files=True)

    # Produce a table of the time taken for various phases
    # The phases are:
    #   1: creation of campaign
    #   2: creation of samples
    #   3: running the cases
    #   4: calculation of statistics including Sobols
    #   5: returning of analysed results
    #   6: saving campaign and pickled results

    # save the results
    pickle.dump(R, open('collected_results.pickle','bw'))

    Timings = pd.DataFrame(np.array([R[r]['times'] for r in list(R.keys())]),
                 columns=['Total', 'Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5', 'Phase 6'],
                 index=[R[r]['order'] for r in list(R.keys())])
    Timings.to_csv(open('Timings.csv', 'w'))
    display(Timings)

    last = -1
    O = [R[r]['order'] for r in list(R.keys())]

    if len(O[0:last]) > 0:
        # Set font properties
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'Times New Roman'
        plt.rcParams['font.size'] = 14

        # Plot
        plt.figure(figsize=(12, 8))
    
        # Compute the RMS error for mean and std
        rms_error_mean = [np.sqrt(np.mean((R[o]['results'].describe('Cl', 'mean') - R[O[last]]['results'].describe('Cl', 'mean'))**2)) for o in O[0:last]]
        rms_error_std = [np.sqrt(np.mean((R[o]['results'].describe('Cl', 'std') - R[O[last]]['results'].describe('Cl', 'std'))**2)) for o in O[0:last]]
    
        # Plot the RMS errors
        plt.semilogy(O[0:last], rms_error_mean, 'o-', label='mean')
        plt.semilogy(O[0:last], rms_error_std, 'o-', label='std')

        plt.xlabel('PCE order')
        plt.ylabel('RMSError compared to order=%s' % (O[last]))
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True)  # Add grid for better readability
        plt.subplots_adjust(top=0.85)  # Adjust top space to fit main title
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to prevent clipping
    
        # Save the figure
        plt.savefig('Convergence_mean_std.png', dpi=300)
        plt.savefig('Convergence_mean_std.pdf', dpi=300)
        plt.show()  # Display the plot
    else:
        print("Not enough orders to plot convergence.")

    # plot the convergence of the first sobol to that of the highest order
    last = -1
    O = [R[r]['order'] for r in list(R.keys())]

    if len(O[0:last]) > 0:
        # Set font properties
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'Times New Roman'
        plt.rcParams['font.size'] = 14

        # Plot
        plt.figure(figsize=(12, 8))
    
        O = [R[r]['order'] for r in list(R.keys())]
        for v in list(R[O[last]]['results'].sobols_first('Cl').keys()):
            plt.semilogy([o for o in O[0:last]],
                        [np.sqrt(np.mean((R[o]['results'].sobols_first('Cl')[v] -
                                        R[O[last]]['results'].sobols_first('Cl')[v])**2)) for o in O[0:last]],
                        'o-', label=v)
    
        plt.xlabel('PCE order')
        plt.ylabel('RMSerror for 1st Sobol compared to order=%s' % (O[last]))
        plt.title('Convergence of First Sobol Indices', fontsize=16)
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True)  # Add grid for better readability
        plt.subplots_adjust(top=0.85)  # Adjust top space to fit main title
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to prevent clipping
    
        # Save the figure
        plt.savefig('Convergence_sobol_first.png', dpi=300)
        plt.savefig('Convergence_sobol_first.pdf', dpi=300)
        plt.show()  # Display the plot
    else:
        print("Not enough orders to plot convergence.")


     # Plot a standard set of graphs for the highest order case
    last = -1  # Index to get the last element
    title = config['title_template'] % list(R.values())[last]['order']  # Use title template from config

    # Lift Coefficient (Cl)
    results = list(R.values())[last]['results']
    results_df = list(R.values())[last]['results_df']
    qoi_name = 'Cl'

    print("Processing Lift Coefficient (Cl)...")
    extract_results(results, [qoi_name])
    plot_distribution(results, results_df, qoi_name, 'Lift Coefficient ($C_L$)', 'Probability Density', 'Distribution of Lift Coefficient', title, 'Lift_Coefficient_Distribution')
    print("Lift Coefficient (Cl) post-processing completed successfully.")

    # Drag Coefficient (Cd)
    qoi_name = 'Cd'

    print("Processing Drag Coefficient (Cd)...")
    extract_results(results, [qoi_name])
    plot_distribution(results, results_df, qoi_name, 'Drag Coefficient ($C_D$)', 'Probability Density', 'Distribution of Drag Coefficient', title, 'Drag_Coefficient_Distribution')
    print("Drag Coefficient (Cd) post-processing completed successfully.")

    # Lift Force (L)
    qoi_name = 'L'

    print("Processing Lift Force (L)...")
    extract_results(results, [qoi_name])
    plot_distribution(results, results_df, qoi_name, 'Lift Force ($L$)', 'Probability Density', 'Distribution of Lift Force (N)', title, 'Lift_Distribution')
    print("Lift Force (L) post-processing completed successfully.")

    # Drag Force (D)
    qoi_name = 'D'

    print("Processing Drag Force (D)...")
    extract_results(results, [qoi_name])
    plot_distribution(results, results_df, qoi_name, 'Drag Force ($D$)', 'Probability Density', 'Distribution of Drag Force (N)', title, 'Drag_Distribution')
    print("Drag Force (D) post-processing completed successfully.")

    # Lift to Drag Ratio (L/D)
    qoi_name = 'L/D'

    print("Processing Lift to Drag Ratio (L/D)...")
    extract_results(results, [qoi_name])
    plot_distribution(results, results_df, qoi_name, 'Lift/Drag Ratio ($L/D$)', 'Probability Density', 'Distribution of Lift/Drag Ratio', title, 'Lift_to_Drag_Distribution')
    print("Lift to Drag Ratio (L/D) post-processing completed successfully.")

    # plot the RMS surrogate error at the PCE vary points
    
    _o = []
    _RMS = []
    for r in R.values():
        results_df = r['results_df']
        results = r['results']
        NACA_surrogate = np.squeeze(np.array(results.surrogate()(results_df[results.inputs])['Cl']))
        NACA_samples = np.squeeze(np.array(results_df['Cl']))
        _RMS.append((np.sqrt((((NACA_surrogate - NACA_samples))**2).mean())))
        _o.append(r['order'])

    # Set font properties
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14

    # Plot
    plt.figure(figsize=(12, 8))
    plt.semilogy(_o, _RMS, 'o-')
    plt.xlabel('PCE order')
    plt.ylabel('RMS error for the PCE surrogate')
    plt.title('Convergence of RMS Surrogate Error', fontsize=16)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True)  # Add grid for better readability
    plt.subplots_adjust(top=0.85)  # Adjust top space to fit main title
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to prevent clipping

    # Save the figure
    plt.savefig('Convergence_surrogate.png', dpi=300)
    plt.savefig('Convergence_surrogate.pdf', dpi=300)
    plt.show()  # Display the plot

