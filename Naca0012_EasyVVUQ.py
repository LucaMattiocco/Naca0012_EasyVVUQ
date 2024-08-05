import os
from shutil import rmtree
import easyvvuq as uq
import chaospy as cp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from easyvvuq.actions import ExecuteLocal

# Get the current working directory of the script, specify the input filename, and specify the output filename
work_dir = os.path.dirname(os.path.abspath(__file__))
input_filename = "Macro_Naca0012_Komega_Optimized.java.jinja2"
out_file = "output.csv"

# Define the path to the template file
template = f"{work_dir}/Macro_Naca0012_Komega_Optimized.java.jinja2"

# Print the path of the template file for verification
print(f"Template path: {template}")

# Read and display the content of the template file
with open(template, 'r') as file:
    content = file.read()

print("Contents of the template file:")
print(content)

campaign_work_dir = os.path.join(work_dir, "Naca0012PCE")

# Clear the target campaign directory with error handling
def remove_dir(path):
    try:
        if os.path.exists(path):
            rmtree(path)
    except PermissionError as e:
        print(f"PermissionError: {e}")
        print("Ensure no files are open and retry.")
        return

remove_dir(campaign_work_dir)
os.makedirs(campaign_work_dir)

db_location = "sqlite:///" + campaign_work_dir + "/campaign.db"
campaign = uq.Campaign(
    name="Naca0012M015",
    db_location=db_location,
    work_dir=campaign_work_dir
)

# Define your parameter space and distributions
params = {
    "velocity": {"type": "float", "default": 51.04},
    "static_temp": {"type": "float", "default": 287.15},
    "ref_pressure": {"type": "float", "default": 101325},
    "Ru": {"type": "float", "default": 8.314},  # Universal gas constant
    "M_air": {"type": "float", "default": 0.0289647},  # Molar mass of air
    "gamma": {"type": "float", "default": 1.4},
    "power" : {"type": "float", "default": 0.5},
}

class CustomDecoder:
    def __init__(self, target_filename=None, output_columns=None):
        self.target_filename = target_filename
        self.output_columns = output_columns

    def parse_sim_output(self, run_info={}):
        # Specify the full path of the CSV file generated by the simulation
        file_path = self.target_filename
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Select the last iteration value
        df_last = df.iloc[-1]
        
        # Convert to a dictionary compatible with EasyVVUQ
        result = df_last.to_dict()
        
        return result

encoder = uq.encoders.JinjaEncoder(template_fname='Macro_Naca0012_Komega_Optimized.java.jinja2', target_filename='Macro_Naca0012_Komega_Optimized.java')
decoder = CustomDecoder(target_filename='C:\\Thesis_Destinus\\Script\\Cl.csv')
execute = ExecuteLocal('C:\\Program Files\\Siemens\\19.02.009-R8\\STAR-CCM+19.02.009-R8\\star\\lib\\win64\\clang15.0vc14.2-r8\\lib\\starccm+.exe -batch Macro_Naca0012_Komega_Optimized.java C:\\Thesis_Destinus\\Script\\TestCase_Naca0012_STT_KOmega_Optimized.sim')

# Define the actions
actions = uq.actions.Actions(
    uq.actions.CreateRunDirectory(root=os.getcwd()),  # Create run directories
    uq.actions.Encode(encoder),  # Encode the parameters
    execute,  # Execute the simulation
    uq.actions.Decode(decoder)  # Decode the results
)

# Add the app to the campaign with the actions
campaign.add_app(name="naca0012_simulation", params=params, actions=actions)

# Specify Sampler
vary = {
    "velocity": cp.Normal(51.04, 0.5),
    "static_temp": cp.Normal(288.15, 0.15),
    "ref_pressure": cp.Normal(101325, 15)
}

# Associate a sampler with the campaign
campaign.set_sampler(uq.sampling.PCESampler(vary=vary, polynomial_order=1))

# Draw samples and print them to verify
campaign.draw_samples()
print(f"Number of samples = {campaign.get_active_sampler().count}")

# Execute the campaign
print('Starting campaign...')
try:
    campaign.execute(sequential=True).collate()
except Exception as e:
    print(e)
print('Finished campaign')

# Get the campaign results
results = campaign.get_collation_result()

# Display results for diagnostic
print(results)

def perform_pce_analysis(results, sampler, qoi_name):
    # Perform PCE analysis using the analysis method
    my_analysis = uq.analysis.PCEAnalysis(
        sampler=sampler,
        qoi_cols=[qoi_name],
    )

    # Apply PCE analysis to the campaign
    campaign.apply_analysis(my_analysis)

    # Execute the analysis
    analysis_results = my_analysis.analyse(data_frame=results)

    # Statistical moments
    mean = analysis_results.describe(qoi=qoi_name, statistic="mean")
    variance = analysis_results.describe(qoi=qoi_name, statistic="var")
    std_dev = analysis_results.describe(qoi=qoi_name, statistic="std")

    print(f"\nStatistical moments for {qoi_name}:")
    print(f"  Mean: {mean}")
    print(f"  Variance: {variance}")
    print(f"  Standard Deviation: {std_dev}")

    # Sobol indices
    sobol_first = analysis_results.sobols_first(qoi=qoi_name)
    sobol_second = analysis_results.sobols_second(qoi=qoi_name)
    sobol_total = analysis_results.sobols_total(qoi=qoi_name)

    print(f"\nSobol indices for {qoi_name}:")
    print("  First order Sobol indices:")
    for param, value in sobol_first.items():
        print(f"    {param}: {value}")

    print("  Second order Sobol indices:")
    for param, value in sobol_second.items():
        print(f"    {param}: {value}")

    print("  Total Sobol indices:")
    for param, value in sobol_total.items():
        print(f"    {param}: {value}")

    # Check if correlation matrices are available and display them
    if 'correlation_matrices' in analysis_results.raw_data:
        correlation_matrix = analysis_results.raw_data['correlation_matrices'][qoi_name]
        print(f"\nCorrelation matrix for {qoi_name}:")
        print(correlation_matrix)
    else:
        print("\nCorrelation matrices are not available.")

    # Use get_distribution to obtain the PDF
    pdf = analysis_results.get_distribution(qoi_name)

    # Generate samples from the PDF
    samples = pdf.sample(10000)  # for example, 10,000 samples

    # Estimate the probability density using gaussian_kde
    kde = gaussian_kde(samples)

    # Generate points to plot the smooth line
    x = np.linspace(min(samples), max(samples), 1000)
    y = kde(x)

    # Plot the probability density for the lift coefficient
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='PCE', color='blue')

    # Add labels and legend
    plt.xlabel('Lift Coefficient')
    plt.ylabel('Probability Density Function')
    plt.legend()
    plt.title('Probability Density Function of the Lift Coefficient')
    plt.show()

    # Plot first and second order Sobol indices in a hierarchical treemap
    fig, ax = plt.subplots(figsize=(10, 10))
    analysis_results.plot_sobols_treemap(qoi=qoi_name, ax=ax)
    plt.title('Treemap of Sobol Indices')
    plt.show()

perform_pce_analysis(results, campaign.get_active_sampler(), "Cl Monitor: Cl Monitor")