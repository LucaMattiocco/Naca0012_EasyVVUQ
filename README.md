# Uncertainty Quantification in CFD Simulations via PCE Methods

## Project Overview

This repository focuses on implementing uncertainty quantification for computational fluid dynamics (CFD) simulations using Polynomial Chaos Expansion (PCE) methods. The main script, `Naca0012_EasyVVUQ_Diff_PCE_Order.py`, interfaces with STAR-CCM+ to facilitate the analysis and is adaptable for various CFD scenarios.

## Repository Contents

- **Java Template File:** `Java.jinja2`
- **Star-CCM+ Simulation File:** `.sim`
- **Python Script for Uncertainty Evaluation:** `Naca0012_EasyVVUQ_Diff_PCE_Order.py`
- **Configuration File:** `config.json`
- **Read Pickle file:**  `ReadPickle.py`

## Setup Instructions

Before running the code, ensure the following steps are completed:

1. **Set File Permissions:** Ensure all files have the necessary access permissions.
2. **Update File Paths:** Modify the file paths in the json file to match your local environment.

## Code Structure

1. **Directory Management:** Handles the organization and management of the repository.
2. **Simulation Tools Setup:** Establishes the tools required for running the simulations.
3. **Post-Processing:** Focuses on analyzing and processing the simulation results.

## Common Issues

A common issue is STAR-CCM+ failing to run the simulations due to incorrect paths. To troubleshoot, run a single simulation using the Windows shell to verify that the paths are correctly configured.

## Detailed Script Description

### `Naca0012_EasyVVUQ_Diff_PCE_Order.py`

**Author:** Luca A. Mattiocco  
**Email:** [luca.mattiocco@cranfield.ac.uk](mailto:luca.mattiocco@cranfield.ac.uk) or [luca.mattiocco@orange.fr](mailto:luca.mattiocco@orange.fr)  
**Date:** 06/08/24  
**Version:** 1.1

### Description

This script facilitates uncertainty quantification for CFD simulations using PCE methods. It interfaces with STAR-CCM+ and can be adapted for different CFD scenarios. The script handles key parameters such as velocity, temperature, and pressure, incorporating probabilistic variations through normal distributions.

**Note:** This script is part of a 3 months master's thesis project at Cranfield University. While every effort has been made to ensure accuracy and functionality, it is provided "as is". Feel free to use, modify, and share, but please give credit where it's due!

### Main Functions

- `remove_dir(path)`: Removes a directory and handles any permission errors.
- `load_config(json_file)`: Loads configuration settings from a JSON file.
- `define_params()`: Defines the parameters for the StarCCM+ model.
- `define_vary()`: Defines the varying quantities for uncertainty quantification.
- `run_pce_campaign(pce_order, use_files=True)`: Runs a PCE campaign for a StarCCM+ model simulation.
- `extract_results(results, qoi_cols)`: Extracts statistical moments and Sobol indices for specified quantities of interest.
- `save_results_to_csv(statistical_moments, sobol_indices, qoi_name, order)`: Saves the statistical moments and Sobol indices to a CSV file.
- `plot_distribution(results, results_df, field, x_label, y_label='Probability Density', plot_title='Distribution', main_title=None, file_name='distribution_plot')`: Plots the raw distribution of samples and PCE Calculated distribution for a specified field and saves the plots to PNG and PDF files.

## Files Generated

The script generates the following files:

- **Sobol Indices and Statistical Moments Files:** CSV files containing Sobol indices for various quantities of interest and computed statistical moments for quantities of interest..
- **Plots:** PNG and PDF files of the distributions of samples and PCE for specified fields.
- **Pickle Files:** Serialized objects for storing intermediate results.
- **Time Logs:** Logs of the time taken for various stages of the simulation and analysis.
- **Simulation Files:** STAR-CCM+ simulation files generated during the process.
- **Results CSV Files:** Comprehensive CSV files of the last iterated simulation on starCCM for each field.

## Usage Instructions

1. **Configure the Parameters:** Ensure the parameters in the `config.json` file are set correctly.
2. **Run the Script:** Execute the main script to perform the uncertainty quantification and analyze the results.

    ```bash
    python Naca0012_EasyVVUQ_Diff_PCE_Order.py
    ```

3. **View Results:** The results will be saved in the specified directories and can be viewed and analyzed using the provided plotting functions.

## Tuning and Running Specific Simulations

You can tune as much as you want. If you want to run just one simulation, specify the same number for both minimum and maximum PCE order. For example:
- To run a simulation with order seven, specify `7` and `7`.
- To run simulations from order 3 to 7, specify `3` and `7`.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests with your improvements.

## License

This project is undertaken as part of a master's thesis at Cranfield University in cooperation with the Swiss company Destinus.

## Contact Information

For any questions or feedback, please contact Luca A. Mattiocco at:

- Email: [luca.mattiocco@cranfield.ac.uk](mailto:luca.mattiocco@cranfield.ac.uk)
- Email: [luca.mattiocco@orange.fr](mailto:luca.mattiocco@orange.fr)
