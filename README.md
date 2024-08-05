# Uncertainty Quantification in CFD Simulations via PCE Methods

## Project Overview

This repository is dedicated to implementing uncertainty quantification for computational fluid dynamics (CFD) simulations using Polynomial Chaos Expansion (PCE) methods. The main script, `Naca0012_EasyVVUQ_Diff_PCE_Order.py`, interfaces with STAR-CCM+ to facilitate the analysis and is adaptable for various CFD scenarios.

## Repository Contents

- **Java Template File:** `Java.jinja2`
- **Star-CCM+ Simulation File:** `.sim`
- **Python Script for Uncertainty Evaluation:** `Naca0012_EasyVVUQ_Diff_PCE_Order.py`
- **Configuration File:** `config.json`

## Setup Instructions

Before running the code, ensure the following steps are completed:

1. **Set File Permissions:** Make sure all files have the necessary access permissions.
2. **Update File Paths:** Modify the file paths in the json file to match your local environment.

## Code Structure

1. **Directory Management:** This section of the code handles the organization and management of the repository.
2. **Simulation Tools Setup:** This section establishes the tools required for running the simulations.
3. **Post-Processing:** This section focuses on analyzing and processing the simulation results.

## Common Issues

A common issue is Star-CCM+ failing to run the simulations due to incorrect paths. To troubleshoot, run a single simulation using the Windows shell to verify that the paths are correctly configured.

## Detailed Script Description

### `Naca0012_EasyVVUQ_Diff_PCE_Order.py`

**Author:** Luca A. Mattiocco  
**Email:** [luca.mattiocco@cranfield.ac.uk](mailto:luca.mattiocco@cranfield.ac.uk) or [luca.mattiocco@orange.fr](mailto:luca.mattiocco@orange.fr)  
**Date:** 31/07/24  
**Version:** 1.1

### Description

This script facilitates uncertainty quantification for computational fluid dynamics (CFD) simulations using Polynomial Chaos Expansion (PCE) methods. It is designed to interface with STAR-CCM+ and can be adapted for different CFD scenarios. The script handles key parameters such as velocity, temperature, and pressure, incorporating probabilistic variations through normal distributions but can be tuned as user desires.

**Note:** This script is part of a master's thesis project. While every effort has been made to ensure accuracy and functionality, it is provided "as is". Feel free to use, modify, and share, but please give credit where it's due!

### Main Functions

- `remove_dir(path)`: Removes a directory and handles any permission errors.
- `load_config(json_file)`: Loads configuration settings from a JSON file.
- `define_params()`: Defines the parameters for the StarCCM+ model.
- `define_vary()`: Defines the varying quantities for uncertainty quantification.
- `run_pce_campaign(pce_order, use_files=True)`: Runs a PCE campaign for a StarCCM+ model simulation.
- `compute_statistical_moments(results, qoi_name)`: Computes statistical moments for a specified quantity of interest.
- `compute_sobol_indices(results, field)`: Computes Sobol indices for a specified field.
- `save_results_to_csv(statistical_moments, sobol_indices, qoi_name, order)`: Saves the statistical moments and Sobol indices to a CSV file.
- `compute_statistics_and_sobol(results, qoi_name, order, field='Default')`: Computes and prints statistical moments and Sobol indices for a specified quantity of interest and field.
- `plot_distribution(results, results_df, field, x_label, y_label='Default', plot_title=None, main_title=None, file_name='Default')`: Plots the distribution of samples and KDEs for a specified field and saves the plots to PNG and PDF files.

## Usage Instructions

1. **Configure the Parameters:** Ensure the parameters in the `config.json` file are set correctly.
2. **Run the Script:** Execute the main script to perform the uncertainty quantification and analyze the results.

    ```bash
    python Naca0012_EasyVVUQ_Diff_PCE_Order.py
    ```

3. **View Results:** The results will be saved in the specified directories and can be viewed and analyzed using the provided plotting functions.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests with your improvements.

## License

This project is licensed under the MIT License.

## Contact Information

For any questions or feedback, please contact Luca A. Mattiocco at:

- Email: [luca.mattiocco@cranfield.ac.uk](mailto:luca.mattiocco@cranfield.ac.uk)
- Email: [luca.mattiocco@orange.fr](mailto:luca.mattiocco@orange.fr)
