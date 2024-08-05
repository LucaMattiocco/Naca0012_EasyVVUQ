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
2. **Update File Paths:** Modify the file paths in the scripts to match your local environment.

## Code Structure

1. **Directory Management:** This section of the code handles the organization and management of the repository.
2. **Simulation Tools Setup:** This section establishes the tools required for running the simulations.
3. **Post-Processing:** This section focuses on analyzing and processing the simulation results.

## Common Issues

A common issue is Star-CCM+ failing to run the simulations due to incorrect paths. To troubleshoot, run a single simulation using the Windows shell to verify that the paths are correctly configured.

## Example Windows Shell Command to Run a Star-CCM+ Simulation

```bash
"C:\Program Files\CD-adapco\Star-CCM+\16.06.007\star\bin\starccm+" -batch simulation.sim

## Detailed Script Description

Detailed Script Description
Naca0012_EasyVVUQ_Diff_PCE_Order.py
Author: Luca A. Mattiocco
Email: luca.mattiocco@cranfield.ac.uk or luca.mattiocco@orange.fr
Date: 31/07/24
Version: 1.1

Description
This script facilitates uncertainty quantification for computational fluid dynamics (CFD) simulations using Polynomial Chaos Expansion (PCE) methods. It is designed to interface with STAR-CCM+ and can be adapted for different CFD scenarios. The script handles key parameters such as velocity, temperature, and pressure, incorporating probabilistic variations through normal distributions but can be tuned as user desires.

Note: This script is part of a master's thesis project. While every effort has been made to ensure accuracy and functionality, it is provided "as is". Feel free to use, modify, and share, but please give credit where it's due!
