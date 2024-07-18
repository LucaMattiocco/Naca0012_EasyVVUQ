# Project Repository Overview

This repository contains the following files:

1. **Java Template File**: `Java.jinja2`
2. **Star-CCM+ Simulation File**: `.sim`
3. **Python Script for Uncertainty Evaluation**: `Python_script.py`

## Setup Instructions

Before running the code, ensure the following steps are completed:

1. **Set File Permissions**: Make sure all files have the necessary access permissions.
2. **Update File Paths**: Modify the file paths in the scripts to match your local environment.

## Code Structure

### 1. Directory Management

This section of the code handles the organization and management of the repository.

### 2. Simulation Tools Setup

This section establishes the tools required for running the simulations.

### 3. Post-Processing

This section focuses on analyzing and processing the simulation results.

## Common Issues

A common issue is Star-CCM+ failing to run the simulations due to incorrect paths. To troubleshoot, run a single simulation using the Windows shell to verify that the paths are correctly configured.

```bash
# Example Windows shell command to run a Star-CCM+ simulation
"C:\Program Files\CD-adapco\Star-CCM+\16.06.007\star\bin\starccm+" -batch simulation.sim

