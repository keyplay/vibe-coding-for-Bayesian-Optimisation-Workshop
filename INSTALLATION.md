# BO_Workshop Environment Setup

This guide provides instructions on how to install and use the `BO_Workshop` Conda environment.

## Prerequisites

Ensure that you have Conda installed on your system. If Conda is not installed, you can install Miniconda or Anaconda:

### Install Miniconda (Recommended)

Miniconda is a minimal Conda installer that includes only Conda and essential dependencies:

**Windows:**

```sh
winget install Anaconda.Miniconda3
```

**Mac/Linux:**

```sh
curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh
```

Alternatively, install Anaconda from [Anaconda&#39;s official website](https://www.anaconda.com/products/distribution).

## Creating the Environment

Once Conda is installed, create the `BO_Workshop` environment using the provided YAML file:

```sh
conda env create -f environment.yml
```

This will create a Conda environment with Python 3.10 and the required packages.

## Activating the Environment

To activate the environment, run:

```sh
conda activate BO_Workshop
```

## Verifying Installation

To check if the installation was successful, run:

```sh
python -c "import numpy, scipy, tqdm, sobol_seq, botorch, matplotlib; print('All packages imported successfully!')"
```

If there are no errors, the environment is set up correctly.

## Deactivating and Removing the Environment

To deactivate the environment, use:

```sh
conda deactivate
```

To remove the environment completely:

```sh
conda env remove -n BO_Workshop
```
