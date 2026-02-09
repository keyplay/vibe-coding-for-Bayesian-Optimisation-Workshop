# 🎉 Bayesian Optimisation Workshop Notebooks

## General Information

This repository contains notebooks and hackathons on the subject of Bayesian Optimisation. The notebook content is roughly divided into three sections: (i) Intuition and Mathematics behind Bayesian Optimisation (ii) Bayesian Optimisation for Chemical Reaction Optimisation and (iii) Bayesian Optimisation with BOTorch.

**Authors:**

- [Yong Lee](https://www.optimlpse.co.uk/author/yong-lee/)

📅 Participants should have received a **preliminary schedule**. Any updates or changes will be communicated.

## Notebook Content Breakdown

### (I) Intuition and Mathematics behind Bayesian Optimisation

This section covers notebooks A to D. The notebooks focuses on the mathematical concepts governing Bayesian Optimisation. The exercises are a mixutre of coding exercises (ex. defining a function) and excersises for concept reinforcement.

A: Statistical Intuition for Gaussian Process

* 1D Gaussian Distribution
* Multivariate Gaussian Distribution
* The Covariance Matrix
* Marginalisation
* Probablistic Conditioning

B: Gaussian Process 

* Covariance Matrix
* Kernel Functions
* Conditioning on Training Data
* GP regression
* The mean function

C: Bayesian Optimisation

* Exploration vs Exploitation
* Acquisition Functions (LCB)
* 1D BO
* Multidimensional BO

D. Multi-batch Bayesian Optimisation 

* Runtime of BO code
* Priors and Posteriors
* Thompson Sampling
* Marginal Thompson Sampling

### (II) Bayesian Optimisation for Chemical Reaction Optimisation

This notebook explores how the fundemental BO workflow (with code from section A -> D) can be applied to chemical reaction optimisation. The focus here is on how the learnt mathematics of BO from previous sections can be applied (rather than to develop an effective BO workflow for reaction optimisation). Attached at the end of the notebook is a list of resources if one would like to deploy and use BO for reaction optimisation.

E: BO for Reaction Optimisation

* Chemical Descriptors
* Optimisation of reaction conditions with BO
* Resources for practical BO

### (III) Bayesian Optimisation with BOTorch

This section covers notebooks F to H. The notebooks focuses on how BO is realistically deloyed for a range of applications and use cases. We move from defining each BO components ourselves from the previous sections to using an efficient BO package  called BOTorch. 

F: GP with BOTorch

* PyTorch Tensor creation and manipulation
* GP regression
* Sampling from the Posterior

G: BO with BoTorch

* Acquisition Functions (LCB, PI, EI)
* Monte Carlo batch Aquisition Functions (qEI, qlogEI, qNEI, qlogNEI)
* Full BO loop

H: Multi-fidelity Bo with BOTorch - Illustrative Example

* Example of 1D input with qMFKG
* Example of multi-dimensional input with Augmented Hartmann

## Exercise and Hackathons - Information and Submission

### Information

There are 3 exercises/hackathons contained in this repository. They call for the construction of a self-contained BO algorithm for the optimisation of various problems:

A. Optimisation of a test function.

B. Optimisation of chemical reaction condition.

C. Optimisation of bioprocess.

### Submission

Paste your code into a .txt file (instead of a .py file) and send it via email to the following email: yl5119@imperial.ac.uk

## Teaching Material and Instructions

The following notebooks and excersises were created by Yong Lee with a section of code adapted from [OptiML PSE Group](https://github.com/OptiMaL-PSE-Lab) !

🖥️ For software requirements, please refer to the [INSTALLATION.md](./INSTALLATION.md).

Alternatively - you can run the notebooks on Google Colab!

## Instructions for Google Colab!

'Colab is a hosted Jupyter Notebook service that requires no setup to use and provides free access to computing resources.' - Google Colabratory (2025)

1. Open Google Colab and Sign In (You will need a Google account)

https://colab.research.google.com/

![image](https://github.com/user-attachments/assets/92703f5d-5b00-4f8d-b523-5716caea1625)

2. Copy the link bellow and Paste it to where it says 'Enter a GitHub URL'

https://github.com/Yong-Labs/Bayesian-Optimisation-Workshop---Master-Student-Version/

![image](https://github.com/user-attachments/assets/416100d6-634f-4e6f-b60e-668e1d705f9a)

3. Click on the desired notebook. You should now be able to access and run code on it!

![image](https://github.com/user-attachments/assets/cc9c531c-32c7-4cf6-a333-4643587a0a5e)
