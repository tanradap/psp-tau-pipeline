# PSP-tau-pipeline

This repository contains QuPath scripts for data pre-processing, python scripts & jupyter notebooks used to create the tau type-specific quantification pipeline for PSP postmortem brain. Folders are detailed below:

**Qupath scripts:** useful scripts for running commands on qupath.

**Sample_size_check:** scripts for checking we have annotated sufficient tau objects to train our model.

**Tuning_parameters:** for tuning hyperparameters  of the classification model.

**Tuning_parameters/scripts:** some example of how to use the provided functions to train your own model.

**Tau_classification:** for classifying tau into type-specific tau burden classes.

**Tau_classification/Pre-trained:** using pre-trained models to perform tau classification.

**Tau_classification/Untrained:** training your own final models & using them to perform tau classification.

**Tau_quantification:** for collating classified tau from multiple slides into a single file, and some useful polar plot functions.
