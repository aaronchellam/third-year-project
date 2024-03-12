# third-year-project

Code from my research-based third year project which explores the use of ensemble learning and deep learning machine learning models to classify remote laser welding (RLW) signal data.

Running Models/compare_classifiers.py will produce test results for the classical models + non-voting ensemble models for every fold. It will also then enumerate the results for all 160+ combinations of ensemble classifiers.
 These results are performed for all datasets so may take some time to run to completion.
 Different types of datasets can be loaded by comment/uncomment out the "d1X, d1y, d2X, d2y, d3X, d3y" lines at the start. There are options for normal statistic feature data, augmented data, and augmented dataset C data.
 
Running Models/Deep_Models/deep_tables.py will produce similar results for an FCNN and CNN model. The hyperparameters are easily modifiable by observing the closure functions in the respective "cnn.py" and "fcnn.py" files in the same folder.
 Again, different forms of the datasets can be used by commenting out lines at the start.
 Alternative CNN architectures are also contained in cnn.py.
 The LSTM model should remain commented out since this can take over an hour to run on dataset 3 and several hours if run on the augmented datasets.


The Util/deep_skfold.py file contains the code used to perform cross-validation for all deep models and classical models.

The Util/deep_logger.py and Util/augmentation.py contain various utility functions used to pre-process and augment the data.

The RLW data is store in the Data folder.

All remaining files contain either utility functions or code that was used to perform intermediate experiments.

Note that many files, particularly the data logging/loading make use of paths from the Pathlib library; these paths have only been tested on windows machines and hence it may not be possible to load data on a Linux or Mac machine.
