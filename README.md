# AE_KDUL

This is the offical repo for:
Detecting Deformation Mechanisms of Metals from Acoustic Emission Signals through Knowledge-Driven Unsupervised Learning

# Main requirements:

* MATLAB (Version >= 9.10 (R2021a))

## Hardware and software
Model training, model evaluation and visualization of results were conducted on a workstation with an Intel Core i9-10900X CPU, a single NVIDIA RTX 3090 GPU and 64 GB memory. The training process, excluding hyperparameter optimization, lasted about 6 hours on the GPU.
Our code is based on Windows 10 and MATLAB R2021a. 

mksqlite needs to be recompiled if you wish to run it on a different operating system. The source code of mksqlite can be found at 
https://sourceforge.net/projects/mksqlite/files.

Several functions in MATLAB Toolboxs were used to support the pipeline, including:

* Deep Learning Toolbox (Version >= 14.2  (R2021a))
* Statistics and Machine Learning Toolbox (Version >= 12.1 (R2021a))
* Signal Processing Toolbox (Version >= 8.6  (R2021a))
* Curve Fitting Toolbox (Version >= 3.5.13  (R2021a))

# Installation guide
Clone this repository locally:
```
git clone https://github.com/XJTU-BYGou/AE_KDUL.git
```

# Demo and usage
Run training model demo in MATLAB by:
```
cd([rootdir,'\script']);
run('main_trainDemo.m');
```
After training, you will get several .mat files in this directory:
`.\script\export\res_trainMdl_date_time` : Each .mat file stores the trained model and identification for the corresponding classifier. a res_MdlEval.mat file stores the performance for model selection.
This will take about an hour, depending on your computer's performance.

To reproduce the results, you can in MATLAB:
```
run('main_compMdl.m');
run('main_application.m');
run('main_compMdl_synData.m');
run('main_trainCR316LSS.m');
```
you will get several .mat files in `.\script\export`. Each .mat file stores the data used for corresponding figure.

`main_compMdl.m`  for Fig. 2 Fig. 3  Fig. S2 and Fig. S5.

`main_application.m` for Fig. 4 and Fig. S6 in testing data.

`main_compMdl_synData.m` for Fig. S4 in synthetic data.

`main_trainCR316LSS.m` for Fig. S7 in new material data.

# Data

All original datasets including synthetic datasets are available at [here](https://nethelp.xjtu.edu.cn/47d1b1258d6eea8951ba8cda0d575240.zip)
