# RB_GEN

RB_GEN is a simple package for generating random binning features for solving 
large-scale kernel classification, regression, and clustering.  

The codes are written in C/C++ and are also warpped by Matlab, Octave and Python
interfaces. You are welcomed to report any bugs to lwu@email.wm.edu. 


# Possibly Required Dependency and Tools

The codes have been tested on MAC OS systems and Linux OS systems. If C++ 
complier is installed, we expect no other dependency needed. Otherwise, 
if only C complier is installed, the expected required tools: GLib, pkg-config.

For Linux: all available. If not, sudo install them.

For Mac: use homebrew to install them. 
1) If not installed homebrew, install homebrew from website: http://brew.sh/ .
2) brew install glib, brew install pkg-config. 


# How To Complie The Codes
If only C/C++ programs are used, type "make" to complie the codes.

If Matlab or Python interfaces are used, go to the folder Matlab or Python and
type "make" to complie the interface codes and "make test" to verify correctness. 


# Input And Output Data Formats
All inputs and outputs are assumed to follow libsvm data format. An example of 
the libsvm data format are shown below:

Label SparseFeatureIndices (starting from 1)
1 1:0.18 4:0.27 5:0.48 6:0.49 7:0.76 8:0.73 9:1 10:1
0 3:0.27 4:0.24 5:0.51 6:0.53 7:0.75 8:0.81 9:1 10:1


# How To Run The Codes
If only C/C++ programs are used, the following examples are provided:

./rb_gen -R 2 -S 5 -O grid_outfile data/liver-disorders.trainscale data/liver-disorders.trainscale.bin > trainfile

./rb_gen -I grid_outfile data/liver-disorders.testscale  data/liver-disorders.testscale.bin > testfile

For Matlab and Python interfaces, it is straightforward to run the codes if 
you are familiar with either of programming languages. 


# How To Cite The Codes
Please cite our work if you like or are using our codes for your projects!

Lingfei Wu, Ian E.H. Yen, Jie Chen, and Rui Yan, “Revisiting Random Binning Feature: 
Fast Convergence and Strong Parallelizability”, In the Proceeding of the 22th SIGKDD 
conference on Knowledge Discovery and Data Mining, 2016.  

@inproceedings{wu2016revisiting, <br/>
  title={Revisiting random binning features: Fast convergence and strong parallelizability}, <br/>
  author={Wu, Lingfei and Yen, Ian EH and Chen, Jie and Yan, Rui}, <br/>
  booktitle={Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining}, <br/>
  pages={1265--1274}, <br/>
  year={2016}, <br/>
  organization={ACM} <br/>
} <br/>


------------------------------------------------------
Contributors: Lingfei Wu, Eloy Romero <br/>
Created date: August 29, 2016 <br/>
Last update: Sep 15, 2018 <br/>

