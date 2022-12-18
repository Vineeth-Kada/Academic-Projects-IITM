================== Part A / Synthetic Data ============================

Command : python3 sythetic.py

Directories:
(a)Data : put train.txt and dev.txt in Synthetic directory
(b)Results : Plots will be saved to PartA_images/synthetic_png directory
==============================================================
GMM for Real Data

How to run our program:
    Step1:
    In the folder RealData please add the traing & Dev data
    It should look like this:
    RealData
    |-> coast
        |-> train 
        |-> dev
    |-> forest
    |-> ...
    |-> ...
    
    Step2:
    > python3 RealData.py
    This will store all the results in the RealData_Results folder.

***********************************************
DTW

How to run our program:
    Step1:
    Please add training & dev data to Handwriting_Data & Isolated_Digits_Data folders respectively.
    Handwriting_Data should contain folder ai, bA, dA,....
    Isolated_Digits_Data should contain folders 1, 4, 5, 7, o
    
    Step2:
    > python3 DTW.py
    This will store all the results in the Results folder, for both Handwritten & Spoken Character data

***********************************************

================== Part B / HMM ============================

Libraries Used :

import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import random
import pandas as pd
from sklearn.metrics import DetCurveDisplay
import os
import sys

Command To be executed : python3 hmm.py

Directory Structures required : 
==> Given HMM-Code directory should present in this directory

i.Data Directories
a)speech_data : should contains different classes data as given in drive for our team
b)character_data : should contains different classes data as given in drive for our team

2.Results Directories
a)HMM_speech_plots : HMM speech part related plots will be saved here
a)HMM_character_plots : HMM character part related plots will be saved here