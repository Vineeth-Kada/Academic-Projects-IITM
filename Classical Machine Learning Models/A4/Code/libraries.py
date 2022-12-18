import numpy as np
from numba import njit
import math
from sklearn import svm
from sklearn.metrics import DetCurveDisplay, ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
import glob
import matplotlib.pyplot as plt
import sys
import matplotlib.pyplot as plt
from sklearn.covariance import empirical_covariance
from sklearn.neural_network import MLPClassifier
