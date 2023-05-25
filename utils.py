import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import sys

def is_numeric_column(df, col_name):
	dtype = df[col_name].dtype
	return np.issubdtype(dtype, np.number)

def count(array):
	return (array.shape[0])

def mean(array):
	return np.nansum(array) / (array.shape[0])

def std(array):
	somme = np.nansum((array - mean(array)) ** 2)
	N = array.shape[0]
	return np.sqrt(somme / N)

def percentile(percentile, array):
	array = np.sort(array)
	result = (percentile / 100) * len(array)
	return array[math.ceil(result) - 1]