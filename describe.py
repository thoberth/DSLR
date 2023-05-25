import numpy as np
import pandas as pd
import sys
from utils import *

def compute_stat(df):
	features_dict = {}
	for col_name in df.columns:
		if is_numeric_column(df, col_name) == True:
			np_colonne = df[col_name].to_numpy()
			np_colonne = np_colonne[~np.isnan(np_colonne)]
			if len(np_colonne) > 0:
				features_dict[col_name] = {
					'Count': count(np_colonne),
					'Mean': mean(np_colonne),
					'Std': std(np_colonne),
					'Min': np.min(np_colonne),
					'25%': percentile(25, np_colonne),
					'50%': percentile(50, np_colonne),
					'75%': percentile(75, np_colonne),
					'Max': np.max(np_colonne) }
	return features_dict

def describe_stat(features_dict):
	print(f'{"":15} |{"Count":>12} |{"Mean":>12} |{"Std":>12} |{"Min":>12} |{"25%":>12} |{"50%":>12} |{"75%":>12} |{"Max":>12} |')
	for key, value in features_dict.items():
		print(f'{key:15.15}', end=' |')
		for k, v in value.items():
			print(f'{v:>12.4f}', end=' |')
		print()

if __name__=="__main__":
	if len(sys.argv) != 2:
		print('Error, program must have 1 parameter')
		exit(1)
	try:
		df = pd.read_csv(sys.argv[1])
	except:
		print('Error while opening the file, please make sure your file is a .csv and the path is correct')
		exit(1)
	features_dict = compute_stat(df)
	describe_stat(features_dict)
