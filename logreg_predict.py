from logreg_train import *
from utils import *

def save_result(DataFrame):
	DataFrame[['Index', 'Hogwarts House']].to_csv('houses.csv', index=False)

def load_thetas(file):
	with open(file, 'rb') as f:
		thetas = pickle.load(f)
	return thetas

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print('Error program must take 2 parameters.')
		exit(1)
	try:
		df = pd.read_csv(sys.argv[1])
		thetas = load_thetas(sys.argv[2])
	except:
		print('Error while opening a file, please make sure your file is a .csv or .pkl and the path is correct')
		exit(1)
	df = pd.read_csv('datasets/dataset_test.csv')
	df['Hogwarts House'] = 'None'
	df.drop(['Care of Magical Creatures', 'Arithmancy'], axis=1, inplace=True)
	df.dropna(axis=0, inplace=True)
	X = None
	for col in df.columns:
		if is_numeric_column(df, col) == True and col != 'Index':
			if type(X) != np.ndarray:
				X = df[col].to_numpy().reshape(-1, 1)
			else:
				X = np.c_[X, df[col].to_numpy().reshape(-1, 1)]
	for i in range(X.shape[1]):
		X[:, i] = normalize(X[:, i])
	preds = []
	for i in range(4):
		preds.append(predict(X, thetas[i]))
	df['Hogwarts House'] = sort_yhat(preds)
	print(df['Hogwarts House'])
	legend = {0: 'Gryffindor', 1: 'Hufflepuff', 2: 'Ravenclaw', 3: 'Slytherin'}
	df['Hogwarts House'] = df['Hogwarts House'].replace(legend)
	save_result(df)
