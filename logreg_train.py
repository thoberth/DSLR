from utils import *

def normalize(x) -> np.ndarray:
	mean = np.sum(x) / len(x)
	std = np.sqrt(np.sum((x - mean)**2) / len(x))
	x_norm = (x - mean) / (std)
	return x_norm

def predict(X, theta):
	pass

def gradient(X, y):
	pass

def fit(X, y, alpha=1e-3, iter=10000):
	pass

def loss(y, y_hat, eps=1e-15):
	pass

if __name__=="__main__":
	if len(sys.argv) != 2:
		print('Error program must take one parameter.')
		exit(1)
	try:
		df = pd.read_csv(sys.argv[1])
	except:
		print('Error while opening the file, please make sure your file is a .csv and the path is correct')
		exit(1)
	df.drop('Index', axis=1, inplace=True)
	# Choisir les données significatives grace au pair plot puis les stockés et les normalizer dans X