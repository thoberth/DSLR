from tqdm import tqdm
from utils import *
import pickle

def normalize(x) -> np.ndarray:
	mean = np.sum(x) / len(x)
	std = np.sqrt(np.sum((x - mean)**2) / len(x))
	x_norm = (x - mean) / (std)
	return x_norm

def predict(X, theta):
	X_extented = np.hstack((np.ones((X.shape[0], 1)), X))
	y_hat = 1 / (1 + np.e ** - X_extented.dot(theta))
	return y_hat

def compute_gradient(X, y, theta):
	X_extented = np.hstack((np.ones((X.shape[0], 1)), X))
	log_gradient = (1/len(y)) * X_extented.T.dot(predict(X, theta) - y)
	return log_gradient

def fit(X, y, alpha=1e-3, iter=15000, algo="GD", nbr_batch=50):
	print('Entrainement du modele:')
	theta = np.random.rand(X.shape[1] + 1, 1).reshape(-1, 1)
	# diviser X et y if algo == MBGD en nbr_batch
	if algo == "MBGD":
		batch_X = np.array_split(X, nbr_batch)
		batch_y = np.array_split(y, nbr_batch)
		i = 0
	for _ in tqdm(range(iter)):
		if algo == "GD":
			log_gradient = compute_gradient(X, y, theta)
		elif algo == "SGD":
			to_train = random.randint(0, X.shape[0] - 1)
			log_gradient = compute_gradient(X[to_train, :].reshape(1, -1), y[to_train, :], theta)
		elif algo == "MBGD":
			if i == nbr_batch:
				i = 0
			log_gradient = compute_gradient(batch_X[i], batch_y[i], theta)
			i += 1
		theta = theta - (alpha * log_gradient)
	return theta

def loss(y, y_hat, eps=1e-15):
	log_loss = (y * np.log(y_hat + eps)) +\
			((1 - y) * np.log(1 - y_hat + eps))
	return (-1/len(y)) * np.sum(log_loss)

def sort_y(y_to_sort, target):
	y = np.zeros(y_to_sort.shape)
	for i, is_target in enumerate(y_to_sort):
		if is_target == target:
			y[i] = 1
	return y

def sort_yhat(preds):
	y_hat = np.zeros(preds[0].shape)
	for i, pred_0, pred_1, pred_2, pred_3 in zip(range(len(y_hat)), preds[0], preds[1], preds[2], preds[3]):
		best = max(pred_0, pred_1, pred_2, pred_3)
		if best == pred_0:
			y_hat[i] = 0
		if best == pred_1:
			y_hat[i] = 1
		if best == pred_2:
			y_hat[i] = 2
		if best == pred_3:
			y_hat[i] = 3
	return y_hat

def save_thetas(to_save):
	with open('theta.pkl', 'wb') as f:
		pickle.dump(to_save, f)

if __name__=="__main__":
	algo = "GD"
	if len(sys.argv) != 2:
		if len(sys.argv) != 3 or (sys.argv[2] != 'SGD' and sys.argv[2] != 'MBGD'):
			print('Error: Usage:\npython logreg_train.py path/to/dataset.csv *BONUS*')
			exit(1)
		algo = sys.argv[2]
	try:
		df = pd.read_csv(sys.argv[1])
	except:
		print('Error while opening the file, please make sure your file is a .csv and the path is correct')
		exit(1)
	# Choisir les données significatives grace au pair plot puis les stockés et les normalizer dans X
	# ne pas prendre 'Care of Magical Creature' et 'Arithmancy' et peut etre 'Potions'
	df.drop(['Index', 'Care of Magical Creatures', 'Arithmancy'], axis=1, inplace=True)
	df.dropna(axis=0, inplace=True)
	y = df['Hogwarts House'].to_numpy().reshape(-1, 1)
	X = None
	for col in df.columns:
		if is_numeric_column(df, col) == True:
			if type(X) != np.ndarray:
				X = df[col].to_numpy().reshape(-1, 1)
			else:
				X = np.c_[X, df[col].to_numpy().reshape(-1, 1)]
	for i in range(X.shape[1]):
		X[:, i] = normalize(X[:, i])
	legend = { 0:'Gryffindor', 1:'Hufflepuff', 2:'Ravenclaw', 3:'Slytherin' }
	theta, preds = [], []
	for i in range(4):
		theta.append(fit(X, sort_y(y, legend[i]), algo=algo))
	save_thetas(theta)
