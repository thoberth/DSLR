from utils import *
import seaborn as sns

if __name__=="__main__":
	df = pd.read_csv('datasets/dataset_train.csv')
	df.drop('Index', axis=1, inplace=True)
	sns.pairplot(df, height=1, aspect=2.5, hue='Hogwarts House', dropna=True)
	plt.show()