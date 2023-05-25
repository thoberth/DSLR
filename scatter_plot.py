from utils import *

if __name__=="__main__":
	df = pd.read_csv('datasets/dataset_train.csv')
	fig, ax = plt.subplots()
	ax.scatter(df.loc[df['Hogwarts House'] == 'Ravenclaw',
                   'Defense Against the Dark Arts'], df.loc[df['Hogwarts House'] == 'Ravenclaw', 'Astronomy'], label='Ravenclaw', c='red', alpha=0.6)
	ax.scatter(df.loc[df['Hogwarts House'] == 'Slytherin',
                   'Defense Against the Dark Arts'], df.loc[df['Hogwarts House'] == 'Slytherin', 'Astronomy'], label='Slytherin', c='blue', alpha=0.6)
	ax.scatter(df.loc[df['Hogwarts House'] == 'Gryffindor',
                   'Defense Against the Dark Arts'], df.loc[df['Hogwarts House'] == 'Gryffindor', 'Astronomy'], label='Gryffindor', c='yellow', alpha=0.6)
	ax.scatter(df.loc[df['Hogwarts House'] == 'Hufflepuff',
                   'Defense Against the Dark Arts'], df.loc[df['Hogwarts House'] == 'Hufflepuff', 'Astronomy'], label='Hufflepuff', c='green', alpha=0.6)
	ax.legend()
	ax.set_xlabel('Defense Against the Dark Arts')
	ax.set_ylabel('Astronomy')
	plt.show()
