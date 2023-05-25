from utils import *

if __name__=="__main__":
	df = pd.read_csv('datasets/dataset_train.csv')
	df.drop('Index', axis=1, inplace=True)
	nbr_of_course = 0
	for col_name in df.columns:
		if is_numeric_column(df, col_name) == True:
			nbr_of_course += 1
	fig, axes = plt.subplots(nrows=math.ceil(nbr_of_course/3), ncols=3,
				figsize=(12, 12))
	i, j = 0, 0
	for col_name in df.columns:
		if is_numeric_column(df, col_name) == True:
			axes[i][j].hist(df.loc[df['Hogwarts House'] == 'Ravenclaw',
					col_name], label='Ravenclaw', alpha=0.8)
			axes[i][j].hist(df.loc[df['Hogwarts House'] == 'Slytherin',
						col_name], label='Slytherin', alpha=0.8)
			axes[i][j].hist(df.loc[df['Hogwarts House'] == 'Gryffindor',
						col_name], label='Gryffindor', alpha=0.8)
			axes[i][j].hist(df.loc[df['Hogwarts House'] == 'Hufflepuff',
						col_name], label='Hufflepuff', alpha=0.8)
			axes[i][j].legend()
			axes[i][j].set_title(col_name)
			j += 1
			if j == 3:
				j = 0
				i += 1
		fig.tight_layout()
	plt.show()