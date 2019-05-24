from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

def get_data():
	df = pd.read_csv('Attrition_Dataset.csv')
	df.shape

	predictors = df.iloc[:, 3:13].values
	label = df.iloc[:, 13].values 

	encoder_1 = LabelEncoder()
	predictors[:, 1] = encoder_1.fit_transform(predictors[:, 1])

	encoder_2 = LabelEncoder()
	predictors[:, 2] = encoder_2.fit_transform(predictors[:, 2])

	onehot = OneHotEncoder(categorical_features = [1])
	predictors = onehot.fit_transform(predictors).toarray()

	predictors = predictors[:, 1:]
	# rus = RandomOverSampler(random_state = 0, ratio = {0:10000, 1:6000})
	# rus.fit(predictors, label)
	# predictors, label = rus.fit_sample(predictors, label)
	predictors_train, predictors_test, label_train, label_test = train_test_split(predictors, label, test_size = 0.2, random_state = 0)

	scaler = StandardScaler()
	predictors_train = scaler.fit_transform(predictors_train)
	predictors_test = scaler.transform(predictors_test)
	return predictors_train, label_train, predictors_test, label_test
