import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN

def get_data():
	RANDOM_SEED = 42
	df = pd.read_csv("creditcard.csv")
	data = df.drop(['Time'], axis=1)

	data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

	x_train, x_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)

	y_train = np.asarray(x_train['Class'])
	x_train = np.asarray(x_train.drop(['Class'], axis = 1))

	y_test = np.asarray(x_test['Class'])
	x_test = np.asarray(x_test.drop(['Class'], axis=1))
	sm = SMOTEENN(random_state=12, ratio = 1.0)
	x_train_res, y_train_res = sm.fit_sample(x_train, y_train)
	return x_train_res, y_train_res, x_test, y_test
