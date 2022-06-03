from tensorflow.keras import losses, models, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from tensorflow.keras.layers import (Convolution2D, Dense, Dropout, GlobalAveragePooling2D, 
                              GlobalMaxPool2D, Input, MaxPool2D, concatenate, Activation,  
                              MaxPooling2D,Flatten,BatchNormalization, Conv2D,AveragePooling2D)
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.datasets import load_iris 
from sklearn.datasets import make_moons 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn import utils
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

import pickle

data = pd.read_csv("./horse.csv", sep=',', nrows=299)
is_null = pd.isnull(data).sum()
print('The number of empty values by column:')
print(is_null)

data = data.drop(columns=['hospital_number', 'nasogastric_reflux_ph','abdomo_appearance','abdomo_protein'])

print('Old Size: %d' % len(data))
data = data.dropna(how = 'any', axis = 'rows')
print('New Size: %d' % len(data))

print('Check that there are no empty values after cleaning:')
is_null = pd.isnull(data).sum()
print(is_null)
data.corr()

data2 = pd.get_dummies(data, columns =['surgery','age','capillary_refill_time','surgical_lesion',
                                       'cp_data','abdominal_distention','temp_of_extremities',
                                      'peripheral_pulse','mucous_membrane','pain','peristalsis',
                                       'nasogastric_reflux','nasogastric_tube','rectal_exam_feces','abdomen'])
data2.head()
data2 = data2.dropna(how = 'any', axis = 'rows')

Selected_features = ['rectal_temp', 'pulse', 'respiratory_rate', 'packed_cell_volume', 'total_protein', 
                     'surgery_no', 'surgery_yes', 'age_adult','age_young','abdominal_distention_moderate',
                     'abdominal_distention_none','abdominal_distention_severe','abdominal_distention_slight',
                    'peripheral_pulse_increased','peripheral_pulse_normal','peripheral_pulse_reduced']
X = data2[Selected_features]
y = data2['outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train.shape
data2.outcome.describe()
(data2.outcome == 0).sum()
sns.pairplot(data2[Selected_features].corr())

plt.show()

clf = GaussianNB()
clf = clf.fit(X_train, y_train)
moonsY_pred = clf.predict(X_test)
NB_score1 = clf.score(X_test, y_test)
print('NB_score ',NB_score1)
confusion_matrix(y_test, moonsY_pred)

# Make pickle file of our model
pickle.dump(clf, open("model.pkl", "wb"))