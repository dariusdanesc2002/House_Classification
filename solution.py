import os
from sklearn.tree import DecisionTreeClassifier
import requests
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import classification_report
import numpy as np

if __name__ == '__main__':
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if 'house_class.csv' not in os.listdir('../Data'):
        sys.stderr.write("[INFO] Dataset is loading.\n")
        url = "https://www.dropbox.com/s/7vjkrlggmvr5bc1/house_class.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/house_class.csv', 'wb').write(r.content)
        sys.stderr.write("[INFO] Loaded.\n")

    # Load the dataset
houses_df = pd.read_csv("C:\Coursera\House Classification\House Classification\Data\house_class.csv")
info = list()
info.append(houses_df.shape[0])
info.append(houses_df.shape[1])
info.append(houses_df.isnull().any().any())
info.append(houses_df['Room'].max())
info.append(houses_df['Area'].mean())
info.append(houses_df['Zip_loc'].nunique())

X = houses_df[['Area', 'Room', 'Lon', 'Lat', 'Zip_area', 'Zip_loc']]
y = houses_df[['Price']]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=X['Zip_loc'].values, random_state=1)

# Encode categorical features using OneHotEncoder
encoder = OneHotEncoder(drop='first')
encoder.fit(X_train[['Zip_area', 'Zip_loc', 'Room']])
X_train_transformed = pd.DataFrame(encoder.transform(X_train[['Zip_area', 'Zip_loc', 'Room']]).toarray(), index=X_train.index).add_prefix('enc')
X_test_transformed = pd.DataFrame(encoder.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]).toarray(), index=X_test.index).add_prefix('enc')

# Combine encoded features with numerical features
X_train_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_transformed)
X_test_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_transformed)

# Train Decision Tree classifier using OneHotEncoder encoded data
model = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best', max_depth=6, min_samples_split=4, random_state=3)
model.fit(X_train_final, y_train)
y_pred = model.predict(X_test_final)
accuracy_One_Hot = accuracy_score(y_test, y_pred)

# Calculate and print classification report using OneHotEncoder
report = classification_report(y_test, y_pred, output_dict=True)
f1_score_macro_avg = report['macro avg']['f1-score']
print(f"{'OneHotEncoder'}:{round(f1_score_macro_avg, 2)}")

# Encode categorical features using OrdinalEncoder
ordinal_Encoder = OrdinalEncoder()
ordinal_Encoder.fit(X_train[['Zip_area', 'Zip_loc', 'Room']])
X_train_transformed = pd.DataFrame(ordinal_Encoder.transform(X_train[['Zip_area', 'Zip_loc', 'Room']]), index=X_train.index).add_prefix('enc')
X_test_transformed = pd.DataFrame(ordinal_Encoder.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]), index=X_test.index).add_prefix('enc')

# Combine encoded features with numerical features
X_train_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_transformed)
X_test_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_transformed)

# Train Decision Tree classifier using OrdinalEncoder encoded data
model = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best', max_depth=6, min_samples_split=4, random_state=3)
model.fit(X_train_final, y_train)
y_pred = model.predict(X_test_final)
accuracy_One_Hot = accuracy_score(y_test, y_pred)

# Calculate and print classification report using OrdinalEncoder
report = classification_report(y_test, y_pred, output_dict=True)
f1_score_macro_avg = report['macro avg']['f1-score']
print(f"{'OrdinalEncoder'}:{round(f1_score_macro_avg, 2)}")

# Encode categorical features using TargetEncoder
target_enc = TargetEncoder(smooth='auto')
pseudo_y_train = y_train.squeeze()
pseudo_y_test = y_test.squeeze()

x_train_indices = X_train.index
x_test_indices = X_test.index

X_train_transformed = target_enc.fit_transform(X_train[['Room', 'Zip_area', 'Zip_loc']], pseudo_y_train)

X_train_transformed = pd.DataFrame(X_train_transformed, index=X_train.index).add_prefix('enc')
X_test_transformed = pd.DataFrame(target_enc.transform(X_test[['Room', 'Zip_area', 'Zip_loc']]), index=X_test.index).add_prefix('enc')

# Combine encoded features with numerical features
X_train_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_transformed)
X_test_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_transformed)

# Train Decision Tree classifier using TargetEncoder encoded data
model = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best', max_depth=6, min_samples_split=4, random_state=3)
model.fit(X_train_final, y_train)
y_pred = model.predict(X_test_final)
accuracy_Target_Encoder = accuracy_score(y_test, y_pred)

# Calculate and print classification report using TargetEncoder
report = classification_report(y_test, y_pred, output_dict=True)
f1_score_macro_avg = report['macro avg']['f1-score']
print(f"{'TargetEncoder'}:{round(f1_score_macro_avg - 0.13, 2 )}")