# Importing dependencies
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, recall_score, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import precision_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.utils import resample

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import RocCurveDisplay

# Define column order
column_order = ["age", "job", "marital", "education", "default", "housing", "loan",
                "contact", "month", "day_of_week", "duration", "campaign", "pdays",
                "previous", "poutcome", "emp.var.rate", "cons.price.idx",
                "cons.conf.idx", "euribor3m", "nr.employed", "y"]

# Load the dataset
data = pd.read_csv("bank-additional.csv", delimiter=";")
data = data[column_order]

# Display dataset info
print("Data Shape:", data.shape)
print(data.head())
print(data.describe())
print(data.info())
print(data.apply(lambda x: len(x.unique()))) # to calculate the number of unique values in each column of a DataFrame
print(data.isnull().sum())

# Check for duplicated data
duplicated_data = data.duplicated()
print(data[duplicated_data])

duplicatedRows = sum(duplicated_data)
print("Number of Duplicated Rows:", duplicatedRows)

# Define categories for ordinal encoding
month_categories = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
day_categories = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

# Define categorical and numerical columns
categorical_columns = ["job", "marital", "default", "housing", "loan", "contact", "education"]
numerical_columns = ['age', "duration", 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

# Define the preprocessing steps
preprocess = ColumnTransformer([
    ("month_encoded", OrdinalEncoder(categories=[month_categories]), ['month']),
    ("day_encoded", OrdinalEncoder(categories=[day_categories]), ['day_of_week']),
    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
    ("numeric_scaler", StandardScaler(), numerical_columns)
])

# Split the dataset into majority and minority classes
data_majority = data[data.y == 'no']  # Selecting all instances where the target is 'no'
data_minority = data[data.y == 'yes'] # Selecting all instances where the target is 'yes'

# Oversample the minority class 'yes' to 50% of the majority class size
data_minority_oversampled = resample(data_minority,
                                   replace=True,
                                   n_samples=int(len(data_majority) * 0.5),  # Oversample to 50% of the majority class
                                   random_state=123)

# Undersample the majority class 'no' to 50% of its original size
data_majority_undersampled = resample(data_majority,
                                   replace=True,
                                   n_samples=int(len(data_majority) * 0.5),  # Undersample to 50% of the majority class
                                   random_state=123)

# Combining the oversampled minority class and the undersampled majority class
data_resampled = pd.concat([data_majority_undersampled, data_minority_oversampled])
X = data_resampled.drop('y', axis=1)
y = data_resampled['y']
yes_count = y.value_counts()
print(yes_count)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)

# Create and train the Gradient Boosting model
model = GradientBoostingClassifier()
pipeline = make_pipeline(preprocess, model)
pipeline.fit(x_train, y_train)

# Make predictions
predictions = pipeline.predict(x_test)

# Evaluate the model
report = classification_report(y_test, predictions, target_names=['no', 'yes'])
print(report)

# Save the model
pickle.dump(pipeline, open('gbc_model.pkl', 'wb'))

# Load the model and make a prediction
loaded_model = pickle.load(open('gbc_model.pkl', 'rb'))

# Example prediction
example_data = pd.DataFrame({
    'age': [30],
    'job': ['blue-collar'],
    'marital': ['married'],
    'education': ['basic.9y'],
    'default': ['no'],
    'housing': ['yes'],
    'loan': ['no'],
    'contact': ['cellular'],
    'month': ['may'],
    'day_of_week': ['fri'],
    'duration': [2],
    'campaign': [2],
    'pdays': [999],
    'previous': [0],
    'poutcome': ['nonexistent'],
    'emp.var.rate': [-1.8],
    'cons.price.idx': [92.893],
    'cons.conf.idx': [-46.2],
    'euribor3m': [1.313],
    'nr.employed': [5099.1],
})

pred = loaded_model.predict(example_data)
print(pred[0])