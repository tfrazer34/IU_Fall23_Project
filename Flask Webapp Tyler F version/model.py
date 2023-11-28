import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle


# Load csv file
# df = pd.read_csv("Iris.csv")
weather_df = pd.read_csv('seattle-weather.csv')

#Select dependant and independant variables

#X = df[['sepal_length','sepal_width','petal_length','petal_width']]
#y = df['species']


X = weather_df.drop(columns=['weather', 'date']) 
y = weather_df['weather']

#Train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Scaling features

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#model instantiation

#classifier = RandomForestClassifier()
classifier = LogisticRegression(max_iter=1000)

# fit model
classifier.fit(X_train, y_train)

# Make a pickle file for the trained model

pickle.dump(classifier, open("model.pkl", 'wb'))
