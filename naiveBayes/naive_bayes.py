#-------------------------------------------------------------------------
# AUTHOR: Joseph Scott
# FILENAME: naive_bayes.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#Reading the training data in a csv file
#--> add your Python code here
db = []

with open('weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
outlook = {'Sunny': 1, 'Overcast': 2, 'Rain': 3}
temperature = {'Hot': 1, 'Mild': 2, 'Cool': 3}
humidity = {'High': 1, 'Normal': 2}
wind = {'Strong': 1, 'Weak': 2}

X = []

for row in db:
    X.append([outlook[row[1]], temperature[row[2]], humidity[row[3]], wind[row[4]]])


#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
class_name = {'Yes': 1, 'No': 2}

Y = []

for row in db:
    Y.append(class_name[row[5]])

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
#--> add your Python code here
dbTest = []
with open('weather_test.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         dbTest.append (row)


#Printing the header os the solution
#--> add your Python code here
print(f'{'Day':<8}{'Outlook':<10}{'Temperature':<13}{'Humidity':<10}{'Wind':<8}{'PlayTennis':<12}{'Confidence'}')

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for row in dbTest:
    testSample = [outlook[row[1]], temperature[row[2]], humidity[row[3]], wind[row[4]]]
    predicted_proba = clf.predict_proba([testSample])[0]
    predicted_class = clf.predict([testSample])[0]

    confidence = max(predicted_proba)
    predicted_label = 'Yes' if predicted_class == 1 else 'No'

    if confidence >= 0.75:
        print(f'{row[0]:<8}{row[1]:<10}{row[2]:<13}{row[3]:<10}{row[4]:<8}{predicted_label:<12}{confidence:.2f}')

