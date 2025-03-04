#-------------------------------------------------------------------------
# AUTHOR: Joseph Scott
# FILENAME: knn.py
# SPECIFICATION: This program tests each instance of the training data to predict the classification of that instance and calculate the error rate.
#                The data being tested is testing how likely an email is to be spam depending on the frequency of different words in the email.
# FOR: CS 4210- Assignment #2
# TIME SPENT: This program took me about 2 hours to complete
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

errors = 0
total_instances = len(db)

#Loop your data to allow each instance to be your test set
for i in db:
    X = []
    Y = []

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    for val in db:
        if val != i:
            fval = []
            for integer in range(len(val)-1):
                fval.append(float(val[integer]))
            X.append(fval[0:20])


    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    class_name = {'ham': 1.0, 'spam': 2.0}
    for val in db:
        if val != i:
            Y.append(class_name[val[20]])


    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = [float(val) for val in i[0:20]]
    testLabel = i[20]

    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted != class_name[testLabel]:
        errors += 1

#Print the error rate
#--> add your Python code here
error_rate = errors / total_instances
print(f'Error rate: {error_rate:.2f}')






