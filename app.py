import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn import datasets, svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import csv

import random
import pickle 

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

def get_model():
	filename = 'training.csv'
	raw_data = open(filename, 'rt')
	reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)

	filename1 = 'train_y.csv'
	raw_data1 = open(filename1, 'rt')
	reader1 = csv.reader(raw_data1, delimiter=',', quoting=csv.QUOTE_NONE)

	scaler = StandardScaler()
	train_x = (scaler.fit_transform(list(reader)[1:]))

	train_y = list(reader1)[1:]

	print("Training....")

	x, y = train_x, train_y


	clf = svm.SVC(C=7120.0, cache_size=200, class_weight=None, coef0=0.0,
	  decision_function_shape='ovr', degree=3, gamma=6.191, kernel='rbf',
	  max_iter=-1, probability=False, random_state=None, shrinking=True,
	  tol=0.001, verbose=False)

	clf.fit(x, y)
	return clf

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    clf=get_model()
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    if(clf.predict(final_features[0:]) == '0'):
        op=clf.predict(final_features[0:])
        return render_template('index.html', prediction_text='Erodable /n try increasing minerals in the soil')
    else:
	    op=clf.predict(final_features[0:])
	    return render_template('index.html', prediction_text='Good to go')
	   


if __name__ == "__main__":
    app.run(debug=True)