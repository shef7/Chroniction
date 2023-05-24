# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'breastcancer.pkl'
classifier = pickle.load(open(filename, 'rb'))

filename = 'lungcancer.pkl'
classifier3 = pickle.load(open(filename, 'rb'))

filename = 'diabetes.pkl'
classifier1 = pickle.load(open(filename, 'rb'))

filename = 'heart.pkl'
classifier2 = pickle.load(open(filename, 'rb'))

app = Flask(__name__)
@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetes():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/lung", methods=['GET', 'POST'])
def lungPage():
    return render_template('lung.html')


@app.route('/predict', methods=['POST'])
def predictPage():
    if request.method == 'POST':
        a = request.form['pregnancies']
        b = request.form['glucose']
        c = request.form['bloodpressure']
        d = request.form['skinthickness']
        e = request.form['insulin']
        f = request.form['bmi']
        g = request.form['dpf']
        h = request.form['age']
        
        data = np.array([[a,b,c,d,e,f,g,h]])
        pred = classifier1.predict(data)
        
        return render_template('predict.html', pred= pred )

        
@app.route('/predict1', methods=['POST'])
def predictPage1():
    if request.method == 'POST':
        a= request.form['radius_mean']
        b = request.form['texture_mean']
        c = request.form['perimeter_mean']
        d = request.form['area_mean']
        e= request.form['smoothness_mean']
        f = request.form['compactness_mean']
        g = request.form['concavity_mean']
        h = request.form['concave_points_mean']
        i = request.form['symmetry_mean']
        j = request.form['radius_se']
        k = request.form['perimeter_se']
        z = request.form['area_se']
        l = request.form['compactness_se']
        m = request.form['concavity_se']
        n = request.form['concave_points_se']
        o = request.form['fractal_dimension_se']
        p = request.form['radius_worst']
        q = request.form['texture_worst']
        r = request.form['perimeter_worst']
        s = request.form['area_worst']
        t = request.form['smoothness_worst']
        u = request.form['compactness_worst']
        v = request.form['concavity_worst']
        w = request.form['concave_points_worst']
        x = request.form['symmetry_worst']
        y = request.form['fractal_dimension_worst']

        data = np.array([[a,b,c,d,e,f,g,h,i,j,k,z,l,m,n,o,p,q,r,s,t,u,v,w,x,y]])
        pred1 = classifier.predict(data)
        
        return render_template('predict1.html', pred1=pred1 )

@app.route('/predict2', methods=['POST'])
def predictPage2():
    if request.method == 'POST':
        a = request.form['age']
        b = request.form['sex']
        c = request.form['cp']
        d = request.form['trestbps']
        e = request.form['chol']
        f = request.form['fbs']
        g = request.form['restecg']
        h = request.form['thalach']
        i = request.form['exang']
        j = request.form['oldpeak']
        k = request.form['slope']
        l = request.form['ca']
        m = request.form['thal']
        
        data = np.array([[a,b,c,d,e,f,g,h,i,j,k,l,m]])
        pred2 = classifier2.predict(data)
        
        return render_template('predict2.html', pred2= pred2 )


@app.route('/predict3', methods=['POST'])
def predictPage3():
    if request.method == 'POST':
        age = request.form['Age']
        sm = request.form['Smokes']
        area = request.form['AreaQ']
        al = request.form['Alkhol']
        
        data = np.array([[age,sm,area,al]])
        pred3 = classifier3.predict(data)
        
        return render_template('predict3.html', pred3= pred3 )


if __name__ == '__main__':
	app.run(debug = True)