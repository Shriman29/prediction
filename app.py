import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('pharm.html')
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    print(int_features)
    c = int_features[0]
    car = str(c)
    
    
    
    import pandas as pd
    import numpy as np 
    import matplotlib.pyplot as plt
    from sklearn import datasets, linear_model
    from sklearn.model_selection import train_test_split

    #read the file
    df = pd.read_csv('Pharmacy.csv')
    college=np.unique(df['College'])
    clg_code=[]
    for i in range(len(college)):
        clg_code.append(i+1)
    # clg_code
    df['College']=df['College'].replace(college,clg_code)
    bak_college=np.array(df['College'])


    # Using only one feature
    x = df.iloc[:, 4].values
    y = df.iloc[:, 5].values

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=0)

    x_train= x_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)

    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    col=df.columns.tolist()[4:5]
    print(col)
    usrip=[]
    for i in col:
        print("==================================================")
        usrip.append(eval(car))

    userpreddt=model.predict([usrip])

    print("You may have change to get entrance in: ",college[clg_code.index(int(userpreddt[0]))])


    # return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))
    return render_template('pharm.html', prediction_text='You may have change to get entrance in:  {}'.format(college[clg_code.index(int(userpreddt[0]))]))



if __name__ == "__main__":
    app.run(debug=True)