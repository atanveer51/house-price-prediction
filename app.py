from flask import Flask,render_template,redirect,request
import pickle
import numpy as np
app = Flask(__name__,template_folder='templates')
model=pickle.load(open('houseprice.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template('index.html')
@app.route('/predict',methods=['GET', 'POST'])
def predict():
    data = [int(x) for x in request.form.values() if len(x)>0]
    print(data)
    final_feature=[np.array(data)]
    pre=model.predict(final_feature)
    print(pre)
    print(data)
    output=round(pre[0],2)
    return render_template('index.html',pre=output)

if __name__ == '__main__':
    app.run(debug=True)
