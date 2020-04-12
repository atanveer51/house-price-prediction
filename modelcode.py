import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv('house_price.csv')
data = data.iloc[:,1:]
enc=LabelEncoder()
data.iloc[:,1]=enc.fit_transform(data.iloc[:,1])
print(data.head())
x=data.iloc[:,[0,1,2,3,4,5]]
y=data.Price
x=pd.get_dummies(x,drop_first=True)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
linear=LinearRegression()
linear.fit(x_train,y_train)
print(linear.predict([[23,2,0,1000,5,0]]))
pickle.dump(linear,open('houseprice.pkl','wb'))
model=pickle.load(open('houseprice.pkl','rb'))
model.predict([[1,1,3000,1,3,1]])