import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
data=pd.read_csv(r'C:\Users\thaku\ml\data.csv')
data =data.drop('Phone Name',axis=1)
x=data[['RAM','ROM/Storage','Back/Rare Camera','Front Camera','Battery']]
y=data['Price in INR']
x_train,x_test,y_train,y_test=train_test_split(x,y ,random_state=100,test_size=0.2)
model = LinearRegression()
model.fit(x_train,y_train)
pickle.dump(model,open('model.pkl','wb'))

