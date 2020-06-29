import numpy as np
import pandas as pd
from sklearn import preprocessing,neighbors,model_selection

##data resource file
df=pd.read_csv("breast-cancer-wisconsin.txt")

##data preprocessing
df.replace('?',-99999,inplace=True)
df.drop(["x1"],1,inplace=True)
x=np.array(df.drop(["y11"],1))
y=np.array(df['y11'])

##split the data into train and test along with target y set
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2)
clf=neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
print(accuracy)

#prediction part
ex=np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
ex=ex.reshape(len(ex),-1)
prediction=clf.predict(ex)
print(prediction)