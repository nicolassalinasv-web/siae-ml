from base import *
X,y=datos_regresion()
print('Dimensiones:',X.shape,'Nulos:',X.isna().sum().sum(),'Duplicados:',X.duplicated().sum())
print(X.describe().round(2))
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=SEED)
m=LinearRegression().fit(Xtr,ytr)
p=m.predict(Xte)
print('MAE:',round(mean_absolute_error(yte,p),2),'R2:',round(r2_score(yte,p),3))
