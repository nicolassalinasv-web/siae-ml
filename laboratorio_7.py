from base import *
X,y=datos_clasificacion()
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
pipe=Pipeline([('escala',StandardScaler()),('modelo',KNeighborsClassifier())])
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
busqueda=GridSearchCV(pipe,{'modelo__n_neighbors':[3,5,7,9]},cv=cv,scoring='accuracy').fit(Xtr,ytr)
print('CV mejor:',busqueda.best_params_,'CV:',round(busqueda.best_score_,3))
print('Test final:',round(busqueda.score(Xte,yte),3))
