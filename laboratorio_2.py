from base import *
X,y=datos_clasificacion()
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
m=Pipeline([('escala',StandardScaler()),('modelo',LogisticRegression(max_iter=1000))]).fit(Xtr,ytr)
print('Clasificacion:',classification_report(yte,m.predict(Xte),zero_division=0))
Z=datos_segmentacion()
cluster=Pipeline([('escala',StandardScaler()),('modelo',DBSCAN(eps=.28,min_samples=5))]).fit(Z)
labels=cluster.named_steps['modelo'].labels_
print('DBSCAN: grupos',len(set(labels))-(1 if -1 in labels else 0),'ruido',sum(labels==-1))
