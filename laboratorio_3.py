from base import *
X,y=datos_clasificacion()
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
for nombre,m in [('Arbol',DecisionTreeClassifier(max_depth=4,random_state=SEED)),('Bosque',RandomForestClassifier(n_estimators=100,random_state=SEED))]:
    m.fit(Xtr,ytr)
    print(nombre,'accuracy',round(accuracy_score(yte,m.predict(Xte)),3),'importancias',np.round(m.feature_importances_,3))
Z=datos_segmentacion()
coords=PCA(n_components=2).fit_transform(StandardScaler().fit_transform(Z))
print('PCA forma:',coords.shape)
