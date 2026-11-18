from base import *
X,y=datos_clasificacion()
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
modelos={'Logistica':Pipeline([('escala',StandardScaler()),('modelo',LogisticRegression(max_iter=1000))]),'Bosque':RandomForestClassifier(n_estimators=100,random_state=SEED),'SVM':Pipeline([('escala',StandardScaler()),('modelo',SVC())])}
resultados=[]
for nombre,m in modelos.items():
    cv=cross_val_score(m,Xtr,ytr,cv=5,scoring='accuracy').mean()
    m.fit(Xtr,ytr)
    resultados.append({'modelo':nombre,'cv_train':round(cv,3),'test':round(m.score(Xte,yte),3)})
print(pd.DataFrame(resultados).to_string(index=False))
print('Interpretar rendimiento, costo de errores y limitaciones antes de recomendar.')
