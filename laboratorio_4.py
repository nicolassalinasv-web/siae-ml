from base import *
X,y=datos_clasificacion()
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
for profundidad in [2,4,8,None]:
    m=DecisionTreeClassifier(max_depth=profundidad,random_state=SEED).fit(Xtr,ytr)
    print('Profundidad',profundidad,'train',round(m.score(Xtr,ytr),3),'test',round(m.score(Xte,yte),3))
