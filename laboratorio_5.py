from base import *
X,y=datos_clasificacion()
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
modelos={'SVM':Pipeline([('escala',StandardScaler()),('modelo',SVC())]),'Naive Bayes':GaussianNB(),'KNN':Pipeline([('escala',StandardScaler()),('modelo',KNeighborsClassifier(n_neighbors=5))])}
for nombre,m in modelos.items():
    m.fit(Xtr,ytr); print(nombre,round(m.score(Xte,yte),3))
Z=StandardScaler().fit_transform(datos_segmentacion())
for nombre,m in [('KMeans',KMeans(n_clusters=3,n_init=10,random_state=SEED)),('Jerarquico',AgglomerativeClustering(n_clusters=3))]:
    labels=m.fit_predict(Z);print(nombre,'silhouette',round(silhouette_score(Z,labels),3))
print('t-SNE solo visualizacion:',TSNE(n_components=2,random_state=SEED,perplexity=30).fit_transform(Z).shape)
