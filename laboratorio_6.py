from base import *
Z=datos_segmentacion()
rng=np.random.default_rng(SEED)
anomalias=pd.DataFrame(rng.uniform(-12,12,size=(12,2)),columns=Z.columns)
Z=pd.concat([Z,anomalias],ignore_index=True)
m=Pipeline([('escala',StandardScaler()),('modelo',IsolationForest(contamination=.04,random_state=SEED))]).fit(Z)
marcas=m.predict(Z)
print('Observaciones',len(Z),'Anomalias detectadas',sum(marcas==-1))
print(Z.loc[marcas==-1].head())
