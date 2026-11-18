"""Experimento de Clase 1. Ejecutar desde la carpeta del proyecto."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base import *
X,y=datos_regresion()
print('Dimensiones:',X.shape,'Nulos:',X.isna().sum().sum(),'Duplicados:',X.duplicated().sum())
print(X.describe().round(2))
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3 if '--prueba-30' in sys.argv[1:] else .2,random_state=SEED)
m=LinearRegression().fit(Xtr,ytr)
p=m.predict(Xte)
print('MAE:',round(mean_absolute_error(yte,p),2),'R2:',round(r2_score(yte,p),3))

print('Entrenamiento:', len(Xtr), 'Prueba:', len(Xte), 'Semilla:', SEED)
if '--referencia' in sys.argv[1:]:
    p_base = np.full(len(yte), ytr.mean())
    print('MAE referencia media:', round(mean_absolute_error(yte, p_base), 2))
    print('Media de ytr:', round(ytr.mean(), 4))
    reduccion = 100 * (1 - mean_absolute_error(yte, p) / mean_absolute_error(yte, p_base))
    print('Reduccion del MAE (%):', round(reduccion, 2))
