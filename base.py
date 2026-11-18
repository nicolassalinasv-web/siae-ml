import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_classification, make_blobs
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report, silhouette_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
SEED = 42

def datos_regresion():
    X,y=make_regression(n_samples=400,n_features=5,noise=18,random_state=SEED)
    return pd.DataFrame(X,columns=[f'indicador_{i}' for i in range(1,6)]),pd.Series(y,name='ventas_estimadas')

def datos_clasificacion():
    X,y=make_classification(n_samples=500,n_features=6,n_informative=4,n_redundant=0,random_state=SEED)
    return pd.DataFrame(X,columns=[f'indicador_{i}' for i in range(1,7)]),pd.Series(y,name='riesgo')

def datos_segmentacion():
    X,_=make_blobs(n_samples=360,centers=3,cluster_std=0.75,random_state=SEED)
    return pd.DataFrame(X,columns=['frecuencia','valor'])
