#projet unsupervised - ZIDAN Lama - DIAGNE Awa Syr 

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
#Import Libraries
import pandas as pd #pour charger, d’aligner et de manipuler les données
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split#diviser le dataset en train et test
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score #scores du modele SVM
from sklearn import metrics
from sklearn.decomposition import KernelPCA# ACP à noyaux
from sklearn.preprocessing import StandardScaler, KernelCenterer #pour scaler les donnés, Pour Centrer une matrice de noyau arbitraire
from sklearn.svm import SVC #pour le modele SVM
from sklearn.model_selection import GridSearchCV #pour appliquer la methode cross validation

import seaborn as sns

from numpy import sqrt #pour utiliser racine carré
import numpy as np  # traitement des arrays numériques

import matplotlib.pyplot as plt #tracage des graphiques
from matplotlib.collections import LineCollection

from scipy.spatial.distance import pdist, squareform #calcul distance pour kernel
from scipy import exp  #importer l'exponentielle pour kernel
from scipy.linalg import eigh #Résoudre un problème de valeurs propres généralisé pour une matrice symétrique réelle.

import random #tirage améatoire

#Chargement et affichage de données: 
df= pd.read_csv("C:\\Users\\Lama\Desktop\\data_tot2.csv",decimal=".", index_col=0)
print("la base de données :\n",df.head())

#pre-traitement des donnees:
 
# supprimer les colonnes suivantes:
drop_list=['CLINIC.OS_MONTHS','Row.names','CLINIC.study']
df.drop(drop_list,axis=1,inplace=True)

#afficher les valeurs manquantes du dataset apres suppression de certains colonnes:
print('Le nombre de valeurs manquantes est :', df.isnull().sum().sum())
#nous avons 7 valeurs manquantes dans la colonne  'CLINIC.OS_STATUS' qualitative 0 et 1 
#suprimmer les 7 valeurs manquants:
df.dropna(inplace=True)
print('Le nombre de valeurs manquantes est :', df.isnull().sum().sum())

print('Les dimensions de notre base est:',df.shape)
print('La répartition de deux madalités de ma variable cible: \n',df['CLINIC.OS_STATUS'].value_counts())
#on trouve que les données sont non equilibrées (mais ceci n'est pas dans le cadre du projet donc on va pas le traiter)

#recoder les modalités en 0 et 1: 
df['CLINIC.OS_STATUS']= df['CLINIC.OS_STATUS'].map({'0:LIVING':0,'1:DECEASED':1})
y=df['CLINIC.OS_STATUS']
X=df.drop(['CLINIC.OS_STATUS'],axis=1)

#centrer et reduire les données:
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_X = scaler.fit_transform(X)
scaled_X=pd.DataFrame(scaled_X, columns=X.columns)
print(scaled_X.head(3))

scaled_X2=scaled_X/sqrt(401)

#status sera les labels des observations sur le graphique de l'ACP par la suite
status = df["CLINIC.OS_STATUS"]
#convertir le type de la colonne status de series à list pour pouvoir les manipuler par la suite:
status = status.tolist()
print('Le type de la variable status est: ',type(status))

features=X.columns
print('Les features sont: \n',features)

#afficher la matrice de correlation entre les gènes:
pw_corr=pd.DataFrame(scaled_X).corr().round(3)
print('La table de corrélations: \n',pw_corr)

#Et le Heat map correspondant:
sns.heatmap(pw_corr,cmap='vlag')

#Appliquer le modele de prediction SVM sur les variables initiales(sans reduction de dimensions):
X_train,X_test,y_train,y_test=train_test_split(scaled_X,y,test_size=0.2,random_state=0)
SV=SVC()
# training the model:
SV.fit(X_train,y_train)
y_pred = SV.predict(X_test)
print('Training score:',(SV.score(X_train,y_train)*100).round(1),'%')
print('test score:',(SV.score(X_test,y_test)*100).round(1),'%')
#train score=85.3%     test_score=64.2%
#21.1% de gap donc le modele est overfit 


#ACP:
n_components=33
pca_model = PCA(n_components)
pca_10 = pca_model.fit_transform(scaled_X2)
np.sum(pca_model.explained_variance_ratio_)

#afficher la variance expliqué arrondi à 2 chiffres pour chaque component :
var_ratio = (pca_model.explained_variance_ratio_*100).round(2)
print('La ratio de la variance expliquée pour chaque composant est :\n',var_ratio)

#la variance cumulée
var_ratio_cum = var_ratio.cumsum().round()
print('La ratio de la variance expliquée cumulée est :\n',var_ratio_cum)

#créer une liste a de ligne egale à la dimension de l'acp
#cette liste sera utilisée pour afficher le graphique ci-apres: 
x_list = range(1, n_components+1)
#histogramme des variances expliqué pour chaque dimension:
#on voit bien que les 3 premieres sont le plus expliquatives 
plt.bar(x_list, var_ratio)
plt.plot(x_list, var_ratio_cum,c="red",marker='o')
plt.xlabel("rang de l'axe d'inertie")
plt.ylabel("pourcentage d'inertie")
plt.title("Eboulis des valeurs propres")
plt.show(block=False)

#afficher les valeurs singuliers de l'ACP:
print('Les valeurs singuliers de l''ACP sont :\n',(pca_model.singular_values_).round(2))

PCA_X=pd.DataFrame(pca_10)
#print(PCA_X)

#Affichons les facteurs principaux:
pcs = pca_model.components_
pcs = pd.DataFrame(pcs)
pcs.columns =features
pcs.index = [f"PCA_X{i}" for i in x_list]
pcs=pcs.round(2)
print(pcs.T)

#graphique: 
#On va afficher le cercle de corrélations en utilisant les deux premieres composantes:
#Définissons nos axes x et y: 
x1, y1 = 0,1
fig, ax = plt.subplots(figsize=(10, 9))
for i in range(0, pca_model.components_.shape[1]):
    ax.arrow(0,
             0,  # Start the arrow at the origin
            pca_model.components_[0, i],  #0 for PC1
            pca_model.components_[1, i],  #1 for PC2
             head_width=0.07,
             head_length=0.07, 
             width=0.02,              )
    plt.text(pca_model.components_[0, i] + 0.05,
             pca_model.components_[1, i] + 0.05,
             features[i])
# affichage des lignes horizontales et verticales
plt.plot([-1, 1], [0, 0], color='grey', ls='--')
plt.plot([0, 0], [-1, 1], color='grey', ls='--')
# nom des axes, avec le pourcentage d'inertie expliqué
plt.xlabel('F{} ({}%)'.format(x1+1, round(100*pca_model.explained_variance_ratio_[x1],1)))
plt.ylabel('F{} ({}%)'.format(y1+1, round(100*pca_model.explained_variance_ratio_[y1],1)))
plt.title("Cercle des corrélations (F{} et F{})".format(x1+1, y1+1))
an = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(an), np.sin(an)) 
plt.axis('equal')
plt.show(block=False)

#Affichage des individus projetés sur le premier plan factoriel: 
plt.figure(figsize=(14,14))#insancier une figure vide
plt.scatter(pca_10[:,0],pca_10[:,1])
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.axvline(x=0,color="black")#ajouter une ligne verticale
plt.axhline(y=0,color="black")#ajouter une ligne horizentale
for i in range(len(status)):
    plt.annotate(status[i], (pca_10[i,0], pca_10[i,1]))#ajouter le statut de vie (0 ou 1)aux individus
plt.show()



# appliquer SVM sur les variables reduites issues de l'ACP:
X_train,X_test,y_train,y_test=train_test_split(PCA_X,y,test_size=0.2,random_state=0)
SV=SVC()
# training the model
SV.fit(X_train,y_train)
y_pred = SV.predict(X_test)
print('Training score:',(SV.score(X_train,y_train)*100).round(1),'%')
print('test score:',(SV.score(X_test,y_test)*100).round(1),'%')
#Training score: 80.3 %
#test score: 69.1 %

#Tuning the hyperparameters de SVM:
my_param_grid = {'C': [0.01,0.1,1,10,50], 'gamma': ['scale','auto',0.01,0.001,], 'kernel': ['rbf','linear']}
grid = GridSearchCV(estimator=SVC(),param_grid= my_param_grid, refit = True, verbose=2, cv=5,scoring='accuracy' )
grid.fit(X_train,y_train)
print(grid.best_params_)
print('Training score after tuning:',(grid.score(X_train,y_train)*100).round(1),'%')
print('test score after tuning:',(grid.score(X_test,y_test)*100).round(1),'%')
#Training score: 80.3 %
#test score: 69.1 %
#meme resultats! le modele ne peut pas encore ameliorer

#ACP à noyaux:
gamma=0.01
n_components=33
#calculer la distance Euclidienne au carré pour tous les paires de points dans le dataset:
sq_dists = pdist(scaled_X2, 'sqeuclidean')
#convertir les distances en une matrice symetrique: mxm
mat_sq_dists = squareform(sq_dists)
#calculer la matrice de kernel
K =np.exp(-gamma * mat_sq_dists)
#centrer la matrice de kernel:
N = K.shape[0]
one_n = np.ones((N,N)) / N
K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

#obtenir les valeurs propres dans l'ordre décroissante avec leurs vecteurs propres 
eigvals, eigvecs = eigh(K)
#afficher les valeurs propres:
print('Les valeurs propres de lacp à noyaux :\n',eigvals)
#afficher les vecteurs propres:
print('Les vecteurs propres de lacp à noyaux :\n',eigvecs)

#obtenir le i vecters propres qui correspondent aux i valeurs propres:
X_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))
print(X_pc)

#affichage:
kpca1 = KernelPCA(n_components=33, kernel='rbf', gamma=0.01)
with plt.style.context(('ggplot')):
    fig, ax = plt.subplots(figsize=(14, 14))
    plt.axvline(x=0,color="black")#add a vertical line
    plt.axhline(y=0,color="black")#add a horizontal line
    ax.scatter(X_pc[:,0], X_pc[:,1], s=100, edgecolors='k')   
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title('Scikit learn')
    for i in range(len(status)):
        plt.annotate(status[i], (X_pc[i,0], X_pc[i,1]))#add the country name to each point
    plt.show()

#Appliquer SVM sur les variables reduites issues de l'ACP à noyaux:
X_train,X_test,y_train,y_test=train_test_split(X_pc,y,test_size=0.2,random_state=0)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
SV=SVC()

# training the model
SV.fit(X_train,y_train)
y_pred = SV.predict(X_test)
print('Training score:',(SV.score(X_train,y_train)*100).round(1),'%')
print('test score:',(SV.score(X_test,y_test)*100).round(1),'%')
#22 de gap==>overfitting
#Training score: 87.5 %
#test score: 65.4 %
#on va essayer de resoudre le probleme:

#tuning hyperparametres:
my_param_grid = {'C': [0.01,0.1,1,10,50], 'gamma': ['scale','auto',0.01,0.001,], 'kernel': ['rbf','linear']}
grid = GridSearchCV(estimator=SVC(),param_grid= my_param_grid, refit = True, verbose=2, cv=5,scoring='accuracy' )
grid.fit(X_train,y_train)
print(grid.best_params_)
print('Training score after tuning:',(grid.score(X_train,y_train)*100).round(1),'%')
print('test score after tuning:',(grid.score(X_test,y_test)*100).round(1),'%')
#Training score: 64.1 %
#test score: 63.0 %
#underfit!
#et le modele n'a pas amelioré!

#Aplliquer SVM sur les variables issues de la méthode Foret aléatoire: 

# #importer Random Forest, on va determiner le nombre des arbres 10000 car on a une dataset de grande dimensionalité
from sklearn.ensemble import RandomForestClassifier
model_RF=RandomForestClassifier(n_estimators=10000, n_jobs=-1)
model_RF.fit(X,y)

#enregistrer l'importance des variables dans feat_importances
feat_importances = pd.Series(model_RF.feature_importances_, index=X.columns)

# choisir le top 23 varaibles les plus importantes:
features=pd.DataFrame(feat_importances.nlargest(23))
features.reset_index(inplace=True)
print('Les features sont :   \n', features)

#pour selectionner les noms des colonnes (variables)
top_features=features['index']
print('Les top features sont : \n', top_features)

#selectionner les 23 variables et les extraire de la matrice X qui contienne 715 variables(genes)
X_RF=scaled_X[top_features]
#print(X_RF)

X_train,X_test,y_train,y_test=train_test_split(X_RF,y,test_size=0.2,random_state=18)
SV=SVC()
#entrainer le modele:
SV.fit(X_train,y_train)
y_pred = SV.predict(X_test)
print('Training score:',(SV.score(X_train,y_train)*100).round(1),'%')
print('test score:',(SV.score(X_test,y_test)*100).round(1),'%')
#random=18
#Training score: 78.4%
#test score: 72.8%

grid = GridSearchCV(estimator=SVC(),param_grid= my_param_grid, refit = True, verbose=2, cv=5,scoring='accuracy' )
grid.fit(X_train,y_train)
print('Training score:',(grid.score(X_train,y_train)*100).round(1),'%')
print('test score:',(grid.score(X_test,y_test)*100).round(1),'%')
#random=18
#Training score: 70%
#test score: 71.6 %