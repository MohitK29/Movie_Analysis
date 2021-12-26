# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 00:35:42 2021

@author: mohit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


#%% Load Data

movie_df=pd.read_csv('movieReplicationSet.csv')
for i in range(0,477):
    movie_df.iloc[:,i].fillna(value=movie_df.iloc[:,i].median(), inplace=True)

                        
df=movie_df.iloc[:, 421:475]
r = np.corrcoef(df,rowvar=False)

zscoredData = stats.zscore(df)
pca = PCA().fit(zscoredData)
eigValues = pca.explained_variance_
loadings_v = pca.components_
u = pca.fit_transform(zscoredData)
covarExplained = (sum(eigValues[:10])/sum(eigValues))*100
print(eigValues)

#%% scree plot:
#Kaiser criterion

numPredictors = 54
plt.bar(np.linspace(1,numPredictors,numPredictors),eigValues)
plt.axhline(y=1, color='r', linestyle='-')
plt.title('Scree plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalues')

#%% 7 Interpreting the factors 

maxFeat=[]
for i in range(11):
    maxFeat.append(np.argmax(loadings_v[i,:]*-1))
    plt.figure()
    plt.bar(np.linspace(1,54,54),loadings_v[i,:]*-1)
    plt.xlabel('Questions')
    plt.ylabel('Loading')

for i in maxFeat:
    print(df.columns[i])

#%% Q2

col=df.columns
for i in range(2):
    for j in range(i+1, 11):
        plt.figure()
        f1=col[maxFeat[i]]
        f2=col[maxFeat[j]]
        plt.scatter(u[:,i], u[:,j])
        plt.xlabel(f1)
        plt.ylabel(f2)
        plt.show()

#%% Clustering

X = (np.array(u[:, 0:10]))
numClusters = 9 # how many clusters are we looping over? (from 2 to 10)
Q = np.empty([numClusters,1]) # init container to store sums
Q[:] = np.NaN # convert to NaN

ans=[]
plt.figure()
# Compute kMeans:
for ii in range(2, 11): # Loop through each cluster (from 2 to 10!)
    kMeans = KMeans(n_clusters = int(ii)).fit(X) # compute kmeans
    cId = kMeans.labels_ # vector of cluster IDs that the row belongs to
    cCoords = kMeans.cluster_centers_ # coordinate location for center of each cluster
    my_dict = {cCoords[i, 0]: np.where(cId== i)[0] for i in range(kMeans.n_clusters)}
    ans.append(my_dict)
    s = silhouette_samples(X,cId) # compute the mean silhouette coefficient of all samples
    # print(s.shape)
    Q[ii-2] = sum(s) # take sum
    # Plot data:
    plt.subplot(3,3,ii-1) 
    plt.hist(s,bins=100) 
    plt.xlim(-0.2,1)
    plt.ylim(0,500)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    plt.title('Sum: {}'.format(int(Q[ii-2])))
    
plt.figure()
plt.plot(np.linspace(2,10,numClusters),Q)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of silhouette scores')
plt.show()


#%% Now let's plot and color code the data
    
c= (np.argmax(Q)+2)
plt.figure()
indexVector = np.linspace(1,c,c) 
for ii in indexVector:
    plotIndex = np.argwhere(cId == int(ii-1))
    plt.plot(u[plotIndex,0],u[plotIndex,1],'o',markersize=1)
    plt.plot(cCoords[int(ii-1),0],cCoords[int(ii-1),1],'o',markersize=5,color='black')  
    plt.xlabel('Features')
    plt.ylabel('Loadings')

#%% Classification:

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc


Y=np.array(movie_df.iloc[:, 0:400])
Y=np.round(Y)
X_train,X_test,y_train,y_test=train_test_split(X,Y)

mod=RandomForestClassifier()
mod.fit(X_train,y_train)
pred_prob = mod.predict_proba(X_test)
y_hat=mod.predict(X_test)
accuracy_score = []
for tru,pred in zip (y_test, y_hat):
     accuracy_score.append(f1_score(tru,pred,average='micro' ))
print(np.mean(accuracy_score)*100)

y_test=np.where(y_test > 3, 1, 0)
y_hat=np.where(y_hat > 3, 1, 0)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(400):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_hat[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_hat.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2

for i in list(range(183,192)):
	plt.plot(
	    fpr[i],
	    tpr[i],
	    label="class %d, ROC curve (auc = %0.2f)" % (i, roc_auc[i]),
	)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC and AUC on three 15 classes of movie dataset.")
plt.show()

#%% NN

import torch
import torch.optim as optim

from torch import nn

movie=np.array(movie_df)

class neural_network(torch.nn.Module):

  def __init__(self):
    super().__init__()
    # TODO

    self.fc1 = nn.Linear(77,200)

    self.fc2 = nn.Linear(200,400)

    self.fc3 = nn.Linear(400,600)

    self.fc4=nn.Linear(600,400)
    
    self.dropout = nn.Dropout(0.2)
    self.relu = nn.ReLU()
    self.sig=nn.Sigmoid()

  def forward(self, x):

    # TODO
    out=self.relu(self.fc1(x))
    out=self.dropout(out)
    out=self.fc2(out)
    out=self.relu(out)
    # out=self.dropout(out)
    out=self.relu(self.fc3(out))
    out=self.dropout(out)
    out=self.relu(self.fc4(out))
    return out
    pass

train_data, test_data = torch.utils.data.random_split(movie, [1000, 97])
train_data_loader=torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4)
test_data_loader=torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=4)

#training
model=neural_network()
criterion = nn.MSELoss()
optimizer = optim.Adam(params = model.parameters(),lr=0.0001)

for val in train_data:
    input=torch.tensor(val[400:]).float()
    labels=torch.tensor(val[:400]).float()
    labels=torch.round(labels)
    pred=model(input)
    pred=pred.unsqueeze(0)
    lables=labels.unsqueeze(0).unsqueeze(0)
    optimizer.zero_grad()
    loss=criterion(pred,labels)
    loss.backward()
    optimizer.step()
  
#testing
c=0
accuracy=[]
for val in test_data:
  c=0
  input=torch.tensor(val[400:]).float()
  labels=torch.tensor(val[:400]).float().unsqueeze(0)
  pred=model(input)
  pred=torch.round(pred)
  labels=torch.round(labels)
  diff=pred-labels
  c=400-torch.count_nonzero(diff)
  accuracy.append(c/400)
accuracy=torch.tensor(accuracy)
print('Accuracy:-',torch.mean(accuracy))
