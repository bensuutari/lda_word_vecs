
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from gensim.models import Doc2Vec
import numpy as np
import matplotlib.pyplot as plt
import pickle
docvecmodel=Doc2Vec.load(os.getcwd()+'/models/docvecmodel.model')
docvectors=np.asarray(docvecmodel.docvecs)
dim_reduc_method='PCA'
labels=['dem','dem','repub','repub']
data=docvectors
SVC_classifier=SVC()
#SVC_classifier.fit(docvectors,labels)
print('start KMeans...')
docvec_kmeans=KMeans(n_clusters=3, random_state=0).fit(docvectors)
print('Kmeans complete!')


lowddoc2vec=pickle.load(open(os.getcwd()+'/graphs/low_D_doc2vec/doc2vec_2d_'+dim_reduc_method,'rb'))
for i in range(0,lowddoc2vec.shape[0]):
	if docvec_kmeans.labels_[i]==0:
		plt.plot(lowddoc2vec[i,0],lowddoc2vec[i,1],'ok')
	elif docvec_kmeans.labels_[i]==1:
		plt.plot(lowddoc2vec[i,0],lowddoc2vec[i,1],'or')
	elif docvec_kmeans.labels_[i]==2:
		plt.plot(lowddoc2vec[i,0],lowddoc2vec[i,1],'ob')
plt.show()
