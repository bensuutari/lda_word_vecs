
from sklearn.decomposition import NMF,LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#from gensim.models import LdaModel
import pickle
import os
import numpy as np
import math
import wordcloud
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from sklearn.manifold import TSNE,MDS
from sklearn.decomposition import PCA
import random

##########Options########################
train_model=False
train_dim_red=True
dim_red_method='TSNE'#TSNE, MDS or PCA
num_dim=2#number of dimensions to reduce to in the dimensionality reduction
##########Options########################

##########Global Parameters########################
add_stopwords=['permalink','set','default','input','value','video','filter','output','frame','options','stream','new','click','window','share','subscribe','embed','reply','gold','save']
stopwords=list(set(STOPWORDS))
stopwords=stopwords+add_stopwords
no_top_words = 10
no_features=1000
no_topics=100
##########Global Parameters########################

def train_test_split_data(data,labels):
	shufflevar=list(zip(data,labels))
	random.shuffle(shufflevar)
	data,labels=zip(*shufflevar)

	train=data[0:11000]#finished here
	test=data[11000:]
	trainlabels=labels[0:11000]
	testlabels=labels[0:11000]
	pickle.dump(train,open(os.getcwd()+'/train_test_split/train_data','wb')) 
	pickle.dump(test,open(os.getcwd()+'/train_test_split/test_data','wb')) 
	pickle.dump(trainlabels,open(os.getcwd()+'/train_test_split/train_labels','wb')) 
	pickle.dump(testlabels,open(os.getcwd()+'/train_test_split/test_labels','wb')) 
	return train,trainlabels,test,testlabels

def runSKLearnLDA(doc_collection,docidentifiers):
	print('Start SKLearnLDA...')
	tf_vec=CountVectorizer(max_df=0.95,min_df=2,max_features=no_features,stop_words='english')
	termfreq=tf_vec.fit_transform(doc_collection)
	#Run LDA using scitkit learn
	print('Constructing LDA model...')
	startlda=time.time()
	ldamodel=LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(termfreq,docidentifiers)
	print('LDA Model Construction Took:'+str((time.time()-startlda)/60)+' minutes.')
	startldavecs=time.time()
	print('Constructing LDA vectors...')
	#ldavecs = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit_transform(termfreq,docidentifiers)	
	ldavecs=ldamodel.transform(termfreq)
	print('LDA Vector Construction Took:'+str((time.time()-startldavecs)/60)+' minutes.')
	print('Completed SKLearnLDA!')
	pickle.dump(ldavecs,open(os.getcwd()+'/models/LDA/ldavectors','wb'))
	pickle.dump(ldamodel,open(os.getcwd()+'/models/LDA/ldamodel','wb'))
	pickle.dump(termfreq,open(os.getcwd()+'/models/LDA/term_frequency','wb'))
	return termfreq,ldamodel,ldavecs

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))


def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return float(len(intersection))/float(len(union))


def display_reduced_dim(data,dim_reduc_type,docidentifiers):
	if train_dim_red:
		print('Starting '+dim_reduc_type+' embedding...')
		startembed=time.time()
		if dim_reduc_type=='TSNE':
			model_embedded=TSNE(n_components=num_dim).fit_transform(data)
		elif dim_reduc_type=='MDS':
			model_embedded=MDS(n_components=num_dim).fit_transform(data)
		elif dim_reduc_type=='PCA':
			model_embedded=PCA(n_components=num_dim).fit_transform(data)
		else:
			print('Error: dimensionality reduction method not supported (use TSNE, MDS or PCA)')
		print(dim_reduc_type+' embedding took: '+str((time.time()-startembed)/60)+' minutes.')
	else:
		print('Loading reduced dimension representation of data....')
		model_embedded=pickle.load(open(os.getcwd()+'/graphs/low_D_doc2vec/LDAvec_2d_'+dim_reduc_type,'rb'))
		print('Loaded!')
	if num_dim==2:
		print('Length of DocSentiment='+str(len(docidentifiers)))
		print('Length of model_embedded='+str(len(model_embedded)))
		pickle.dump(model_embedded,open(os.getcwd()+'/graphs/low_D_doc2vec/LDAvec_2d_'+dim_reduc_type,'wb'))
		for ii in range(0,model_embedded.shape[0]):
			print('plotted point '+str(ii))
			if docidentifiers[ii]==0:# in ['conservative','republican','libertarian']
				plt.plot(model_embedded[ii,0],model_embedded[ii,1],'or')
			elif docidentifiers[ii]==1:# in ['liberal','democrats','socialism']
				plt.plot(model_embedded[ii,0],model_embedded[ii,1],'ob')
			elif docidentifiers[ii]==2:# in ['moderatepolitics','politics']
				plt.plot(model_embedded[ii,0],model_embedded[ii,1],'ok')
		plt.show()
	elif num_dim==3:
		pickle.dump(model_embedded,open(os.getcwd()+'/graphs/low_D_doc2vec/LDAvec_3d'+dim_reduc_type,'wb'))
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(model_embedded[:,0],model_embedded[:,1],model_embedded[:,2],c='r',marker='o')
		plt.show()
	else:
		print('Error: Please use either 2 or 3 dimensions for dimensionality reduction!')

	return model_embedded


#subreds=['conservative','liberal']

'''
wordcloudtext=str()
for i in documents:
	wordcloudtext=wordcloudtext+' '+i
'''

'''
#NMF is able to use tf-idf
tfidf_vectorizer=TfidfVectorizer(max_df=0.95,min_df=2,max_features=no_features,stop_words='english')
tfidf=tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names=tfidf_vectorizer.get_feature_names()
# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
'''
#LDA can only use raw term counts for LDA because it is a probabilitstic graphical model
#tf_vectorizer=CountVectorizer(max_df=0.95,min_df=2,max_features=no_features,stop_words='english')
#tf=tf_vectorizer.fit_transform(documents)
loaddata=pickle.load(open(os.getcwd()+'/data/postdata2.pickle','rb'))
urls=loaddata.url
documents=list()
docsentiment=list()
docIDs=list()
for i,j,k in zip(loaddata.websitetext,loaddata.submissionID,loaddata.subreddit):
	if (type(i) is str):
		documents.append(i)
		docIDs.append(j)
		if k in ['conservative','republican','libertarian','the_congress']:
			docsentiment.append(0)#conservative
		elif k in ['liberal','democrats','politics','socialism']:
			docsentiment.append(1)#liberal
		elif k=='moderatepolitics':
			docsentiment.append(2)#neutral
		else:
			docsentiment.append(3)#unclassified



traindocs,trainlabels,testdocs,testlabels=train_test_split_data(documents,docsentiment)	



if train_model:
	tf_train, lda_train, ldavectors_train = runSKLearnLDA(traindocs+testdocs,trainlabels+testlabels)
else:
	#tf=pickle.load(open(os.getcwd()+'/models/LDA/term_frequency','rb'))
	lda_train=pickle.load(open(os.getcwd()+'/models/LDA/ldamodel','rb'))
	ldavectors_train=pickle.load(open(os.getcwd()+'/models/LDA/ldavectors','rb'))

display_reduced_dim(ldavectors_train,dim_red_method,trainlabels)

