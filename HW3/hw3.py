# Information Retrieval Homework-3
# 
# abbreviation
# tf: 				term frequency in one doc
# maxtf: 			the frequency of most frequency indexed item tf in one doc
# collectionsize: 	the number of doc in this collection
# df: 				the number of document contain this term
# doclen: 			the length of the document, not count stopword
# avgdoclen: 		the avg doc len in collection 
# idf: 				log(collectionsize/df)

from __future__ import division
import re
import pickle
import math
import os
import glob
import timeit
import heapq
import sys
import struct
import xml.etree.ElementTree as ET
from struct import calcsize
from string import punctuation
from stemming.porter2 import stem
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet


class relevance_model:
	
	#initiation relevance_model
	def __init__(self):
		pass
	#step 5-1 comput weight
	def comput_weight_1(self,tf,maxtf,df,collectionsize):
		result=(0.4+0.6*(math.log(tf+0.5,10)/math.log(maxtf+1.0,10)))*(math.log(collectionsize/df,10)/math.log(collectionsize,10))
		return round(result,4)

	#step 5-2 comput weight
	def comput_weight_2(self,tf,doclen,avgdoclen,df,collectionsize):
		result=(0.4+0.6*(tf/(tf+0.5+1.5*(doclen/avgdoclen)))*math.log(collectionsize/df,10)/math.log(collectionsize,10))
		return round(result,4)

	# cosine similarity
	def c_s():
		pass

	def set_collection_size(self,size):
		self.collectionsize=size

class IR_Tools:
	
	def __init__(self):
		pass

	def get_pos_tag(self,text):
		if text[0]=='J':
			return wordnet.ADJ
		elif text[0]=='V':
			return wordnet.VERB
		elif text[0]=='N':
			return wordnet.NOUN
		elif text[0]=='R':
			return wordnet.ADV
		else:
			return ''

	def lemmatization(self,text):
		wordnet_lemmatizer=WordNetLemmatizer()
		treebank_tag=nltk.pos_tag(text)
		new_list=[]
		for i in treebank_tag:
			tag = self.get_pos_tag(i[1])
			if tag is not '':
				# delete u
				new_list.append(wordnet_lemmatizer.lemmatize(i[0],pos=tag).encode("utf-8"))
			else:
				new_list.append(wordnet_lemmatizer.lemmatize(i[0]))
		return new_list

	def tokenize(self,text):
		text= re.sub(r'([:,])([^\d])', r' \1 \2',text)
		text=' '.join(filter(None, (word.strip(punctuation) for word in text.split())))
		text= re.sub((r'[\]\[\(\)\{\}\<\>]'),r" ",text)
		text= re.sub((r"([^' ])('[sS]|'[mM]|'[dD]|') "),r"\1 \2 ",text)
		text= re.sub((r"([^' ])('ll|'re|'ve|n't) "),r"\1 \2 ",text)
		text= re.sub((r"[-_]")," ",text)
		text= re.sub("$\d+\W+|\b\d+\b|\W+\d+$", " ", text)
		return text.split()


class Dictionary_Base:
	#Term Dictionary
	stop_words=list()
	doclen_list=list()
	collectionsize=0
	title_list=list()
	#lemma token table
	#data structure {'term1':['df','tf',[{docid,term_frequency,max_frequency,doc_len},{}]}
	lemma_table=dict()
	#data structure of query table {'term1':['df','tf','maxtf'],''}
	query_tabl=dict()
	query_maxtf=0
	#data structure of doc_term table {'term1':['df','tf','maxtf'],''}
	doc_term_table=dict()
	doc_max_tf=0
	doc_len=0

	def __init__ (self):
		#generate stop word list
		self.get_stop_word()
	
	def set_collection_size(self,size):
		self.collectionsize=size

	def get_avg_doc_len(self):
		total=0
		for i in self.doclen_list:
			total+=i[0]
		return int(round(total/self.collectionsize))

	def read_doc(self,f):
		ir=IR_Tools()
		file_table=[]
		file_content=self.get_content_between_tag(f)
		file_table= ir.tokenize(file_content.rstrip().lower())
				#load file to lemma table
		#print self.remove_stop_word(file_table)
		self.Lemma_Dictionary(self.remove_stop_word(file_table))


	def get_content_between_tag(self,f):
		tree=ET.parse(f)
		root=tree.getroot()
		content=''
		for child in root:
			if child.tag=='DOCNO':
				content+=child.text
			if child.tag=='TITLE':
				content+=child.text
				self.title_list.append(child.text)
			if child.tag=='TEXT':
				content+=child.text
		return content
	
	def get_content_between_title(self,f):
		tree=ET.parse(f)
		root=tree.getroot()
		content=''
		for child in root:
			if child.tag=='TITLE':
				content+=child.text
		return content


	def read_query(self,q):
		ir=IR_Tools()
		q=q.rstrip().lower()
		q=ir.tokenize(q)
		q=self.remove_stop_word(q)
		q=ir.lemmatization(q)
		#print q
		self.query_Dictionary(q)

	def Lemma_Dictionary(self,T):
		ir=IR_Tools()
		# get doc id
		article_no=int(T[0])
		# get doc len
		doc_len=len(T[1:])
		# append to doclen_list
		self.doclen_list.append((doc_len,article_no))
		T=T[1:]
		# lemmatizaiton list
		T=ir.lemmatization(T)
		#need get max term frequence
		max_tf=self.most_common(T)
		for lemma_item in T:
			if lemma_item not in self.lemma_table:
				self.lemma_table[lemma_item]=[1,1,[{"DOCNO":article_no,"tf":1,"max_tf":max_tf,"doclen":doc_len}]]
			elif any(d["DOCNO"]== article_no for d in self.lemma_table[lemma_item][2]):
				self.lemma_table[lemma_item][1]+=1
				(item for item in self.lemma_table[lemma_item][2] if item["DOCNO"]== article_no).next()["tf"]+=1
			else:
				self.lemma_table[lemma_item][0]+=1
				self.lemma_table[lemma_item][1]+=1
				self.lemma_table[lemma_item][2].append({"DOCNO":article_no,"tf":1,"max_tf":max_tf,"doclen":doc_len})

	def query_Dictionary(self, T):
		#maxtf
		self.query_table=dict()
		max_tf_query=self.most_common(T)
		self.query_maxtf=max_tf_query
		#tf
		for query_item in T:
			if query_item not in self.query_table:
				self.query_table[query_item]=[1,1,max_tf_query]
			else:
				self.query_table[query_item][1]+=1
		#print self.query_table

	def get_doc_term(self,docid):
		self.doc_term_table=dict()
		for key, val in self.lemma_table.iteritems():
			for i in val[2]:
				if i['DOCNO']==docid:
					#key=term : value= [df, tf, max_tf ]
					self.doc_max_tf=i['max_tf']
					self.doc_len=i['doclen']
					self.doc_term_table[key]=[val[0],i['tf'],i['max_tf']]

	def query_vector(self,query):
		#self.get_doc_term(docid)
		#self.read_query(query)
		#maxtf=self.query_table
		r_m=relevance_model()
		query_vector_list=[]
		for collection_key,collection_val in self.lemma_table.iteritems():
			if collection_key not in self.query_table.keys():
				#in collection not in query
				#tf,maxtf,df,collectionsize
				query_vector_list.append(r_m.comput_weight_1(0,self.query_maxtf,collection_val[0],self.collectionsize))
			else:
				#in collection and in query
				#query data structure {'term1':['df','tf','maxtf'],''}
				#query_vector_list.append([doc_key,r_m.comput_weight_1(self.query_table[doc_key][1],self.query_maxtf,doc_val[0],self.collectionsize)])
				query_vector_list.append(r_m.comput_weight_1(self.query_table[collection_key][1],self.query_maxtf,collection_val[0],self.collectionsize))
				#query term in table
		return query_vector_list


	def doc_vector(self,docid):

		self.get_doc_term(docid)
		r_m=relevance_model()
		doc_vector_list=[]
		#in collection
		for collection_key,collection_val in self.lemma_table.iteritems():
			if collection_key not in self.doc_term_table.keys():
				#in collection not in doc
				#tf,maxtf,df,collectionsize
				doc_vector_list.append(r_m.comput_weight_1(0,self.doc_max_tf,collection_val[0],self.collectionsize))
			else:
				#in collection and in doc
				#query data structure {'term1':['df','tf','maxtf'],''}
				#query_vector_list.append([doc_key,r_m.comput_weight_1(self.query_table[doc_key][1],self.query_maxtf,doc_val[0],self.collectionsize)])
				doc_vector_list.append(r_m.comput_weight_1(self.doc_term_table[collection_key][1],self.doc_max_tf,collection_val[0],self.collectionsize))
				#query term in table
		return doc_vector_list

	def query_vector_2(self,query):
		#self.get_doc_term(docid)
		#self.read_query(query)
		#maxtf=self.query_table
		r_m=relevance_model()
		query_vector_list=[]
		for collection_key,collection_val in self.lemma_table.iteritems():
			if collection_key not in self.query_table.keys():
				#in collection not in query
				#comput_weight_2(self,tf,doclen,avgdoclen,df,collectionsize):
				query_vector_list.append(r_m.comput_weight_2(0,1,1,collection_val[0],self.collectionsize))
			else:
				#in collection and in query
				#query data structure {'term1':['df','tf','maxtf'],''}
				query_vector_list.append(r_m.comput_weight_2(self.query_table[collection_key][1],1,1,collection_val[0],self.collectionsize))
				#query term in table
		return query_vector_list

	def doc_vector_2(self,docid):
		self.get_doc_term(docid)
		r_m=relevance_model()
		doc_vector_list=[]
		avg_len=self.get_avg_doc_len()
		#in collection
		for collection_key,collection_val in self.lemma_table.iteritems():
			if collection_key not in self.doc_term_table.keys():
				#in collection not in doc
				#comput_weight_2(self,tf,doclen,avgdoclen,df,collectionsize):
				doc_vector_list.append(r_m.comput_weight_2(0,self.doc_len,avg_len,collection_val[0],self.collectionsize))
			else:
				#in collection and in doc
				#query data structure {'term1':['df','tf','maxtf'],''}
				#query_vector_list.append([doc_key,r_m.comput_weight_1(self.query_table[doc_key][1],self.query_maxtf,doc_val[0],self.collectionsize)])
				doc_vector_list.append(r_m.comput_weight_2(self.doc_term_table[collection_key][1],self.doc_len,avg_len,collection_val[0],self.collectionsize))
				#query term in table
		return doc_vector_list

	def comput(self,query,iden):
		self.read_query(query)
		q_v= self.query_vector(query)
		buf=''
		fil=open("Result_Weighting1_Query_"+iden,"w")
		mom=math.sqrt(sum(j**2 for j in q_v))
		norm_q_v = [i/mom for i in q_v]
		buf+='Query:'+str(query)+'\n'
		buf+='Query Vector:\n'
		buf+=str(norm_q_v)
		#print 'Query Vector:'
		#print norm_q_v
		buf+='\n'
		result=[]
		for docid in range(1,self.collectionsize+1):
			d_v= self.doc_vector(docid)
			mom2=math.sqrt(sum(j**2 for j in d_v))
			norm_d_v = [i/mom2 for i in d_v]
			#print norm_d_v
			result.append((docid,sum([i*j for i,j in zip(norm_q_v,norm_d_v)]),norm_d_v))	
		result= sorted(result,key=lambda x: x[1], reverse=True)[:5]
		count=1
		for i in result:
			print 'rank-',count
			print 'Doc_id:',i[0],' Score:',i[1] 
			print 'Title:', self.title_list[i[0]-1].strip()
			buf+= 'Doc_id:'+str(i[0])+' Score:'+str(i[1])+'\n Doc_Title'+str(self.title_list[i[0]-1].strip())
			buf+= '\n'
			buf+= 'Document Vector:'+'\n'
			buf+= str(i[2])
			buf+= '\n'
		fil.write(buf)



	def comput2(self,query,iden):
		self.read_query(query)
		q_v= self.query_vector_2(query)
		buf=''
		fil=open("Result_Weighting2_Query_"+iden,"w")
		mom=math.sqrt(sum(j**2 for j in q_v))
		norm_q_v = [i/mom for i in q_v]
		buf+='Query:'+str(query)+'\n'
		buf+='Query Vector:\n'
		buf+=str(norm_q_v)
		#print 'Query Vector:'
		#print norm_q_v
		#print norm_q_v
		result=[]
		for docid in range(1,self.collectionsize+1):
			d_v= self.doc_vector_2(docid)
			mom2=math.sqrt(sum(j**2 for j in d_v))
			norm_d_v = [i/mom2 for i in d_v]
			#print norm_d_v
			result.append((docid,sum([i*j for i,j in zip(norm_q_v,norm_d_v)]),norm_d_v))		
		#return top 5 
		result= sorted(result,key=lambda x: x[1], reverse=True)[:5]
		count=1
		for i in result:
			print 'rank-',count
			print 'Doc_id:',i[0],' Score:',i[1]
			print 'title:', self.title_list[i[0]-1].strip()
			buf+= 'Doc_id:'+str(i[0])+' Score:'+str(i[1])+'\n Doc_Title:'+str(self.title_list[i[0]-1].strip())
			buf+= '\n'
			buf+= 'Document Vector:'+'\n'
			buf+= str(i[2])
			buf+= '\n'
		fil.write(buf)

	def get_stop_word(self):
		filep= 'stopwords'
		obj=open(filep,'r')
		lines=obj.readlines()
		for x in lines:
			self.stop_words.append(x.rstrip('\n').strip())

	def remove_stop_word(self,l):
		newlist=[]
		for item in l:
			if item in self.stop_words:
				continue
			else:
				newlist.append(item)
		return newlist
	#get most frequent term
	def most_common(self,L):
		term =max(set(L), key=L.count)
		return L.count(term)


def main():
	#I/O read query
	#record how long the program need to acquire the text collection
	#starttime=timeit.default_timer()
	#Start Load Cranfield
	query_list=['what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft ',
	'what are the structural and aeroelastic problems associated with flight of high speed aircraft',
	'what problems of heat conduction in composite slabs have been solved so far',
	'can a criterion be developed to show empirically the validity of flow solutions for chemically reacting gas mixtures based on the simplifying assumption of instantaneous local chemical equilibrium ',
	'what chemical kinetic system is applicable to hypersonic aerodynamic problems ',
	'what theoretical and experimental guides do we have as to turbulent couette flow behaviour',
	'is it possible to relate the available pressure distributions for an ogive forebody at zero angle of attack to the lower surface pressures of an equivalent ogive forebody at angle of attack',
	'what methods -dash exact or approximate -dash are presently available for predicting body pressures at angle of attack',
	'papers on internal /slip flow/ heat transfer studies',
	'are real-gas transport properties for air available over a wide range of enthalpies and densities ',
	'is it possible to find an analytical,  similar solution of the strong blast wave problem in the newtonian approximation ',
	'how can the aerodynamic performance of channel flow ground effect machines be calculated ',
	'what is the basic mechanism of the transonic aileron buzz',
	'papers on shock-sound wave interaction',
	'material properties of photoelastic materials',
	'can the transverse potential flow about a body of revolution be calculated efficiently by an electronic computer',
	'can the three-dimensional problem of a transverse potential flow about a body of revolution be reduced to a two-dimensional problem',
	'are experimental pressure distributions on bodies of revolution at angle of attack available',
	'does there exist a good basic treatment of the dynamics of re-entry combining consideration of realistic effects with relative simplicity of results',
	'has anyone formally determined the influence of joule heating,  produced by the induced current,  in magnetohydrodynamic free convection flows under general conditions'
	]
	cwd=os.getcwd()
	files=[]
	#collection size
	collectionsize=0
	if len(sys.argv)>1:
		files= glob.glob("/people/cs/s/sanda/cs6322/Cranfield/*")
	else:
		files= glob.glob(cwd+"/Cranfield/*")
	#read argv from command line
	if(len(files)<1):
		print "No file has been found"
		sys.exit()

	Dict_B=Dictionary_Base()
	#Dict_B.read_doc(cwd+"/Cranfield/cranfield0001")
	#collectionsize+=2
	#Dict_B.read_doc(cwd+"/Cranfield/cranfield0002")
	#Read_file(cwd+"/Cranfield/cranfield1209")
	#Read_file(cwd+"/Cranfield/cranfield1210")

	for i in files:
		collectionsize+=1
		Dict_B.read_doc(i)
	
	#set collection_size
	Dict_B.set_collection_size(collectionsize)
	
	#use weighting-1 to apply queries
	print 'start query with weighting-1 schema'
	doc_n=1
	for i in query_list:
		#query name
		print i
		Dict_B.comput(i,str(doc_n))
		doc_n+=1
	
	print 'start query with weighting-2 schema'
	#use weighting-2 to apply queries
	doc_n=1
	for i in query_list:
		print i
		Dict_B.comput2(i,str(doc_n))
		doc_n+=1
	

if __name__ == '__main__':
   main()
