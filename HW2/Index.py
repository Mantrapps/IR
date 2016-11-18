# Information Retrieval Homework-2
# 
# 
# 
# Term attributes in dictionary
# - Document Frequency (df)
# - Term Frequency (tf) in collection
# - Inverted List
#   - maxtf
#   - doclen
#   - term_frequecny


from __future__ import division
import re
import pickle
import os
import glob
import timeit
import heapq
import sys
import struct
from struct import calcsize
from string import punctuation
from stemming.porter2 import stem
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet



#token table
token_table=dict()
#stemmed token table
#data structure {'term1':['df','tf',[{docid,term_frequency,max_frequency,doc_len},{}]}
stem_table=dict()
#lemma token table
#data structure {'term1':['df','tf',[{docid,term_frequency,max_frequency,doc_len},{}]}
lemma_table=dict()
#stop words
stop_words=list()

#master table
map_table=list()

#largest max_tf_docid
max_tf_in_collection=[]
#Largest doc_len
largest_doc_len_in_collection=[]

#number of inverted list
num_inverted_list_v1_uncompresse=0
num_inverted_list_v1_compressed=0
num_inverted_list_v2_unompresse=0
num_inverted_list_v2_compressed=0

#convert bits to bytes
def bitstring_to_bytes(s):
	v = int(s, 2)
	b = bytearray()
	while v:
		b.append(v & 0xff)
		v >>= 8
	return bytes(b[::-1])

#build lemma_table, uncompress 
def Lemma_Dictionary(T):
	#need get doc length including stop word
	article_no=int(T[0])
	doclen=len(T[1:])
	largest_doc_len_in_collection.append((doclen,article_no))
	#need calculate doc freq, belongs to term as key
	#need calculate term freq, belongs to term as key
	
	#clean stop word
	T=stop_word_remove(T[1:])
	#need get max term frequence
	max_tf=most_common(T)
	max_tf_in_collection.append((max_tf,article_no))
	for items in T:
		lemma_item=lemmatization(items)
		#print lemma_item
		if lemma_item not in lemma_table:
			#if lemma_item not in lemma_table docfq,termfq,posting(sorted)]
			lemma_table[lemma_item]=[1,1,[{"article_no":article_no,"term_frequency":1,"max_frequency":max_tf,"doc_len":doclen}]]
		elif any(d["article_no"]== article_no for d in lemma_table[lemma_item][2]):
			#if lemma_item in lemma_table and article_no in posting [{},{}]
			lemma_table[lemma_item][1]+=1
			(item for item in lemma_table[lemma_item][2] if item["article_no"]== article_no).next()["term_frequency"]+=1
		else:
			#if lemma_item in lemma_table but article_no not in posting list
			# add df
			lemma_table[lemma_item][0]+=1
			lemma_table[lemma_item][1]+=1
			lemma_table[lemma_item][2].append({"article_no":article_no,"term_frequency":1,"max_frequency":max_tf,"doc_len":doclen})

#generate Lemma table uncompressed binary file
def lemma_uncompression_binary_file():
	#lemma_table {'term1':['df','tf',[{docid,term_frequency,max_frequency,doc_len},{}]}
	fil=open("Index_Version1.uncompress","wb")
	#buffersize for header 
	buffersize=0
	#buffer1 for map_table
	buffer1=''
	#buffer2 for Lemma term
	buffer2=''
	lemma_pointer=0
	#buffer3 for posting list
	buffer3=''
	posting_pointer=0
	for lemma_item_key,lemma_item_values in lemma_table.iteritems():
		#lemma term
		term=lemma_item_key
		#encoding
		#print term
		termb=term.encode('UTF-8')
		#lemma term in binary
		buffer2+=termb
		#term posting list in binary
		postingb=''
		#build max_frequency_, doc_len, term_frequency, article_no
		for item in lemma_item_values[2]:
			postingb+=struct.pack(">4i",item['doc_len'],item['term_frequency'],item['max_frequency'],item['article_no'])
		buffer3+=postingb
		#document frequency
		df=lemma_item_values[0]
		#term frequency in collection
		cf=lemma_item_values[1]
		#print posting=lemma_item_values[2]
		# map_table (df-4byte,cf-4byte,posting pointer-4byte, term pointer-4byte, posting max_frequency length)
		buffer1+=struct.pack(">4i",df,cf,posting_pointer,lemma_pointer)
		#lemma term starting pointer
		lemma_pointer+=len(termb)
		posting_pointer+=len(postingb)
		#print struct.unpack("ii",buffer)
	map_tablesize=len(buffer1)
	lemmasize=len(buffer2)
	#header file, lemma term starting pointer, Posting List starting pointer
	headsize=calcsize(">2i")
	global num_inverted_list_v1_uncompresse 
	num_inverted_list_v1_uncompresse=map_tablesize/calcsize(">4i")
	#print headsize+map_tablesize,headsize+map_tablesize+lemmasize
	fil.write(struct.pack(">2i",headsize+map_tablesize,headsize+map_tablesize+lemmasize))
	#map table
	fil.write(buffer1)
	#term
	fil.write(buffer2)
	#posting list
	fil.write(buffer3)

def gamma_code(i):
	bina=bin(i)[2:]
	offset=bina[1:]
	lenoffset=len(offset)
	val='1'*lenoffset+'0'
	val=val+offset
	return bitstring_to_bytes(val)

def gamma_code_d(i):
	bina=bin(i)[2:]
	offset=bina[1:]
	lenoffset=len(offset)
	val='1'*lenoffset+'0'
	val=val+offset
	return val

def delta_code(i):
	bina=bin(i)[2:]
	offset=bina[1:]
	lenbin=len(bina)
	val=gamma_code_d(lenbin)
	val=val+offset
	return bitstring_to_bytes(val)

#generate lemma table binary file compressed with block compression and gamma code, k=8
def lemma_compression_binary_file():
	#lemma_table {'term1':['df','tf',[{docid,term_frequency,max_frequency,doc_len},{}]}
	fil=open("Index_Version1.compressed","wb")
	#buffersize for header 
	buffersize=0
	#buffer1 for map_table
	buffer1=''
	#buffer2 for Lemma term
	buffer2=''
	lemma_pointer=0
	#buffer3 for posting list
	buffer3=''
	posting_pointer=0
	blockcount=0
	for lemma_item_key,lemma_item_values in sorted(lemma_table.iteritems()):
		blockcount+=1
		#lemma term
		term=lemma_item_key
		global num_inverted_list_v1_compressed 
		num_inverted_list_v1_compressed+=1
		#encoding
		#print term
		termb=term.encode('UTF-8')
		#1 byte to store the term length
		new_termb=struct.pack(">b",len(termb))+termb		
		#lemma term in binary
		buffer2+=new_termb
		#term posting list in binary
		postingb=''
		#need sort
		#build max_frequency, doc_len, term_frequency, article_no
		for item in lemma_item_values[2]:
			#first article number in posting file
			first_article_no=lemma_item_values[2][0]['article_no']
			# doc len , term frequency, article_no GAP. Use Gamma Code
			if item['article_no']== first_article_no:
				#first item in posting list, gamma_code for doc_len
				postingb+=gamma_code(item['doc_len'])+gamma_code(item['term_frequency'])+gamma_code(item['max_frequency'])+gamma_code(item['article_no'])
				#postingb+=struct.pack(">3i",item['doc_len'],item['term_frequency'],item['article_no'])
			else:
				gap=item['article_no']-first_article_no
				postingb+=gamma_code(item['doc_len'])+gamma_code(item['term_frequency'])+gamma_code(item['max_frequency'])+gamma_code(gap)
				#postingb+=struct.pack(">3i",item['doc_len'],item['term_frequency'],item['article_no']-first_article_no)
		buffer3+=postingb
		#document frequency Use Gamma Code
		df=lemma_item_values[0]
		#term frequency in collection Use Gamma Code
		cf=lemma_item_values[1]
		#print posting=lemma_item_values[2]
		# map_table (df-4byte,cf-4byte,posting pointer-4byte, term pointer-4byte)
		#block =8
		if blockcount==8:
			#use gamma code in df,cf 
			buffer1+=gamma_code(df)+gamma_code(cf)
			buffer1+=struct.pack(">2i",posting_pointer,lemma_pointer)
			blockcount=0;
		else:
			buffer1+=gamma_code(df)+gamma_code(cf)
			buffer1+=struct.pack(">i",posting_pointer)
		#lemma term starting pointer
		lemma_pointer+=len(new_termb)
		posting_pointer+=len(postingb)
		#print struct.unpack("ii",buffer)
	map_tablesize=len(buffer1)
	lemmasize=len(buffer2)
	#header file, lemma term starting pointer, Posting List starting pointer
	headsize=calcsize(">2i")
	
	#print headsize+map_tablesize,headsize+map_tablesize+lemmasize
	fil.write(struct.pack(">2i",headsize+map_tablesize,headsize+map_tablesize+lemmasize))
	#map table
	fil.write(buffer1)
	#term
	fil.write(buffer2)
	#posting list
	fil.write(buffer3)

#build stem table, uncompress
def Stem_Dictionary(T):

	#need get doc length including stop word
	doclen=len(T[1:])
	#need calculate doc freq, belongs to term as key
	#need calculate term freq, belongs to term as key
	article_no=int(T[0])
	#clean stop word
	T=stop_word_remove(T[1:])
	#need get max term frequence
	max_tf=most_common(T)
	for items in T:
		stem_item=stem(items)
		if stem_item not in stem_table:
			#if stem_item not in stem table docfq,termfq,posting(sorted)]
			stem_table[stem_item]=[1,1,[{"article_no":article_no,"term_frequency":1,"max_frequency":max_tf,"doc_len":doclen}]]
		elif any(d["article_no"]== article_no for d in stem_table[stem_item][2]):
			#if stem_item in stem table and article_no in posting [{},{}]
			stem_table[stem_item][1]+=1
			(item for item in stem_table[stem_item][2] if item["article_no"]== article_no).next()["term_frequency"]+=1
		else:
			#if stem item in stem stabl but article_no not in posting list
			# add df
			stem_table[stem_item][0]+=1
			stem_table[stem_item][1]+=1
			stem_table[stem_item][2].append({"article_no":article_no,"term_frequency":1,"max_frequency":max_tf,"doc_len":doclen})

#generate stem table uncompressed binary file
def stem_uncompression_binary_file():
	#stem_table {'term1':['df','tf',[{docid,term_frequency,max_frequency,doc_len},{}]}
	fil=open("Index_Version2.uncompress","wb")
	#buffersize for header 
	buffersize=0
	#buffer1 for map_table
	buffer1=''
	#buffer2 for stem term
	buffer2=''
	stem_pointer=0
	#buffer3 for posting list
	buffer3=''
	posting_pointer=0

	for stem_item_key,stem_item_values in stem_table.iteritems():
		#stem term
		term=stem_item_key
		#encoding
		#print term
		termb=term.encode('UTF-8')
		#sterm term in binary
		buffer2+=termb
		#term posting list in binary
		postingb=''
		#build max_frequency, doc_len, term_frequency, article_no
		for item in stem_item_values[2]:
			postingb+=struct.pack(">4i",item['doc_len'],item['term_frequency'],item['max_frequency'],item['article_no'])
		buffer3+=postingb
		#document frequency
		df=stem_item_values[0]
		#term frequency in collection
		cf=stem_item_values[1]
		#print posting=stem_item_values[2]
		# map_table (df-4byte,cf-4byte,posting pointer-4byte, term pointer-4byte)
		buffer1+=struct.pack(">4i",df,cf,posting_pointer,stem_pointer)
		#stem term starting pointer
		stem_pointer+=len(termb)
		posting_pointer+=len(postingb)
		#print struct.unpack("ii",buffer)
	map_tablesize=len(buffer1)
	stemsize=len(buffer2)
	#header file, Stem term starting pointer, Posting List starting pointer
	headsize=calcsize(">2i")
	global num_inverted_list_v2_uncompresse 
	num_inverted_list_v2_uncompresse=map_tablesize/calcsize(">4i")
	#print headsize+map_tablesize,headsize+map_tablesize+stemsize
	fil.write(struct.pack(">2i",headsize+map_tablesize,headsize+map_tablesize+stemsize))
	#map table
	fil.write(buffer1)
	#term
	fil.write(buffer2)
	#posting list
	fil.write(buffer3)

#front coding  size of the first word_samewords*extra_words+size$
def front_coding(termlist):
	samewords=os.path.commonprefix(termlist)
	new_termb=struct.pack(">b",len(termlist[0]))+samewords.encode('UTF-8')+'*'.encode('UTF-8')+termlist[0][(len(samewords)-len(termlist[0])):]
	for item in termlist[1:]:
		extralen=len(item)-len(samewords)
		new_termb+=struct.pack(">b",extralen)+'$'.encode('UTF-8')+item[-extralen:].encode('UTF-8')
	return new_termb

#generate stem table binary file compressed with delta code
def stem_compression_binary_file():
	#stem_table {'term1':['df','tf',[{docid,term_frequency,max_frequency,doc_len},{}]}
	fil=open("Index_Version2.compressed","wb")
	#buffersize for header 
	buffersize=0
	#buffer1 for map_table
	buffer1=''
	#buffer2 for stem term
	buffer2=''
	stem_pointer=0
	#buffer3 for posting list
	buffer3=''
	posting_pointer=0
	blockcount=8
	termlist=[]
	#need sort first
	for stem_item_key,stem_item_values in sorted(stem_table.iteritems()):
		#stem term
		term=stem_item_key
		global num_inverted_list_v2_compressed 
		num_inverted_list_v2_compressed+=1		
		#term posting list in binary
		postingb=''
		#build max_frequency, doc_len, term_frequency, article_no
		for item in stem_item_values[2]:
			#first article number in posting file
			first_article_no=stem_item_values[2][0]['article_no']
			# doc len , term frequency, article_no GAP. Use Delta Code
			if item['article_no']== first_article_no:
				#first item in posting list, Delta code for doc_len
				postingb+=delta_code(item['doc_len'])+delta_code(item['term_frequency'])+delta_code(item['max_frequency'])+delta_code(item['article_no'])
				#postingb+=struct.pack(">3i",item['doc_len'],item['term_frequency'],item['article_no'])
			else:
				gap=item['article_no']-first_article_no
				postingb+=delta_code(item['doc_len'])+delta_code(item['term_frequency'])+delta_code(item['max_frequency'])+delta_code(gap)
				#postingb+=struct.pack(">3i",item['doc_len'],item['term_frequency'],item['article_no']-first_article_no)
		buffer3+=postingb
		#document frequency Use delta Code
		df=stem_item_values[0]
		#term frequency in collection Use delta Code
		cf=stem_item_values[1]
		#print posting=stem_item_values[2]
		# map_table (df-4byte,cf-4byte,posting pointer-4byte, term pointer-4byte)
		#block =8
		if blockcount==8:
			#use delta code in df,cf 
			buffer1+=delta_code(df)+delta_code(cf)
			buffer1+=struct.pack(">2i",posting_pointer,stem_pointer)

			blockcount=1;
		else:
			buffer1+=delta_code(df)+delta_code(cf)
			buffer1+=struct.pack(">i",posting_pointer)
			blockcount+=1
		#font coding 
		
		if blockcount==8:
			front_coding_termb=front_coding(termlist)
			buffer2+=front_coding_termb
			stem_pointer+=len(front_coding_termb)
			termlist=[]
			termlist.append(term)
		else:
			termlist.append(term)
		#stem term starting pointer
		
		posting_pointer+=len(postingb)
		#print struct.unpack("ii",buffer)
	map_tablesize=len(buffer1)
	stemsize=len(buffer2)
	#header file, stem term starting pointer, Posting List starting pointer
	headsize=calcsize(">2i")
	
	#print headsize+map_tablesize,headsize+map_tablesize+stemsize
	fil.write(struct.pack(">2i",headsize+map_tablesize,headsize+map_tablesize+stemsize))
	#map table
	fil.write(buffer1)
	#term
	fil.write(buffer2)
	#posting list
	fil.write(buffer3)

#get most frequent term
def most_common(L):
	term =max(set(L), key=L.count)
	return L.count(term)

#read file and store tokens in token table
def Read_file(f):
	file_table=[]
	obj=open(f)
	lines= obj.readlines()
	lines_after=list()
	for x in lines:
		#clean SGML tags
		result=re.sub("<[^>]*>", "",x)
		if result.rstrip() is not '':
			file_table+= tokenize(result.rstrip().lower())
	
	#load file to lemma table
	Lemma_Dictionary(file_table)
	#load file to stem table
	Stem_Dictionary(file_table)

#stop words
def stop_word():
	filep= 'stopwords'
	obj=open(filep,'r')
	lines=obj.readlines()
	for x in lines:
		stop_words.append(x.rstrip('\n'))
#stop words remove
def stop_word_remove(l):
	newlist=[]
	for item in l:
		if item in stop_words:
			continue
		else:
			newlist.append(item)
	return newlist

#tokenization
def tokenize(text):
    #replace , . to ''
    #text= re.sub((r'\B\"\w*( \w*)*\"\B'),r"",text)
    #text= re.sub((r'\B\'\w*( \w*)*\'\B'),r"",text)

    #replace stick comman with ', '

    text= re.sub(r'([:,])([^\d])', r' \1 \2',text)
    #clean all punctuation besides space
    text=' '.join(filter(None, (word.strip(punctuation) for word in text.split())))
    #replace [] () <> to ''
    text= re.sub((r'[\]\[\(\)\{\}\<\>]'),r" ",text)
    #parse 'm 's 'd to ''
    text= re.sub((r"([^' ])('[sS]|'[mM]|'[dD]|') "),r"\1 \2 ",text)
    #parse 'll  're 've n't and keep it
    text= re.sub((r"([^' ])('ll|'re|'ve|n't) "),r"\1 \2 ",text)
    #parse words with hyphen and underline 
    text= re.sub((r"[-_]")," ",text)
    #ingore numbers
    text= re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)
    
    return text.split()
#Stemming
def stemming(t_table):
	for key, value in t_table.iteritems():
		stemkey=stem(key)
		if stemkey not in stem_table:
			stem_table[stemkey]=value
		else:
			stem_table[stemkey]+=value

#get pos tag first
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

#Lemmatization
def lemmatization(text):
	wordnet_lemmatizer=WordNetLemmatizer()
	postag=get_wordnet_pos(text)
	if postag is not '':
		return wordnet_lemmatizer.lemmatize(text,pos=postag)
	else:
		return wordnet_lemmatizer.lemmatize(text)

#get file size
def getSize(filename):
    st = os.stat(filename)
    return st.st_size

def Question(text):
	#print '\"Reynolds\":', 'df:','tf:','Inverted list length:'
	dfreq=stem_table[stem(text)][0]
	tfreq=stem_table[stem(text)][1]
	postingb=''
	for item in stem_table[stem(text)][2]:
		postingb+=struct.pack(">4i",item['doc_len'],item['term_frequency'],item['max_frequency'],item['article_no'])

	print text, 'df:',dfreq,'tf:',tfreq,'Inverted list length(bytes):',len(postingb)

#main program
def main():
	#record how long the program need to acquire the text collection
	starttime=timeit.default_timer()
	#generate stop word list
	stop_word()
	#Start Load Cranfield
	cwd=os.getcwd()
	files=[]
	filenum=0
	if len(sys.argv)>1:
		files= glob.glob("/people/cs/s/sanda/cs6322/Cranfield/*")
	else:
		files= glob.glob(cwd+"/Cranfield/*")
	#read argv from command line
	if(len(files)<1):
		print "No file has been found"
		sys.exit()
	#Read_file(cwd+"/Cranfield/cranfield1209")
	#Read_file(cwd+"/Cranfield/cranfield1210")
	
	for i in files:
		filenum+=1
		Read_file(i)
	

	stoptime1=timeit.default_timer()
	print 'Running Time to load the Cranfield collection to memory:', round(stoptime1-starttime,2), 'second'
	#Load Cranfield complete

	#v1 uncompressed
	lemma_uncompression_binary_file()
	stoptime2=timeit.default_timer()
	print 'Running Time to build v1 uncompressed index:', round(stoptime2-stoptime1,2), 'second'
	print 'Index_Version1.uncompressed File Size:', getSize(cwd+"/Index_Version1.uncompress"), ' bytes'
	#v1 compressed 
	lemma_compression_binary_file()
	stoptime3=timeit.default_timer()
	print 'Running Time to build v1 compressed index:', round(stoptime3-stoptime2,2), 'second'
	print 'Index_Version1.compressed File Size:', getSize(cwd+"/Index_Version1.compressed"), ' bytes'
	#v2 uncompressed
	stem_uncompression_binary_file()
	stoptime4=timeit.default_timer()
	print 'Running Time to build v2 uncompressed index:', round(stoptime4-stoptime3,2), 'second'
	print 'Index_Version2.uncompressed File Size:', getSize(cwd+"/Index_Version2.uncompress"), ' bytes'
	#v2 compresssed
	stem_compression_binary_file()
	stoptime5=timeit.default_timer()
	print 'Running Time to build v2 compressed index:', round(stoptime5-stoptime4,2), 'second'
	print 'Index_Version2.compressed File Size:',getSize(cwd+"/Index_Version2.compressed"), ' bytes'



	print 'The number of Inverted Lists in Index_Version1:', 'uncompress: ',num_inverted_list_v1_uncompresse, 'compressed: ',num_inverted_list_v1_compressed
	print 'The number of Inverted Lists in Index_Version2:', 'uncompress: ',num_inverted_list_v2_uncompresse, 'compressed: ',num_inverted_list_v2_compressed

	Question('reynolds')
	Question('nasa')
	Question('prandtl')
	Question('flow')
	Question('pressure')
	Question('boundary')
	Question('shock')

	print '\"NASA\"', 'df: ',stem_table[stem('nasa')][0]
	print 'First item in Posting list:', 'term_frequency: ', stem_table[stem('nasa')][2][0]['term_frequency'], 'Doc_len: ',stem_table[stem('nasa')][2][0]['doc_len'], 'Max_tf: ',stem_table[stem('nasa')][2][0]['max_frequency']
	print 'Second item in Posting list:', 'term_frequency: ', stem_table[stem('nasa')][2][1]['term_frequency'], 'Doc_len: ',stem_table[stem('nasa')][2][1]['doc_len'], 'Max_tf: ',stem_table[stem('nasa')][2][1]['max_frequency']
	print 'Third item in Posting list:', 'term_frequency: ', stem_table[stem('nasa')][2][2]['term_frequency'], 'Doc_len: ',stem_table[stem('nasa')][2][2]['doc_len'], 'Max_tf: ',stem_table[stem('nasa')][2][2]['max_frequency']

#term from index1, largest df and lowest df, if more than one, list all -4 point
	df_db_index1=[]
	for itemkey,itemvalue in lemma_table.iteritems():
		df_db_index1.append((itemvalue[0],itemkey))
	maxdf_index1=max(df_db_index1,key=lambda x:x[0])[0]
	print 'largest df from index1:'
	for item in df_db_index1:
		if item[0]==maxdf_index1:
			print item[1]
	mindf_index1=min(df_db_index1,key=lambda x:x[0])[0]
	fil=open('lowest_df_index1','w')
	print 'lowest df from index1 stored to file'
	for item in df_db_index1:
		if item[0]==mindf_index1:
			fil.write(item[1])
	fil.close()
	print ''
#term from index2, largest df and lowest df, if more than one, list all -4 point
	df_db_index2=[]
	for itemkey,itemvalue in stem_table.iteritems():
		df_db_index2.append((itemvalue[0],itemkey))
	maxdf_index2=max(df_db_index2,key=lambda x:x[0])[0]
	print 'largest df from index2:'
	for item in df_db_index2:
		if item[0]==maxdf_index2:
			print item[1]
	mindf_index2=min(df_db_index2,key=lambda x:x[0])[0]
	print 'lowest df from index2 stored to file'
	fil=open('lowest_df_index2','w')
	for item in df_db_index2:
		if item[0]==mindf_index2:
			fil.write(item[1])
	fil.close()
	print ''


#From collection, list largest max_tf in collection, the document with largest doclen in the collection -4+5 point
	print 'largest doclen document in the collection:'
	print 'Cranfield' , max(largest_doc_len_in_collection,key=lambda x:x[0])[1]
	print 'largest term frequency document in the collection:'
	print 'Cranfield', max(max_tf_in_collection,key=lambda x:x[0])[1]



if __name__ == '__main__':
   main()