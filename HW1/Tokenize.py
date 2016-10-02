# Information Retrieval Homework-1
#
# Author: Kai Zhu <kxz160030@utdallas.edu>

from __future__ import division
import re
import os
import glob
import timeit
import heapq
import sys
from string import punctuation
from stemming.porter2 import stem

#token table
token_table=dict()
#stemmed table
stem_table=dict()
#store the token table
def Token_Tabletization(T):
	article_no=T[0]
	for items in T[1:]:
		if items not in token_table:
			#keep for generate token table with doc id
			#token_table[items]=[article_no]
			token_table[items]=1
		else:
			token_table[items]+=1
		#elif article_no not in token_table[items]:
			#token_table[items].append(article_no)

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
	Token_Tabletization(file_table)

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

    return text.split()

#stemming
def stemming(t_table):
	for key, value in t_table.iteritems():
		stemkey=stem(key)
		if stemkey not in stem_table:
			stem_table[stemkey]=value
		else:
			stem_table[stemkey]+=value



#get top 30 frequency in token list
def top30(t_table):
	n=1
	for item in heapq.nlargest(30,t_table,key=t_table.get):
		print '		',n,item,t_table[item]
		n+=1
	return '\n'


def main():
	#record how long the program need to acquire the text collection
	starttime=timeit.default_timer()
	#get current path
	cwd=os.getcwd()
	files=[]
	#read argv from command line
	if len(sys.argv)>1 and sys.argv[1]=='Sanda':
		files= glob.glob("/people/cs/s/sanda/cs6322/Cranfield/*")
	else:
		files= glob.glob(cwd+"/Cranfield/*")
	filenum=0
	if(len(files)<1):
		print "No file has been found in Cranfield folder"
		sys.exit()

	for i in files:
		filenum+=1
		Read_file(i)

	#Read_file('cranfield0001')
	stoptime=timeit.default_timer()
	#print max(token_table, key=token_table.get)
	Number_of_tokens=sum(token_table.itervalues())
	Number_of_unique_tokens=len(token_table)
	Number_of_one_tokens=sum( x == 1 for x in token_table.itervalues() )
	#print top 30 frequency in token table
	
	print '1.The number of tokens in the Cranfield text collections:',Number_of_tokens
	print '2.The number of unique (e.g. distinct) tokens in the Cranfield text collection:',Number_of_unique_tokens
	print '3.The number of tokens that occur only once in the Cranfield text collection:',Number_of_one_tokens
	print '4.The 30 most frequent word tokens in the Cranfield:\n',top30(token_table)
	print '5.The average number of word tokens per document:',round(Number_of_tokens/filenum,2)
	print 'Running Time to acquire the Cranfield coolection:', round(stoptime-starttime,2), 'second'
	#star
	stemming(token_table)
	Number_of_distinct_stems=len(stem_table)
	Number_of_one_stems=sum( x == 1 for x in stem_table.itervalues() )
	print '\n'
	print '1.The number of distinct stems in the Cranfield text collection:',Number_of_distinct_stems
	print '2.The number of stems that occur only once in the Cranfield text collection:',Number_of_one_stems
	print '3.The 30 most frequent stems in the Cranfield:\n',top30(stem_table)
	print '4.The average number of word tokens per document:',round(Number_of_distinct_stems/filenum,2)


	

if __name__ == '__main__':
   main()