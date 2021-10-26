import nltk
import numpy as np
import pandas as pd
 
#download the universal tagset from nltk
nltk.download('universal_tagset')
 
# reading the brown tagged sentences
train_set = list(nltk.corpus.brown.tagged_sents(tagset='universal')[:10000])
test_set = list(nltk.corpus.brown.tagged_sents(tagset='universal')[10150:10153])


# split data into training and validation set in the ratio 80:20
# create list of train and test tagged words
train_tagged_words = [ tup for tuple in train_set for tup in tuple ]
test_tagged_words = [ tup[0] for tuple in test_set for tup in tuple ]

#use set datatype to check how many unique tags are present in training data
tags = {tag for word,tag in train_tagged_words}


# compute Emission Probability
def word_given_tag(word, tag, train_bag = train_tagged_words):
    tag_list = [pair for pair in train_bag if pair[1]==tag]
    count_tag = len(tag_list)#total number of times the passed tag occurred in train_bag
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0]==word]
    #now calculate the total number of times the passed word occurred as the passed tag.
    count_w_given_tag = len(w_given_tag_list)
    return (count_w_given_tag, count_tag)

# compute  Transition Probability
def t2_given_t1(t2, t1, train_bag = train_tagged_words):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t==t1])
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        if tags[index]==t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)

# creating t x t transition matrix of tags, t= no of tags
# Matrix(i, j) represents P(jth tag after the ith tag)
trans_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
for i, t1 in enumerate(list(tags)):
    for j, t2 in enumerate(list(tags)): 
        trans_matrix[i, j] = t2_given_t1(t2, t1)[0]/t2_given_t1(t2, t1)[1]


# # convert the matrix to a df for better readability
trans_df = pd.DataFrame(trans_matrix, columns = list(tags), index=list(tags))
trans_df.head(13)

def Viterbi(words, trans_df, train_bag = train_tagged_words ):
    state = []
    T = list(set([pair[1] for pair in train_bag]))
     
    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = [] 
        for tag in T:
            if key == 0:
                transition_p = trans_df.loc['.', tag]
            else:
                transition_p = trans_df.loc[state[-1], tag]
                 
            # compute emission and state probabilities
            emission_p = word_given_tag(words[key], tag)[0]/word_given_tag(words[key], tag)[1]
            state_probability = emission_p * transition_p    
            p.append(state_probability)
             
        pmax = max(p)
        # getting state for which probability is maximum
        state_max = T[p.index(pmax)] 
        state.append(state_max)
    return list(zip(words, state))

tagged_seq = Viterbi(test_tagged_words, trans_df)
print(tagged_seq)

