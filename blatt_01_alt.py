# Assignment01 - IRTM, Momo Takamatsu, Sandro Weick, Tana Deeg

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle


def normalize(text: str, multiple=True):
    """
    Takes a string and normalizes it.
    @param text      a String
    @param multiple  optional parameter to simplify normalize() if the text input is just one word
    @return  final   a list containing all important tokens from the original text str in normalized form
                    !important, always returns a list, even if multiple=False
    """ 
    final = []
    # create a list containing stopwords from english and german (both langauages used in data)
    stops = stopwords.words("english") + stopwords.words("german")
    stemmer = PorterStemmer()

    # if the input text contains multiple words, tokenize them, remove all non alpha elements
    # remove stop words and stem them
    if multiple:
        tokens = nltk.word_tokenize(text)
        no_not_alpha = [word for word in tokens if word.isalpha()]
        non_stop = [word for word in no_not_alpha if word not in stops]
        for not_stemmed in non_stop:	
            final.append(stemmer.stem(not_stemmed))

    # if input text is just a single word only stem it
    # because search terms should not be removed, even if they won`t lead to results 
    else:
        stem = stemmer.stem(str(text))
        final.append(stem)
    
    # returns a list containing only normalized tokens
    return final


def index(filename: str):
    """
    Takes a filename as data source and creates a dictionary, postingslist and pandas DataFrame from it.
    
    @param filename     string with path (or name) to data
    @return dictionary  returns a dictionary with types as keys and pointer to postingslist, 
                        as well as the length of the corresponding postingslist
    @return postings    is a dictionary that contains entries for all types and a sorted postingslist
    @return df          a pandas dataframe that stores the data, can later be used to reaccess the documents
    """
    tuple_list = []
    dictionary = {}
    postings = {}

    with open(filename, newline='') as csvfile:
        
        # read csv data into a pandas dataFrame, skip lines that are wrongly formatted
        df = pd.read_csv(csvfile, delimiter="\t", header=None, names =["date", "docID", "author", "handle", "body"], on_bad_lines="skip")
        # create a column with index
        df.reset_index(inplace=True)
        # iterate over rows in DataFrame, normalize the body and create a tuple list, that contains index,docID and token 
        # for each token in the body
        for index, docID, text in zip(df["index"], df["docID"], df["body"]):
            tokens = normalize(text)
            for tok in tokens:
                tuple_list.append((int(index), int(docID), tok))
        # sort the tuple list alphabetically
        cleaned_tuple_list = sorted(tuple_list, key=lambda x: x[2])

        # create dictionary and postings entry from the sorted tuple list, in this implementation we use the token
        # as key in both the dictionary and the postings datastructure
        # postings entries contain both the index from the dataFrame, as well as the docID given in the data
        for index, docID, token in cleaned_tuple_list:
            if token not in dictionary:
                dictionary[token] = token
                postings[dictionary[token]] = [(index, docID)]
            else:
                postings[dictionary[token]].append((index, docID))
        
        # go through the dictionary and add the length of the postings entry for each type
        for key in dictionary:
            dictionary[key] = [dictionary[key], len(postings[dictionary[key]])]
    
    # return all data structures we created
    return dictionary, postings, df


"""
def create_bi_gram_dict(dictionary):
    bigram_dict = {}
    for key in dictionary:
        for i in len(key)
            try:
                if key[i-1].concat(key[i]) not in bigram_dict:
                    bigram_dict[key[i-1].concat(key[i])] = [key]
                else:
                    bigram_dict[key[i-1].concat(key[i])].append(key)
            except IndexError:
                 if "$".concat(key[i]) not in bigram_dict:
                    bigram_dict["$".concat(key[i])] = [key]
                else:
                    bigram_dict["$".concat(key[i])].append(key)

"""

"""
def and_compare(term1: str, term2: str, dict, postings, df):

    term1_list = query(term1,dict=dict, postings=postings, df=df)
    term2_list = query(term2,dict=dict, postings=postings, df=df)

    term_1_iter = iter(term1_list)
    term_2_iter = iter(term2_list)
        
    x = next(term_1_iter)
    y = next(term_2_iter)

    # find "AND" matches by comparing entries in both postingslists pairwise
    # if they match increment both by one, else increase the one with smaller index
    # stop iterating, once one would be increased, but has no more values
    flag = True
    while flag:
        try:
            if x[0] == y[0]:
                res_list.append(x)
                x = next(term_1_iter)
                y = next(term_2_iter)
                    
            elif x[0] < y[0]:
                x = next(term_1_iter)
            else:
                y = next(term_2_iter)
        except StopIteration:
            flag = False
            
    # get the DataFrame entry for every result
    # TODO do this in another function
    #for res in res_list:
    #    detailed_res_list.append((df.iloc[res[0]]))

    return res_list

"""
"""
def get_detailed_results(res_list: [], df):
    detailed_res_list = []
    for res in res_list:
        detailed_res_list.append((df.iloc[res[0]]))
    return detailed_res_list
"""
    


# TODO change query to accept terms as a list instead of term1, term2...
def query(term1: str, term2=None, dict=None, postings=None, df=None):
    """
    Search for a postings list for one term, or search for 'AND' occurrences of two terms

    @param term1        if only term1 is given, return postings list for term1, else return res and detailed_res lists
    @param term2        optional, if given, compute results for term1 AND term2
    @param dict         optional parameter dict, if dict, postings and df are not given, create them using index()
    @param postings     see dict
    @param df           see dict
    @return res_list            returns all index, docID pairs, that match both terms
    @return detailed_res_list   returns the dataFrame entries for all results
    """
    res_list = []
    detailed_res_list = []

    # if dict, postings or df are None, create them using index()
    try:
        if dict == None or postings == None or df == None:
            dict, postings, df = index("tweets.csv")
    except ValueError:
        pass
    
    # normalize term1, so that it matches with the types in dictionary
    # TODO implement wanted behaviour for terms as list, if len = 1 --> one term, we can normalize all
    # TODO  terms inside one iteration over list
    term1 = normalize(term1, multiple=False)[0]
    
    # TODO for assignment2: also split each term into its bigram with $ at the start
    # if no term2, return postings for term1
    if term2 == None:
        return postings[dict[term1][0]]

    

    # TODO do all this in another function and_compare()
    # if term2 not None, normalize term2, get postings list for both 
    else:
        term2 = normalize(term2, multiple=False)[0]
        term1_list = query(term1,dict=dict, postings=postings, df=df)
        term2_list = query(term2,dict=dict, postings=postings, df=df)
        
        # create iterator for postings lists
        term_1_iter = iter(term1_list)
        term_2_iter = iter(term2_list)
        
        x = next(term_1_iter)
        y = next(term_2_iter)

        # find "AND" matches by comparing entries in both postingslists pairwise
        # if they match increment both by one, else increase the one with smaller index
        # stop iterating, once one would be increased, but has no more values
        flag = True
        while flag:
            try:
                if x[0] == y[0]:
                    res_list.append(x)
                    x = next(term_1_iter)
                    y = next(term_2_iter)
                    
                elif x[0] < y[0]:
                    x = next(term_1_iter)
                else:
                    y = next(term_2_iter)
            except StopIteration:
                flag = False
            
        # get the DataFrame entry for every result
        for res in res_list:
            detailed_res_list.append((df.iloc[res[0]]))

        return res_list, detailed_res_list




if __name__ == "__main__":

    # only execute this once, this creates pickle (.pkl) files for the dictionary, postings and df
    """
    dictionary , postings, df = index("tweets.csv")
    with open('saved_dictionary_alt.pkl', 'wb+') as f:
        pickle.dump(dictionary, f)
    with open('saved_postings_alt.pkl', 'wb+') as f:
        pickle.dump(postings, f)
    with open("saved_df_alt.pkl", "wb+") as f:
        df.to_pickle(f)
    """
      
    # access the formerly created pickle (.pkl) files, this only works 
    # if the commented code above was executed previously
    with open('saved_dictionary_alt.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    with open('saved_postings_alt.pkl', 'rb') as f:
        loaded_postings = pickle.load(f)
    with open("saved_df_alt.pkl", "rb") as f:
        loaded_df = pd.read_pickle(f)


    res_both, res_both_detailed = query("tumors","cancer",  dict=loaded_dict, postings=loaded_postings, df=loaded_df)
    # print subset of query results
    for i in range(10):
        print(res_both_detailed[i]["body"])

"""
OUTPUT:

New Nanobots Kill Cancerous Tumors by Cutting off Their Blood Supply: https://t.co/g05sqIYGcK - #DigitalEconomy - February 19, 2018 at 08:01PM
A role for iNOS in inducing tumors in the gut #colorectal #cancer https://t.co/6Y4cHiR3cI
Sekali cancerous, risikonya kambuh lagi makin tinggi. Jaga-jaga saja. Kemarin si el** tuh, dia yg tumor gondok, sekarang jadi thyroid cancer lalu dinuklir --[NEWLINE][NEWLINE]Aih setop setop tante cakap macam mana la kebenaran pahit tu
New DNA nanorobots successfully target and kill off cancerous tumors https://t.co/Lq9oagDUay via @TechCrunch
An int. research team has succeeded in stopping the growth of malignant melanoma by reactivating a protective mechanism that prevents tumor cells from dividing https://t.co/Y6aXxCqoYr @MDC_Berlin #Cancer #CancerResearch
@StickProfessor did you know we had a breakthrough in Germany about leukemia?[NEWLINE]It already helped in 400 of 400 cases. [NEWLINE]The person still has Cancer, sadly, but the Tumor in the Brainis gone ^^
our new paper is out: We analyzed all Pubmed papers and all clinical trials related to cancer #immunotherapy and identified promising trends (including chemo+checkpoint, GI tumors, stroma and apoptosis) -> https://t.co/Cm97awMV3M @C_ReyesAldasoro @halama_immuno @nekvalous https://t.co/Tfq6QfUuQZ
Cancer ‘vaccine’ eradicates tumors in mice, holds promise in humans - https://t.co/hLSVBUrN8w via @Shareaholic
A cancer 'vaccine' is completely eliminating tumors in mice - New York Daily News https://t.co/jsDdA79eTf
A cancer 'vaccine' is completely eliminating tumors in mice - New York Daily News https://t.co/NZ2xXqeWYm
"""