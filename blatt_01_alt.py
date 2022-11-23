# Assignment02 - IRTM, Momo Takamatsu, Sandro Weick, Tana Deeg

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


def create_bi_gram_dict(dictionary):
    """
    @param dictionary requires pre existing term_dictionary
    @return returns bi_gram_dictionary with bi_gram as key and all terms containing that 
            bi_gram as value
    """
    bigram_dict = {}
    for key in dictionary:
        for i in range(len(key)+1):
            if i == len(key):
                if key[i-1]+"$" not in bigram_dict:
                    bigram_dict[key[i-1]+"$"] = [key]
                else:
                    bigram_dict[key[i-1]+"$"].append(key)
            elif i-1 >=0:
                if key[i-1]+(key[i]) not in bigram_dict:
                    bigram_dict[key[i-1]+(key[i])] = [key]
                else:
                    bigram_dict[key[i-1]+(key[i])].append(key)
            else:
                if "$"+(key[i]) not in bigram_dict:
                    bigram_dict["$"+(key[i])] = [key]
                else:
                    bigram_dict["$"+(key[i])].append(key)
    return bigram_dict
    

def and_compare(term1: str, term2: str, term_dict, postings):
    """
    @param term1       query term 1
    @param term2       query term 2
    @param term_dict
    @param postings
    @return res_list   list with all postings_list entries that contain both terms
    """
    
    res_list = []

    term1_list = get_postings(term1, term_dict, postings)
    term2_list = get_postings(term2, term_dict, postings)

    term_1_iter = iter(term1_list)
    term_2_iter = iter(term2_list)

    try:   
        x = next(term_1_iter)
        y = next(term_2_iter)
    except StopIteration:
        pass

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
        except UnboundLocalError:
            flag = False
            

    return res_list

def get_detailed_results(res_list, df):
    """
    @param res_list
    @param df
    """
    detailed_res_list = []
    for res in res_list:
        detailed_res_list.append((df.iloc[res[0]]))
    return detailed_res_list

def get_wildcard_grams(term: str):
    """
    @param term         a term containing a wildcard
    @return wild_query  list with all bi grams, based on wildcard query term
    """
    
    wild_query = []
    saw_wild = False
    
    # iterate over each letter of term
    for i in range(len(term)+1):
        # when we reach te last letter of a term, and it is not a star
        # append letter + $
        if i == len(term) and term[i-1] != "*":
            wild_query.append(term[i-1]+"$")
        # do not append letter + $ if the last letter is *
        elif i == len(term):
            pass
        elif i-1 >=0:
            # $a*
            if term[i] == "*" and i-1 == 0:
                wild_query.append("$"+term[i-1])
                saw_wild = True
            # ab*
            if term[i]=="*":      
                saw_wild = True     
                # ...*a
            elif saw_wild:          
                saw_wild = False  
            # ...ab
            else:
                wild_query.append(term[i-1]+term[i])
        # $*
        elif term[i] == "*":
            saw_wild = True
        # $a
        else:
            wild_query.append("$"+term[i])
    return wild_query

def get_postings(term, term_dict, postings):
    """
    @param term
    @param term_dict
    @param pos_entries  list containing postings_list entry of term
    """
    pos_entries = postings[term_dict[term][0]]
    return pos_entries


def one_wild(wild, norm_term, bi_gram_dict, term_dict, posting):
    """
    @param wild
    @param norm_term
    @param bi_gram_dict 
    @param term_dict    
    @param posting  
    @return res_list    list with all postings_entries containing both 
                        wild and norm_term
    """
    res_list = []
    wild_query = get_wildcard_grams(wild)
    
    wild_terms = set()
    temp_set = set()
    for term in bi_gram_dict[wild_query[0]]:
        wild_terms.add(term)
    
    for gram in wild_query[1:]:
        for term in bi_gram_dict[gram]:
            temp_set.add(term)
        wild_terms = wild_terms.intersection(temp_set)
        temp_set.clear()

    for element in wild_terms:
        for res in and_compare(element, norm_term, term_dict, posting):
            res_list.append(res)

    return res_list


def query(term1: str, term2=None, term_dict=None, postings=None, df=None, bi_gram_dict=None):
    """
    Search for a postings list for one term, or search for 'AND' occurrences of two terms

    @param terms        list of terms that should be compared
    @param dict         optional parameter dict, if dict, postings and df are not given, create them using index()
    @param postings     see dict
    @param df           see dict
    @return res_list            returns all index, docID pairs, that match both terms
    @return detailed_res_list   returns the dataFrame entries for all results
    """
    res_list = []

    wildcard_1 = False
    wildcard_2 = False

### check wether term 1 and/or term 2 are wildcard queries
    # check for wildcards in term1; if no wildcard, normalize term
    if "*" in term1:
        wildcard_1 = True
    else:
        term1 = normalize(term1, multiple=False)[0]
    
    if term2 == None and not wildcard_1:
        return get_postings(term1, term_dict, postings)
    
    elif term2 == None and wildcard_1:
        res_list = []
        wild_query = get_wildcard_grams(term1)
    
        # initialize a wild_terms set with initial term-contenders 
        # based on the first bi_gram
        wild_terms = set()
        temp_set = set()
        for term in bi_gram_dict[wild_query[0]]:
            wild_terms.add(term)
    
        # iterate over all bi_grams and intersect the wild_terms set
        # with all term-contenders,
        # in the end, wild_terms only contains terms, that contain all elements
        # from the wild_query bi_gram list
        for gram in wild_query[1:]:
            for term in bi_gram_dict[gram]:
                temp_set.add(term)
            wild_terms = wild_terms.intersection(temp_set)
        term_list = list(wild_terms)

        for term in term_list:
            res_list.extend(get_postings(term))
        
        return res_list

    # check for wildcards in term2; if no wildcard, normalize term
    elif "*" in term2:
        wildcards_2 = True
    else:
        term2 = normalize(term2, multiple=False)[0]

    
###  both query terms contain wildcards
    # get query results for terms both containing wildcards
    if wildcard_1 and wildcard_2:
     
        # result for term1
        wild_query_1 = get_wildcard_grams(term1)
    
        wild_terms_1 = set()
        temp_set_1 = set()
        for term in bi_gram_dict[wild_query_1[0]]:
            wild_terms_1.add(term)
    
        for gram in wild_query_1[1:]:
            for term in bi_gram_dict[gram]:
                temp_set_1.add(term)
            wild_terms_1 = wild_terms_1.intersection(temp_set)
            temp_set_1.clear()

        # result for term2 
        wild_query_2 = get_wildcard_grams(term2)
    
        wild_terms_2 = set()
        temp_set_2 = set()
        for term in bi_gram_dict[wild_query_2[0]]:
            wild_terms_2.add(term)
    
        for gram in wild_query_2[1:]:
            for term in bi_gram_dict[gram]:
                temp_set_2.add(term)
            wild_terms_2 = wild_terms_2.intersection(temp_set_2)
            temp_set_2.clear()
        
        # combine both results of term1 and term2 with and_compare function
        for t1 in wild_terms_1:
            for t2 in wild_terms_2:
                for res in and_compare(t1, t2, term_dict, postings):
                    res_list.append(res)

### only one of the terms contains wildcards
    # term1 containing wildcard and term2 not
    elif wildcard_1:
        res_list = one_wild(term1,term2, bi_gram_dict, term_dict, postings)

    # term2 containing wildcard and term1 not
    elif wildcards_2:
        res_list = one_wild(term2, term1, bi_gram_dict, term_dict, postings)
    
    else:
        res_list = and_compare(term1, term2)

    # get detailed results
    detailed_res  = get_detailed_results(res_list, df)

    return res_list, detailed_res
        



if __name__ == "__main__":
    
    # access the formerly created pickle (.pkl) files, this only works 
    # if the commented code above was executed previously
    with open('saved_dictionary_alt.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    with open('saved_postings_alt.pkl', 'rb') as f:
        loaded_postings = pickle.load(f)
    with open("saved_df_alt.pkl", "rb") as f:
        loaded_df = pd.read_pickle(f)

    with open("saved_bi_gram_dict.pkl", "rb") as f:
        loaded_bigram_dict = pickle.load(f)

    res_1, res_1_detailed = query("germ*","tumor",  term_dict=loaded_dict, postings=loaded_postings, df=loaded_df, bi_gram_dict=loaded_bigram_dict)
    res_2, res_2_detailed = query("german","tum*",  term_dict=loaded_dict, postings=loaded_postings, df=loaded_df, bi_gram_dict= loaded_bigram_dict)
    res_3, res_3_detailed = query("ge*man","tumor",  term_dict=loaded_dict, postings=loaded_postings, df=loaded_df, bi_gram_dict=loaded_bigram_dict)
    res_4, res_4_detailed = query("german","*umo*",  term_dict=loaded_dict, postings=loaded_postings, df=loaded_df, bi_gram_dict=loaded_bigram_dict)
    
    for el in res_1_detailed[:10]:
        print(el["body"])
    print()
    for el in res_2_detailed[:10]:
        print(el["body"])
    print()
    for el in res_3_detailed[:10]:
        print(el["body"])  
    print()
    for el in res_4_detailed[:10]:
        print(el["body"])

    
"""
OUTPUT:

@GermanLetsPlay Alles gute zum Geburtstag Manu, schon ganze 26 Jahre und das trotz Tumor im Hals xD
@GermanLetsPlay Herzlichen Glückwunsch zum 26. Geburtstag Herr Tumor von German Lets Play
@BadGirly_13 @Paulidraws @GermanLetsPlay @Paluten Ein Tumor haha XD
@GermanLetsPlay ach manu das kennt man ja ünd alles gute zum geburtstag bleib so wie du bbist [NEWLINE]lg dein tumor#missmaggy
@GermanLetsPlay alles gute zum Geburtstag du alter Tumor!
@ShinyClove @OdinakaJesus @LetsTeddybaer GermanLetsPlay ist tumor und Mensch. Paluten Kürbis und Mensch. OdinakaJesus ist Gott und Mensch[NEWLINE]Wtf mein Weltbild hat sich verändert woooow
@GermanLetsPlay Alles gute zum Geburtstag Manu, schon ganze 26 Jahre und das trotz Tumor im Hals xD
@GermanLetsPlay Herzlichen Glückwunsch zum 26. Geburtstag Herr Tumor von German Lets Play
@BadGirly_13 @Paulidraws @GermanLetsPlay @Paluten Ein Tumor haha XD
@GermanLetsPlay ach manu das kennt man ja ünd alles gute zum geburtstag bleib so wie du bbist [NEWLINE]lg dein tumor#missmaggy

The German word Pakationzeln means a malignant tumour of plasmacytes.
Germans call a tumour composed of nerve cells a Regfühnteinenmangsauchvenfortglerggesätzenaliseudern.
@GermanLetsPlay Herzlichen Glückwunsch zum 26. Geburtstag Herr Tumor von German Lets Play
@GermanLetsPlay Herzlichen Glückwunsch zum 26. Geburtstag Herr Tumor von German Lets Play
pictures of anal tumors man has sex with german shepherd porn zane's sex chronicles dvd janet  https://t.co/tAzURYwgra
@Ben_Aaronovitch @iSalome_chan As I understand it, Malignität is only used in medical terms, like Malignität eines Tumors. Bösartigkeit is the German word for it, but can be used for humans as well, e.g. "Die Bösartigkeit des Gesichtslosen".
|| Okay and now, I'm kinda disgusted because I saw something on Tumblr I didn't want to see.[NEWLINE][NEWLINE]Some of my fellow Germans are weird. Weirder than I am.

@GerardZalcman @dplanchard @jco interesting to see if ongoing irAE correlate with ongoing tumor control, as in this patient with remission and arthralgia, now more than 12 months after cessation of nivo https://t.co/CaA3PE9hO6
@GerardZalcman @dplanchard @jco interesting to see if ongoing irAE correlate with ongoing tumor control, as in this patient with remission and arthralgia, now more than 12 months after cessation of nivo https://t.co/CaA3PE9hO6
@GermanLetsPlay Herzlichen Glückwunsch zum 26. Geburtstag Herr Tumor von German Lets Play
@GermanLetsPlay Herzlichen Glückwunsch zum 26. Geburtstag Herr Tumor von German Lets Play
pictures of anal tumors man has sex with german shepherd porn zane's sex chronicles dvd janet  https://t.co/tAzURYwgra
@Ben_Aaronovitch @iSalome_chan As I understand it, Malignität is only used in medical terms, like Malignität eines Tumors. Bösartigkeit is the German word for it, but can be used for humans as well, e.g. "Die Bösartigkeit des Gesichtslosen".

Montag. Damit ist alles gesagt, oder?[NEWLINE][NEWLINE]#deutschland #comedy #lustig #witzig #montag #deutsch #geschenk #wahrheit german #bilder #fun #memes #schwarzer #humor #schwarzerhumor #lol  #happy #lachen #blogger_de #germanblogger #germanblog https://t.co/g06ICQsPyj
der Spaß #fun[NEWLINE]die Freude #joy[NEWLINE]der Humor #humor[NEWLINE][NEWLINE]#German #vocabulary https://t.co/BliNRRRKm6
Ich glaube, ich will nie wieder Luftballons geschenkt bekommen. Und du?[NEWLINE][NEWLINE]#deutschland #comedy #lustig #witzig #geburtstag #deutsch #geschenk #wahrheit german #bilder #fun #memes #schwarzer #humor #schwarzerhumor #lol  #happy #lachen #blogger_de #germanblogger #germanblog https://t.co/9FLgMoRFfv
@GermanLetsPlay Herzlichen Glückwunsch zum 26. Geburtstag Herr Tumor von German Lets Play
@GermanLetsPlay Herzlichen Glückwunsch zum 26. Geburtstag Herr Tumor von German Lets Play
pictures of anal tumors man has sex with german shepherd porn zane's sex chronicles dvd janet  https://t.co/tAzURYwgra
@Ben_Aaronovitch @iSalome_chan As I understand it, Malignität is only used in medical terms, like Malignität eines Tumors. Bösartigkeit is the German word for it, but can be used for humans as well, e.g. "Die Bösartigkeit des Gesichtslosen".
Montag. Damit ist alles gesagt, oder?[NEWLINE][NEWLINE]#deutschland #comedy #lustig #witzig #montag #deutsch #geschenk #wahrheit german #bilder #fun #memes #schwarzer #humor #schwarzerhumor #lol  #happy #lachen #blogger_de #germanblogger #germanblog https://t.co/g06ICQsPyj
Ich glaube, ich will nie wieder Luftballons geschenkt bekommen. Und du?[NEWLINE][NEWLINE]#deutschland #comedy #lustig #witzig #geburtstag #deutsch #geschenk #wahrheit german #bilder #fun #memes #schwarzer #humor #schwarzerhumor #lol  #happy #lachen #blogger_de #germanblogger #germanblog https://t.co/9FLgMoRFfv
⠀⠀⠀⠀❝You know, I'm actually pretty content with this.❞[NEWLINE][NEWLINE]╔[NEWLINE]║⠀⠀ • Finally a promo! [NEWLINE]║⠀⠀ • The Cooking Summoner[NEWLINE]║⠀⠀ • Not new to verse[NEWLINE]║⠀⠀ • German, sweet kid [NEWLINE]║⠀⠀ • SFW, ships with chem. [NEWLINE]║⠀⠀ • FERP only[NEWLINE]╚[NEWLINE][NEWLINE]⠀⠀⠀⠀DM for recruiting! https://t.co/K6qmDy4lDI
"""