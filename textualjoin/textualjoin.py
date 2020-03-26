import re
import nltk
import spacy
import numpy as np
import pandas as pd
from unidecode import unidecode
from nltk.corpus import stopwords
from collections import defaultdict
from nested_dict import nested_dict
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

class TextualJoin(object):
    """
    This class allows one to join two pandas dataframes based on text. Two
    pandas dataframe must be given: a left side (in) and a right side (out)
    dataframe. All combinations of both in and out text are performed and 
    the join is done with those who have the highest cosine similarity. Pre-existing
    aggregations can be provided before combining the left and right data.

    Arguments:
        in_df (required, pd.Dataframe): the left side dataframe.
        out_df (required, pd.DataFrame): the right side dataframe.
        aggregation_keys_arr (optional, list): the aggregation keys to make all combinations
        of the right and left sides.
        text_key (required, str): the text column name.
        language (required, str): portuguese or english.
    """
    
    def __init__(self,
                 in_df,
                 out_df,
                 text_key,
                 aggregation_keys_arr = None, 
                 language='portuguese'):
        
        if type(in_df) != pd.core.frame.DataFrame:
            raise Exception("Input data must be a pandas dataframe.")

        if type(out_df) != pd.core.frame.DataFrame:
            raise Exception("Output data must be a pandas dataframe.")
        
        if len(in_df.columns) != len(out_df.columns):
            raise Exception("Input and Output data must have the same column names.")
            
        if list(in_df.columns) != list(out_df.columns):
            raise Exception("Input and Output data must have the same column names.")
        
        self.in_df = in_df
        self.out_df = out_df
        
        if aggregation_keys_arr is None:
            self.in_df['aggregattion_index'] = 1
            self.out_df['aggregattion_index'] = 1
            self.aggregation_keys_arr = ['aggregattion_index']
        else:
            self.aggregation_keys_arr = aggregation_keys_arr
            
        self.text_key = text_key
        self.language = language

        if self.language=='portuguese':
            self.nlp = spacy.load('pt')
        
    def _custom_tokenizer(self,
                          text):
        """
        This function creates a customized tokenizer that
        performs tokenization, regex text transformations, word lemmatization,
        stopword removal and symbol removal. Moreover, this function
        uses a Porter Stemmer for removing morphological affixes.
        Arguments:
            text: the input text
        Output:
            clean_tokens: the tokenized text
        """
        
        if self.language == 'portuguese':
            # Create nlp object
            doc = self.nlp(text)

            # Lemmatizer
            clean_tokens = [w.lemma_.lower().strip() for w in doc]

            symbols_regex = r'[!@#$%^&*(),.?":{}|<>]'
            clean_tokens = [re.sub(symbols_regex,"",unidecode(t.lower().strip())) for t in clean_tokens]

            symbols_list = ['_', '-', '?', '!', '.', '@', '#', '$', '%', '^', '&', '*', '(', ')', '[', ']', '/', ',', '']
            stemmer = SnowballStemmer(language='portuguese')
            clean_tokens = [stemmer.stem(token) for token in clean_tokens if token != '' and 
                            token not in symbols_list]

            clean_tokens = [i.split('-') for i in clean_tokens]

            flatten = lambda l: [item for sublist in l for item in sublist]

            clean_tokens = flatten(clean_tokens)

            return clean_tokens
    
    def _create_in_out_dict(self,
                            in_df,
                            out_df,
                            aggregation_keys_arr,
                            text_key):
        """
        This function creates the keys and values for computing
        the unified in/out dictionary.
        
        Arguments:
            in_df: the in (left) dataframe
            out_df: the out (right) dataframe
            aggregation_keys_arr: the aggregation keys to make all combinations
            of the right and left sides.
            text_key: the text column name
        """

        in_out_df = pd.merge(in_df, out_df, how='inner', on=aggregation_keys_arr)

        in_text_key = text_key + "_x"
        out_text_key = text_key + "_y"

        keys = [tuple(key) for key in in_out_df[aggregation_keys_arr].values]

        in_values = list(in_out_df[in_text_key].values)
        out_values = list(in_out_df[out_text_key].values)

        return in_values, out_values, keys

    def _create_unified_dict(self,
                             in_values,
                             out_values,
                             keys):
        """
        This function creates a unified dictionary with all
        combinations of input (left) and output (right) values
        for each aggregation key.
        
        Arguments:
            in_values: the input (left) values
            out_value: the output (right) values
            keys: the aggregation keys
        
        Output:
            in_out_dict: the unified in/out dictionary
        """
        
        in_out_dict = nested_dict(2, list)

        idx=0
        for key in keys:
            if in_values[idx] not in in_out_dict[key]['in']:
                in_out_dict[key]['in'].append(in_values[idx])
            if out_values[idx] not in in_out_dict[key]['out']:
                in_out_dict[key]['out'].append(out_values[idx])
            idx+=1

        return in_out_dict
    
    def _join_in_out_data(self,
                          corpus,
                          in_out_dict):
        """
        This function joins the left and the right values
        that contains the highest cosine similarity.
        
        Arguments:
            corpus: the corpus object
            in_out_dict: the input (left)/ output (right)
            dictionary for each aggregation key.
        
        Output:
            in_out_dict_update: the joined in/out dictionary
        """
        in_out_dict_updated = nested_dict(2, list)

        for key in in_out_dict.keys():

            in_arr = corpus.transform(in_out_dict[key]['in'])
            out_arr = corpus.transform(in_out_dict[key]['out'])

            in_out_sim = np.argmax(cosine_similarity(in_arr,out_arr),axis=1)

            for idx, text_in in enumerate(in_out_dict[key]['in']):
                in_out_dict_updated[key][text_in] = in_out_dict[key]['out'][in_out_sim[idx]]

        return in_out_dict_updated
    
    def _dict_to_df(self,
                    in_out_dict_updated):
        """
        This function converts the in/out joined
        dictionary into a pandas dataframe with 
        the following columns: index, in, out.
        
        Arguments:
            in_out_dict_updated: the joined in/out dictionary.The
            dictionary keys corresponds to the "in" (left) values and the
            dictionary values corresponds to the "out" (right) values.
        
        Output:
            df: a pandas dataframe
        """
        index = []
        in_arr = []
        out_arr = []
        for key_1 in in_out_dict_updated.keys():
            for key_2, value in in_out_dict_updated[key_1].items():
                index.append(key_1)
                in_arr.append(key_2)
                out_arr.append(value)

        df = pd.DataFrame()
        df['index'] = pd.Series(index)
        df['in'] = pd.Series(in_arr)
        df['out'] = pd.Series(out_arr)

        return df
    
    def fit(self):
        """
        This function performs all steps required for join
        the left and right dataframes.
        
        Output:
            unified_df: the unified dataframe
        """
        
        #Create a corpus
        corpus_texts = np.asarray(pd.concat([self.in_df[self.text_key],
                                             self.out_df[self.text_key]]))
        vectorizer = CountVectorizer(tokenizer=self._custom_tokenizer, token_pattern=None)
        corpus = vectorizer.fit(corpus_texts)
        
        #Create a unified in/out dictionary
        in_values, out_values, keys = self._create_in_out_dict(in_df=self.in_df, 
                                                               out_df=self.out_df,
                                                               aggregation_keys_arr=self.aggregation_keys_arr,
                                                               text_key=self.text_key)
        
        in_out_dict = self._create_unified_dict(in_values=in_values,
                                                out_values=out_values,
                                                keys=keys)
        
        #Join in/out data using cosine similarity
        in_out_dict_updated = self._join_in_out_data(corpus=corpus,
                                                     in_out_dict=in_out_dict)
        
        #Make final dataframe
        self._unified_df = self._dict_to_df(in_out_dict_updated)
        
        return self._unified_df