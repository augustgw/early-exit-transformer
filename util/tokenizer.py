"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
import io
import editdistance

class Tokenizer:

    def __init__(self):
        '''
        self.spacy_de = spacy.load('de_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm')
        '''
    def tokenize_de(self, text):
        """
        Tokenizes German text from a string into a list of strings
        """
        return None#[tok.text for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return None#[tok.text for tok in self.spacy_en.tokenizer(text)]

def load_dict(file_path):
    dict=[]
    with io.open(file_path, encoding = 'utf-8') as f:
        for line in f:
            dict += [line.strip("\n")]
    return dict

def apply_lex(predicted, lexicon):
    lex_out=[]
    for w in predicted.split(" "):
        if w in lexicon:
            lex_out += [w]
        else:  
            min_lex=99999
            w_min=""
            for w_lex in lexicon:
                d_lex=editdistance.eval(w, w_lex)
                if d_lex < min_lex:
                    min_lex = d_lex
                    w_min = w_lex
            lex_out += [w_min]
                    
    return " ".join([str(item) for item in lex_out])
