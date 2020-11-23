import logging

import numpy as np
from allennlp.commands.elmo import ElmoEmbedder

from transformers import RobertaConfig, RobertaModel
from transformers import RobertaTokenizer

logger = logging.getLogger(__name__)


class RobertaEmbedding(object):
    '''
    A wrapper class for Roberta from Transformers
    '''
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = RobertaModel.from_pretrained('roberta-base',
                                                  return_dict=True)

    def get_token_mapping(self, sentence):
        embeddings = self.tokenizer(' '.join(sentence.get_tokens_strings()),
                                    return_tensors="pt")
        counter = 0
        mapping = {k: [] for k, _ in enumerate(sentence.get_tokens_strings())}
        for i, token in enumerate(
                self.tokenizer.convert_ids_to_tokens(
                    embeddings["input_ids"].tolist()[0])):
            if token == "<s>" or token == "</s>":
                continue
            elif token[0] == "Ġ":
                counter += 1
                mapping[counter].append(i)
            else:
                mapping[counter].append(i)
                continue
        return mapping

    def get_roberta_seq(self, sentence):
        '''
        This function gets a sentence object and returns Roberta embeddings
        :param sentence: a sentence object
        :return: the averaged roberta embeddings
        '''
        sent_string = ' '.join(sentence.get_tokens_strings())
        embeddings = self.tokenizer(sent_string, return_tensors="pt")
        outputs = self.model(**embeddings)
        output = outputs.last_hidden_state
        return output


class ElmoEmbedding(object):
    '''
    A wrapper class for the ElmoEmbedder of Allen NLP
    '''
    def __init__(self, options_file, weight_file):
        logger.info('Loading Elmo Embedding module')
        self.embedder = ElmoEmbedder(options_file, weight_file)
        logger.info('Elmo Embedding module loaded successfully')

    def get_elmo_avg(self, sentence):
        '''
        This function gets a sentence object and returns and ELMo embeddings of
        each word in the sentences (specifically here, we average over the 3 ELMo layers).
        :param sentence: a sentence object
        :return: the averaged ELMo embeddings of each word in the sentences
        '''
        tokenized_sent = sentence.get_tokens_strings()
        embeddings = self.embedder.embed_sentence(tokenized_sent)
        output = np.average(embeddings, axis=0)

        return output
