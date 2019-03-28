from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

import logging
import typing
import os
import io
import re
from future.utils import PY3
from typing import Any, Dict, List, Optional, Text
from training_data import Message
logger = logging.getLogger(__name__)

import spacy
nlp = spacy.load('zh', parser=False)

class CountVectorsFeaturizer():
    """Bag of words featurizer

    Creates bag-of-words representation of intent features
    using sklearn's `CountVectorizer`.
    All tokens which consist only of digits (e.g. 123 and 99
    but not ab12d) will be represented by a single feature."""
    name = "intent_featurizer_count_vectors"
    provides = ["text_features"]
    requires = []

    def __init__(self, component_config=None):
        """Construct a new count vectorizer using the sklearn framework."""
        # super(CountVectorsFeaturizer, self).__init__(component_config)
        # regular expression for tokens
        self.token_pattern = r'(?u)\b\w\w+\b'
        # remove accents during the preprocessing step
        self.strip_accents = None
        # list of stop words
        self.stop_words = None
        # min number of word occurancies in the document to add to vocabulary
        self.min_df = 1
        # max number (fraction if float) of word occurancies
        # in the document to add to vocabulary
        self.max_df = 1.0
        # set ngram range
        self.min_ngram = 1
        self.max_ngram = 1
        # limit vocabulary size
        self.max_features = None
        # declare class instance for CountVect
        self.vect = None
        # preprocessor
        self.preprocessor = lambda s: re.sub(r'\b[0-9]+\b', 'NUMBER', s.lower())

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["sklearn"]

    def train(self, training_data, cfg=None, **kwargs):
        """Take parameters from config and
            construct a new count vectorizer using the sklearn framework."""
        from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

        # use even single character word as a token
        self.vect = CountVectorizer(token_pattern=self.token_pattern, strip_accents=self.strip_accents,analyzer='char',
                                    stop_words=self.stop_words, ngram_range=(self.min_ngram,self.max_ngram),
                                    max_df=self.max_df,min_df=self.min_df, max_features=self.max_features, preprocessor=self.preprocessor)
        # self.vect = TfidfVectorizer(token_pattern=self.token_pattern,stop_words=self.stop_words,
        #                             ngram_range=(self.min_ngram,self.max_ngram),preprocessor=self.preprocessor,
        #                             max_features=self.max_features,max_df=self.max_df, strip_accents=self.strip_accents,min_df=self.min_df)
        lem_exs = [' '.join([t.lemma_ for t in nlp(example)]) for example in training_data]
        print(lem_exs)
        try:
            X = self.vect.fit_transform(lem_exs).toarray()
        except ValueError:
            self.vect = None
            return
        print(X.shape)
        # for i, example in enumerate(training_data):
        #     # create bag for each example
        #     example.set("text_features", X[i])
        return X


    def process(self, message, **kwargs):
        # type: # (Message, **Any) -> Message
        if self.vect is None:
            logger.error("There is no trained CountVectorizer: "
                         "component is either not trained or "
                         "didn't receive enough training data")
        else:
            print([self._lemmatize(message)])
            bag = self.vect.transform([self._lemmatize(message)]).toarray()
            message.set("text_features", bag)

        return bag
    @staticmethod
    def _lemmatize(message):
        # print(nlp(message))
        message.set("spacy_doc", nlp(message.text))
        return ' '.join([t.lemma_ for t in message.get("spacy_doc")])
        # return ' '.join([t.lemma_ for t in nlp(message)])


    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):
        # type: (...) -> CountVectorsFeaturizer
        def pycloud_unpickle(file_name):
            # type: (Text) -> Any
            """Unpickle an object from file using cloudpickle."""
            from future.utils import PY2
            import cloudpickle

            with io.open(file_name, 'rb') as f:  # pragma: no test
                if PY2:
                    return cloudpickle.load(f)
                else:
                    return cloudpickle.load(f, encoding="latin-1")
        # meta = model_metadata.for_component(cls.name)

        if model_dir :
            file_name = 'intent_featurizer_count_vectors.pkl'
            featurizer_file = os.path.join(model_dir, file_name)
            return pycloud_unpickle(featurizer_file)
        else:
            logger.warning("Failed to load featurizer. Maybe path {} "
                           "doesn't exist".format(os.path.abspath(model_dir)))
            return CountVectorsFeaturizer(meta)

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory.
        Returns the metadata necessary to load the model again."""

        def pycloud_pickle(file_name, obj):
            # type: (Text, Any) -> None
            """Pickle an object to a file using cloudpickle."""
            import cloudpickle

            with io.open(file_name, 'wb') as f:
                cloudpickle.dump(obj, f)
        featurizer_file = os.path.join(model_dir, self.name + ".pkl")
        pycloud_pickle(featurizer_file, self)
        return {"featurizer_file": self.name + ".pkl"}
