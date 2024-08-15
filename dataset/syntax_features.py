import spacy
from spacy.tokens import Doc
from typing import Dict, List
from itertools import chain

from datasets import Dataset

from data_loader import Data


# TODO: Revise this part later for data analysis and syntax focused data augmentation

"""
class SyntaxFeatures(Data):
    def __init__(self, dataset: Dataset, mapping_dict: Dict, spacy_model=None, context_windows: int = 1):
        super().__init__(dataset, mapping_dict)
        if spacy_model is not None:
            self.nlp = spacy.load(spacy_model)
        else:
            self.nlp = None
        self.context_windows = context_windows

    def __call__(self):
        features = dict(tokens=[],
                        ner_tags=[],
                        entity_tokens=[],
                        entity_tags=[],
                        tokens_ids=[],
                        entity_tokens_ids=[],
                        context_ids=[],
                        context_windows=[],
                        dependencies=[],
                        head_ids=[],
                        pos=[]
                        )
        for (tokens,
             ner_tags,
             entity_tokens,
             entity_tags,
             tokens_ids,
             entity_tokens_ids,
             window_context_ids,
             dependency_rel,
             dependency_head_ids,
             part_of_speech) in self.extract_linguistics_features():
            features["tokens"].append(tokens)
            features["ner_tags"].append(ner_tags)
            features["entity_tokens"].append(entity_tokens)
            features["entity_tags"].append(entity_tags)
            features["tokens_ids"].append(tokens_ids)
            features["entity_tokens_ids"].append(entity_tokens_ids)
            features["context_ids"].append(window_context_ids)
            features["context_windows"].append(self.context_windows)
            features["dependencies"].append(dependency_rel)
            features["head_ids"].append(dependency_head_ids)
            features["pos"].append(part_of_speech)
        return Dataset.from_dict(features)

    def extract_linguistics_features(self):
        for tokens, ner_tags in self.get_sequence():
            tokens_ids = [i for i, _ in enumerate(tokens)]
            entity_tokens, entity_tags, entity_tokens_ids = self.get_entity(tokens, ner_tags)
            window_context_ids = self.get_windows_context_ids(tokens, ner_tags)
            dependency_rel = self.get_dependency_relations(tokens)
            dependency_head_ids = self.get_dependency_head_ids(tokens)
            part_of_speech = self.get_part_of_speech(tokens)

            yield (tokens,
                   ner_tags,
                   entity_tokens,
                   entity_tags,
                   tokens_ids,
                   entity_tokens_ids,
                   window_context_ids,
                   dependency_rel,
                   dependency_head_ids,
                   part_of_speech
                   )

    def get_dependency_relations(self, sequence: List):
        return [token.dep_ for token in self.nlp(Doc(self.nlp.vocab, sequence))]

    def get_dependency_head_ids(self, sequence: List):
        return [token.head.i for token in self.nlp(Doc(self.nlp.vocab, sequence))]

    def get_part_of_speech(self, sequence: List):
        return [token.pos_ for token in self.nlp(Doc(self.nlp.vocab, sequence))]
"""