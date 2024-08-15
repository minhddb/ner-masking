from itertools import chain
from datasets import Dataset
from typing import Dict, List

from dataset import Data
from dataset import SequenceSegmentation


class ContextEntityExtraction:
    """ Extract entity or context windows segments."""

    def __init__(self, sequence: List, tags: List):
        self.sequence = sequence
        self.tags = tags
        self.segmentation = SequenceSegmentation(self.sequence, self.tags)

    def extract_entity(self):
        """ Extract entity tokens, tags and ids from input sequence."""
        entity_tokens = []
        entity_tags = []
        entity_tokens_ids = []
        for segment, entity, tokens_ids in self.segmentation.get_annotated_segment():
            entity_tokens.append(segment)
            entity_tags.append(entity)
            entity_tokens_ids.append(tokens_ids)
        return entity_tokens, entity_tags, entity_tokens_ids

    def extract_context_windows_ids(self, windows_size: int = 1):
        """ Extract position ids for tokens within a context window."""
        # segmentation = SequenceSegmentation(sequence, tags)
        context_tokens_ids = []
        for segment, entity, entity_tokens_pos in self.segmentation.get_annotated_segment():
            context_ids = self._get_context_windows(token_ids=entity_tokens_pos,
                                                    n_tokens=len(self.sequence),
                                                    windows_size=windows_size
                                                    )
            context_tokens_ids.append(context_ids)
        return context_tokens_ids

    @staticmethod
    def _get_context_windows(token_ids: List, n_tokens: int, windows_size: int = 1):
        """
        Extract left and right context from sequence
        :param token_ids: Input token ids as list
        :param n_tokens: Number of tokens
        :param context_windows: Context window size
        """
        l_context = [i for i in range(token_ids[0] - windows_size, token_ids[0]) if i >= 0]
        r_context = [i for i in range(token_ids[-1] + 1, token_ids[-1] + windows_size + 1) if i <= n_tokens - 1]
        return list(chain.from_iterable([l_context, r_context]))


class EntityCount(Data):
    """ Get entity distribution from dataset."""

    def __init__(self, dataset: Dataset, mapping_dict: Dict):
        super().__init__(dataset, mapping_dict)

    def occurrence_ratio(self):
        """ Compute distribution ratio for each entity."""
        occurrence_dict = self.count_occurrences()
        return {ent: round(cnt / sum(occurrence_dict.values()), 4) for ent, cnt in occurrence_dict.items()}

    def count_occurrences(self):
        """ Count entity occurrences in dataset."""
        dist_dict = {}
        for sequence, tags in self.sequence_generator():
            segmentation = SequenceSegmentation(sequence, tags)
            for segment, annotated, _ in segmentation.get_annotated_segment():
                entity = normalise_entity(annotated)
                if entity not in dist_dict:
                    dist_dict.update({entity: 1})
                else:
                    dist_dict[entity] += 1
        return {ent: count for ent, count in sorted(dist_dict.items(), key=lambda item: item[1])}


def normalise_entity(tags: List):
    """ Get entity class from B-tag."""
    return tags[0].strip("B-")
