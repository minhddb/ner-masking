from itertools import chain
from typing import Dict, List
import datasets
import random
import math


from dataset.utils import ContextEntityExtraction
from dataset import Data


class DatasetMasking(Data):
    def __init__(self, dataset: datasets.Dataset, mapping_dict: Dict, tag_name: str = "ner_tags"):
        super().__init__(dataset, mapping_dict)
        self.tag_name = tag_name

    def mask(self, strategy: str = "entity", windows_size: int = 1, mask_token: str = "[MASK]", p_mask: float = 0.5):
        """

        :param strategy: Masking strategy. Defualt: 'entity'. Options: "windows": Mask context window tokens,
         "outside": Mask contextual information outside of context windows.
        :param windows_size: Context windows for masking.
         :param mask_token:
        :param p_mask:  Masking probability. Default: 0.5
        """
        masks_dataset = dict(tokens=[], ner_tags=[])
        entity_sequences = [seq["tokens"] for seq in self.get_entity_sequences(tag_name=self.tag_name)]
        random.seed(p_mask)
        random_samples = random.sample(entity_sequences, math.floor(len(entity_sequences) * p_mask))
        for sequence, tags in self.sequence_generator():
            if sequence in random_samples:
                sequence_mask = SequenceMasking(sequence, tags, mask_token)
                if strategy == "entity":
                    for masked in sequence_mask.mask_entity():
                        sequence = masked
                if strategy == "windows":
                    for masked in sequence_mask.mask_context(windows_size=windows_size):
                        sequence = masked
                if strategy == "outside":
                    for masked in sequence_mask.mask_context(windows_size=windows_size, mask_windows=False):
                        sequence = masked
                if strategy == "all_context":
                    for masked in sequence_mask.mask_context(mask_windows=False, mask_all=True):
                        sequence = masked
                if strategy == "remove":
                    sequence = [mask_token for _ in range(len(sequence))]
            if not all(token == mask_token for token in sequence):
                masks_dataset["tokens"].append(sequence)
                masks_dataset["ner_tags"].append(tags)
        return datasets.Dataset.from_dict(masks_dataset)


class SequenceMasking:
    def __init__(self, sequence: List, tags: List, mask_token: str = "[MASK]"):
        self.sequence = sequence
        self.tags = tags
        self.mask_token = mask_token
        self.extractor = ContextEntityExtraction(self.sequence, self.tags)
        self.entity_tokens_ids = list(
            chain.from_iterable(
                self.extractor.extract_entity()[2]
            )
        )

    def mask_entity(self):
        """ Mask entity tokens based on tokens ids."""
        yield [
            token if i not in self.entity_tokens_ids else self.mask_token
            for i, token in enumerate(self.sequence)
        ]

    def mask_context(self, windows_size: int = 1, mask_windows: bool = True, mask_all: bool = False):
        """
        Mask contextual information within the given tokens sequence
        :param mask_all:
        :param windows_size: Size of context windows
        :param mask_windows: Boolean indicator whether tokens from context windows should be masked.
        Mask if True, else mask other contextual tokens
        """
        context_windows_tokens_ids = list(chain.from_iterable(
            self.extractor.extract_context_windows_ids(windows_size=windows_size)
        )
        )
        if mask_windows:
            # Mask context windows tokens and ignore tokens that belong to an entity
            yield [
                token if i not in context_windows_tokens_ids or i in self.entity_tokens_ids
                else self.mask_token
                for i, token in enumerate(self.sequence)
            ]
        else:
            if not mask_all:
                # Mask all contextual tokens that are not contained in list of context windows
                if context_windows_tokens_ids:
                    yield [
                        token if i in self.entity_tokens_ids or i in context_windows_tokens_ids
                        else self.mask_token
                        for i, token in enumerate(self.sequence)
                    ]
                else:
                    yield self.sequence
            else:
                # Mask all context tokens
                yield [self.mask_token if i not in self.entity_tokens_ids else token for i,token in enumerate(self.sequence)]


if __name__ == "__main__":
    tokens = ['The', 'European', 'Commission', 'said', 'on', 'Thursday', 'it', 'disagreed', 'with', 'German', 'advice',
              'to', 'consumers', 'to', 'shun', 'British', 'lamb', 'until', 'scientists', 'determine', 'whether', 'mad',
              'cow', 'disease', 'can', 'be', 'transmitted', 'to', 'sheep', '.']
    tags = ['O', 'B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

    sequence_masking = SequenceMasking(tokens, tags)

    masked_context_windows = sequence_masking.mask_context(windows_size=4)
    masked_all_context = sequence_masking.mask_context(mask_windows = False, mask_all=True)
    masked_entity = sequence_masking.mask_entity()

    for s, t, mcw, mac, me in zip(tokens, tags, masked_context_windows, masked_all_context, masked_entity):
        print(s, t, mcw, mac, me, sep="\n")
