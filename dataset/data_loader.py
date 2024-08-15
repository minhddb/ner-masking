from dataclasses import dataclass
from typing import Dict, Iterable

from datasets import load_dataset
from datasets import Dataset


# TODO: Write an option to load custom dataset with following extensions: txt, tsv, csv and json
class CustomDatasetLoader:
    pass


@dataclass
class DatasetLoader:
    """
    """
    hf_dataset_name_or_path: str = None

    def __call__(self):
        dataset = load_dataset(self.hf_dataset_name_or_path)
        label2id = {label: i for i, label in enumerate(dataset["train"].features["ner_tags"].feature.names)}
        id2label = {i: label for label, i in label2id.items()}
        return dataset, label2id, id2label


class Data:
    def __init__(self, dataset: Dataset, mapping_dict: Dict):
        self.dataset = dataset
        self.mapping_dict = mapping_dict
        self.non_entity_id = list(self.mapping_dict.keys())[-1]
        self.label_name = "ner_tags"

    def sequence_generator(self):
        for sequence in self.convert_id_to_entity():
            yield sequence["tokens"], sequence["ner_tags"]

    def convert_id_to_entity(self):
        for seq in self._yield_from(self.dataset):
            sequence = {"id": None, "tokens": [], "ner_tags": []}
            sequence.update({"id": seq["id"], "tokens": seq["tokens"],
                             "ner_tags": [self.mapping_dict[tag] for tag in seq[self.label_name]]})
            yield sequence

    def get_entity_sequences(self, tag_name: str = "ner_tags"):
        """
        Return all sequences containing at least one entity as list.
        :param tag_name: Name of entity column.
        """
        return [sequence for sequence in self._yield_from(self.dataset) if not all(tag == self.non_entity_id for tag in sequence[tag_name])]

    @staticmethod
    def _yield_from(iterable: Iterable):
        yield from iterable
