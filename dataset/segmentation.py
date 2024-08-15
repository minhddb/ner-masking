from typing import List
from itertools import combinations


class SequenceSegmentation:
    def __init__(self, sequence: List[str], labels: List[str]):
        self.sequence = sequence
        self.labels = labels

    def __len__(self):
        """ Return number of segment combinations when len() is called"""
        list_of_segments = [segment[0] for segment in self.get_annotated_segment()]
        counter = 0
        for i in range(1, len(list_of_segments) + 1):
            for _ in combinations(list_of_segments, r=i):
                counter += 1
        return counter

    def get_tags_based_segments(self):
        """
        Split tokens sequence into different segments based on corresponding tags.
        :return: List of separated tokens segments
        """
        segments = []
        segment = []
        for i, _ in enumerate(self.labels):
            try:
                if self.labels[i] == "O":
                    segment.append(self.sequence[i])
                    if self.labels[i + 1].startswith("B-"):
                        segments.append(segment)
                        segment = []
                else:
                    segment.append(self.sequence[i])
                    if ((self.labels[i].startswith("I-")
                         or self.labels[i].startswith("B-"))
                            and not self.labels[i + 1].startswith("I-")):
                        segments.append(segment)
                        segment = []
            except IndexError:
                segments.append(segment)
        return segments

    def get_annotated_segment(self):
        """
        Yield annotated tokens sequence and their positions within the sequence for further processing.
        :return: List of segment tokens and list of position ids.
        E.g.: ["B-name", "I-name", "O", "O", "O"] --> ["Token_0", "Token_1"], [0, 1]
        """
        span_tokens = []
        tags = []
        positions = []
        for i, _ in enumerate(self.labels):
            try:
                if (self.labels[i].startswith("B-") or
                        self.labels[i].startswith("I-")
                ):
                    span_tokens.append(self.sequence[i])
                    tags.append(self.labels[i])
                    positions.append(i)
                    if (self.labels[i + 1] == "O"
                            or self.labels[i + 1].startswith("B-")
                    ):
                        yield span_tokens, tags, positions
                        span_tokens, tags, positions = [], [], []
            except IndexError:
                yield span_tokens, tags, positions
                span_tokens, tags, positions = [], [], []


if __name__ == "__main__":
    tokens = ["My", "name", "is", "Monkey", "D.", "Ruffy", "and", "I'm", "King", "of", "the", "pirate", "."]
    tags = ["O", "O", "O", "B-PER", "I-PER", "I-PER", "O", "O", "B-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "O"]
    assert len(tokens) == len(tags), f"{len(tokens)}, {len(tags)}"
    segmentation = SequenceSegmentation(tokens, tags)
    print()
