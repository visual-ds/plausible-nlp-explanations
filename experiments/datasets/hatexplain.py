import json
import numpy as np
from pathlib import Path

from ..dataset import Dataset

class HateXplain(Dataset):
    """HateXplain dataset.
    
    Attributes:
        dataset (list of dict): List of dictionaries with 'tokens' (list of
            str), 'rationales' (list of int), 'label' (str), and
            'negative_rationales' (list of list of int).
        train_dataset (list of dict): List of dictionaries with 'tokens' (list
            of str), 'rationales' (list of int), 'label' (str), and
            'negative_rationales' (list of list of int).
        test_dataset (list of dict): List of dictionaries with 'tokens' (list
            of str), 'rationales' (list of int), 'label' (str), and
            'negative_rationales' (list of list of int).
    """
    dataset = None
    train_dataset = None
    test_dataset = None
    split_token = "#"

    def __init__(
        self,
        path=None,
        negative_rationales=None,
        shuffle=True,
        random_state=42,
        filter=True,
        all_labels=False,
    ):
        """
        Args:
            path (str): Path to HateXplain dataset file.
            negative_rationales (int): Number of negative rationales per
                sample. If None, then the negative rationale is the opposite of
                the positive one.
            shuffle (bool): If True, shuffle the dataset.
            random_state (int): Random seed.
            filter (bool): If True, filter the dataset after treating it.
                Filters include, for instance, samples from class offensive.
                When not filtering, negative rationales are not added, for
                instance. Defaults to True.
            all_labels (bool): If True, use all labels. If False, use only
                'normal' and 'hatespeech' labels. Defaults to False.
        """
        self.filter = filter
        self.all_labels = all_labels
        if path is None:
            here_path = Path(__file__).resolve()
            repo_path = here_path.parents[2]
            path = repo_path / "data" / "hatexplain" / "dataset.json"
            path = str(path)
        if self.filter:
            super().__init__(path, negative_rationales, shuffle, random_state)
        else:
            self.random_state = random_state
            self.random = np.random.RandomState(self.random_state)
            self.load_dataset(path)
            self.treat_dataset()

    def load_dataset(self, path):
        """Load HateXplain dataset from file.
        
        Args:
            path (str): Path to HateXplain dataset file.
        """
        with open(path, 'r') as f:
            self.dataset = json.load(f)

    def create_safe_2d_array(self, lists):
        """Create 2d array from list of lists.

        Avoid the creation of the array when the lists are not of the same
        length.

        Args:
            lists (list of list): List of lists.

        Returns:
            np.array: NumPy array (None if the lists are not of the same
                length).
        """
        assert type(lists) == list, "`lists` must be a list."
        lens = {len(list_) for list_ in lists}
        if len(lens) == 1:
            return np.array(lists)
        else:
            return None

    def get_consensus_from_opinions(self, opinions):
        """Given a list of opinions if each token is a rationale, return the
        consensus of the annotators.
        
        Args:
            opinions (list of list of int): List of opinions.

        Returns:
            list of int: Consensus of opinions (empty list if there is no
                opinion, None if there is no consensus).
        """
        assert type(opinions) == list, "`opinions` must be a list."
        if len(opinions) > 0:
            opinions = self.create_safe_2d_array(opinions)
            if opinions is not None:
                consensus = opinions.mean(axis=0) > 0.5
                return list(consensus.astype(int))
            else:
                # If it was not possible to create a 2d array, i.e., the
                # lists are not of the same length, return None.
                return None
        else:
            return []

    def get_label_from_annotators(self, annotators):
        """Extract the label consensus from the annotators.

        Annotators must agree on the label.

        Args:
            annotators (list of dict): List of dictionaries with 'label'
            (str).

        Returns:
            str: Label consensus ("normal" or "hatespeech", or None if there is
            no consensus).
        """
        assert type(annotators) == list, "`annotators` must be a list."
        assert len(annotators) == 3, \
            "`annotators` list must have 3 dictionaries."
        labels = [annotator['label'] for annotator in annotators]
        if labels.count("normal") >= 2:
            return "normal"
        elif labels.count("hatespeech") >= 2:
            return "hatespeech"
        elif labels.count("offensive") >= 2:
            if self.all_labels:
                return "offensive"
            else:
                return None
        else:
            return None

    def filter_dataset(self):
        """Filter dataset."""
        # Filter None labels and rationales
        self.dataset = [
            data for data in self.dataset \
                if data['label'] is not None and data['rationales'] is not None
        ]
        # Filter different length of tokens and rationales
        self.dataset = [
            data for data in self.dataset \
                if len(data['tokens']) == len(data['rationales']) or \
                    len(data['rationales']) == 0
        ]

    def treat_dataset(self):
        """Treat HateXplain dataset."""
        self.dataset = list(self.dataset.values())
        self.dataset = [{
            'tokens': data['post_tokens'],
            'rationales': self.get_consensus_from_opinions(data['rationales']),
            'label': self.get_label_from_annotators(data['annotators'])
        } for data in self.dataset]
        if self.filter:
            self.filter_dataset()
