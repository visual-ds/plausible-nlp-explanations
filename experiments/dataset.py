import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

class Dataset(ABC):
    """Generic dataset for explainability experiments."""
    def __init__(
        self,
        path,
        negative_rationales=None,
        shuffle=True,
        random_state=42
    ):
        """
        Args:
            path (str): Path to dataset file(s).
            negative_rationales (int): Number of negative rationales per
                sample. If None, then the negative rationale is the opposite of
                the positive one.
            shuffle (bool): If True, shuffle the dataset.
            random_state (int): Random seed.
        """
        self.random_state = random_state
        self.random = np.random.RandomState(self.random_state)
        self.load_dataset(path)
        self.treat_dataset()
        self.check_split_token()
        self.check_n_tokens()
        if shuffle:
            self.shuffle_dataset()
        self.add_negative_rationales(negative_rationales)
        self.split_train_test()

    @property
    @abstractmethod
    def dataset(self):
        """Returns:
            list of dict: List of dictionaries with 'tokens' (list of str),
                'rationales' (list of int), 'label' (str), and
                'negative_rationales' (list of list of int).
        """
        pass

    @property
    @abstractmethod
    def train_dataset(self):
        """Returns:
            list of dict: List of dictionaries with 'tokens' (list of str),
                'rationales' (list of int), 'label' (str), and
                'negative_rationales' (list of list of int).
        """
        pass

    @property
    @abstractmethod
    def test_dataset(self):
        """Returns:
            list of dict: List of dictionaries with 'tokens' (list of str),
                'rationales' (list of int), 'label' (str), and
                'negative_rationales' (list of list of int).
        """
        pass

    @property
    @abstractmethod
    def split_token(self):
        """Returns:
            str: Token to split text into tokens using explainers, e.g., Lime. Token's characters must not be present in the dataset.
        """
        pass

    @abstractmethod
    def load_dataset(self, path):
        """Load dataset from file(s).
        
        Args:
            path (str): Path to dataset file(s).
        """
        pass

    @abstractmethod
    def treat_dataset(self):
        """Treat dataset."""
        pass

    def shuffle_dataset(self):
        """Shuffle dataset."""
        self.random.shuffle(self.dataset)

    def add_negative_rationales(self, negative_rationales):
        """Add negative rationales to each sample.
        
        Args:
            negative_rationales (int): Number of negative rationales per
                sample. If None, then the negative rationale is the opposite of
                the positive one.
        """
        if negative_rationales is None:
            for sample in self.dataset:
                rationales = sample['rationales']
                sample['negative_rationales'] = [
                    self.opposite_negative_rationales(rationales)
                ]
        else:
            for sample in self.dataset:
                rationales = sample['rationales']
                sample['negative_rationales'] = [
                    self.random_negative_rationales(rationales) \
                        for _ in range(negative_rationales)
                ]

    def opposite_negative_rationales(self, rationales):
        """Create opposite negative rationales from positive rationales.

        Args:
            rationales (list of int): List of positive rationales.

        Returns:
            list of int: List of negative rationales.
        """
        return [1 - rationale for rationale in rationales]

    def random_negative_rationales(self, rationales):
        """Create random negative rationales from positive rationales.

        Args:
            rationales (list of int): List of positive rationales.

        Returns:
            list of int: List of negative rationales.
        """
        return list(self.random.permutation(rationales))

    def check_split_token(self):
        """Check if split token's characters are not present in the dataset."""
        all_chars = set([
            char for sample in self.dataset for token in sample['tokens'] \
                for char in token
        ])
        for char in set(self.split_token):
            assert char not in all_chars, \
                (
                    f"Split token \"{self.split_token}\" contains character "
                    f"\"{char}\"."
                )
            
    def check_n_tokens(self):
        """Check if all samples have two or more tokens."""
        for i, sample in enumerate(self.dataset):
            assert len(sample['tokens']) >= 2, \
                f"Sample {i} ({sample['tokens']}) has less than two tokens."

    def split_train_test(self):
        """Split dataset into train and test sets."""
        self.train_dataset, self.test_dataset = train_test_split(
            self.dataset,
            test_size=0.2,
            random_state=self.random_state,
            stratify=[sample['label'] for sample in self.dataset]
        )

    def sample_in_html(self, sample):
        """Args:
            sample (dict): Example dictionary with 'tokens' (list of str) and
                'rationales' (list of int). Can have other keys.

        Returns:
            str: HTML string.
        """
        html = ""
        span_style = "style=\"background-color:MediumSeaGreen;\""

        tokens = sample['tokens']
        rationales = sample['rationales']
        if len(rationales) != 0:
            for token, rationale in zip(tokens, rationales):
                if rationale == 1:
                    html += f"<span {span_style}>{token}</span>"
                else:
                    html += token
                html += " "
            html = html[:-1]
            html = html.replace(f"</span> <span {span_style}>", " ")
        else:
            html += " ".join(tokens)

        span_style = "style=\"color:Black;background-color:White;\""
        html = f"<span {span_style}>{html}</span>"
        return html
