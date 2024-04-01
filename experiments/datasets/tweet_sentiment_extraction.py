import pandas as pd
import re
import string
from pathlib import Path

from ..dataset import Dataset

class TweetSentimentExtraction(Dataset):
    dataset = None
    train_dataset = None
    test_dataset = None
    split_token = "°"

    def __init__(
        self,
        path=None,
        negative_rationales=None,
        shuffle=True,
        random_state=42,
        all_labels=False,
    ):
        """
        Args:
            path (str): Path to the Tweet Sentiment Extraction dataset file.
            negative_rationales (int): Number of negative rationales per
                sample. If None, then the negative rationale is the opposite of
                the positive one.
            shuffle (bool): If True, shuffle the dataset.
            random_state (int): Random seed.
            all_labels (bool): If True, use all labels. If False, use only
                positive and negative labels. Defaults to False.
        """
        self.all_labels = all_labels
        if path is None:
            here_path = Path(__file__).resolve()
            repo_path = here_path.parents[2]
            path = (
                repo_path / "data" / "tweet_sentiment_extraction" / "train.csv"
            )
            path = str(path)
        super().__init__(path, negative_rationales, shuffle, random_state)

    def load_dataset(self, path):
        """Load Tweet Sentiment Extraction dataset from file.
        
        Args:
            path (str): Path to the Tweet Sentiment Extraction dataset file.
        """
        self.dataset = pd.read_csv(path)

    def treat_dataset(self):
        # Filter classes
        if not self.all_labels:
            self.dataset = self.dataset[
                (self.dataset['sentiment'] == "positive") | \
                    (self.dataset['sentiment'] == "negative")
            ]
        else:
            self.dataset = self.dataset[
                (self.dataset['sentiment'] == "positive") | \
                    (self.dataset['sentiment'] == "negative") | \
                        (self.dataset['sentiment'] == "neutral")
            ]
        dataset = []
        for i in range(len(self.dataset)):
            row = self.dataset.iloc[i]
            text = row['text']
            if type(text) != str:
                print(f"Skipping row {i} because text is not a string.")
                continue
            assert len(text) > 0, "DataFrame entry is an empty string."
            rationale = row['selected_text']
            assert type(rationale) == str, \
                f"DataFrame entry is not a string but a {type(rationale)}."
            assert len(rationale) > 0, "DataFrame entry is an empty string."
            label = row['sentiment']

            # Extract character range of rationale
            ratio_length = len(rationale)
            for j in range(len(text)):
                begin = j
                if text[begin:begin + ratio_length] == rationale:
                    break
            end = begin + ratio_length
            assert end <= len(text)
            ratio_range = set(range(begin, end))

            # Tokenize
            regex_special_chars = ".+*?^$()[]{}|\\"
            punctuation = string.punctuation
            punctuation = [
                char if char not in regex_special_chars else re.escape(char) \
                for char in punctuation
            ]
            punctuation = "".join(punctuation)
            tokens = re.split(f"([\\s{punctuation}])", text)

            # Iterate over tokens to select rationales by range intersection
            rationales = []
            length = 0
            for token in tokens:
                token_range = set(range(length, length + len(token)))
                if ratio_range.intersection(token_range):
                    rationales.append(1)
                else:
                    rationales.append(0)
                length += len(token)

            # Filter space and empty tokens
            tokens_n_rationales = list(zip(tokens, rationales))
            tokens_n_rationales = [item for item in tokens_n_rationales \
                if item[0] not in {" ", ""}]
            tokens, rationales = zip(*tokens_n_rationales)

            # Guarantee all samples have two or more tokens
            if len(list(tokens)) < 2:
                continue

            dataset.append({
                'tokens': list(tokens),
                'rationales': list(rationales),
                'label': label
            })
        self.dataset = dataset
