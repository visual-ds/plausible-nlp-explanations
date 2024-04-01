import pandas as pd
from pathlib import Path

from ..dataset import Dataset

class HatEval(Dataset):
    dataset = None
    train_dataset = None
    test_dataset = None
    split_token = "Ð"
    def __init__(
        self,
        path=None,
    ):
        """
        Args:
            path (str): Path to HateXplain dataset file.
        """
        if path is None:
            here_path = Path(__file__).resolve()
            repo_path = here_path.parents[2]
            path = repo_path / "data" / "hateval2019"
            path = str(path)
        super().__init__(path, None, False, 42)

    def load_dataset(self, path):
        self.train_dataset = pd.read_csv(
            Path(path) / "hateval2019_en_train.csv",
            header=0,
            index_col="id",
        )
        self.test_dataset = pd.read_csv(
            Path(path) / "hateval2019_en_test.csv",
            header=0,
            index_col="id",
        )
        self.dev_dataset = pd.read_csv(
            Path(path) / "hateval2019_en_dev.csv",
            header=0,
            index_col="id",
        )

    def convert_label(self, label):
        if label == 1:
            return "hatespeech"
        elif label == 0:
            return "normal"
        else:
            assert False, f"Unknown label \"{label}\"."

    def treat_dataset(self):
        """Treat dataset."""
        train_dataset = []
        for _, row in self.train_dataset.iterrows():
            tokens = row['text'].split()
            if len(tokens) >= 2:
                train_dataset.append(
                    {
                        'tokens': row['text'].split(),
                        'rationales': [],
                        'label': self.convert_label(row['HS']),
                    }
                )
            else:
                print(
                    f"Removing sample with less than 2 tokens: {row['text']}"
                )
        self.train_dataset = train_dataset

        test_dataset = []
        for _, row in self.test_dataset.iterrows():
            tokens = row['text'].split()
            if len(tokens) >= 2:
                test_dataset.append(
                    {
                        'tokens': tokens,
                        'rationales': [],
                        'label': self.convert_label(row['HS']),
                    }
                )
            else:
                print(
                    f"Removing sample with less than 2 tokens: {row['text']}"
                )
        self.test_dataset = test_dataset

        dev_dataset = []
        for _, row in self.dev_dataset.iterrows():
            tokens = row['text'].split()
            if len(tokens) >= 2:
                dev_dataset.append(
                    {
                        'tokens': tokens,
                        'rationales': [],
                        'label': self.convert_label(row['HS']),
                    }
                )
            else:
                print(
                    f"Removing sample with less than 2 tokens: {row['text']}"
                )
        self.dev_dataset = dev_dataset

        self.dataset = self.train_dataset + self.test_dataset + self.dev_dataset
        self.test_dataset = self.test_dataset + self.dev_dataset
        del self.dev_dataset

    def split_train_test(self):
        pass
