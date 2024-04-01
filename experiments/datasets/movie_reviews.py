from pathlib import Path

from ..dataset import Dataset

class MovieReviews(Dataset):
    dataset = None
    train_dataset = None
    test_dataset = None
    split_token = "°"
    
    def __init__(
        self,
        path=None,
        negative_rationales=None,
        shuffle=True,
        random_state=42
    ):
        """
        Args:
            path (str): Path to Movie Reviews dataset folder.
            negative_rationales (int): Number of negative rationales per
                sample. If None, then the negative rationale is the opposite of
                the positive one.
            shuffle (bool): If True, shuffle the dataset.
            random_state (int): Random seed.
        """
        if path is None:
            here_path = Path(__file__).resolve()
            repo_path = here_path.parents[2]
            path = repo_path / "data" / "movie_reviews"
            path = str(path)
        super().__init__(path, negative_rationales, shuffle, random_state)
    
    def load_dataset(self, path):
        """Load dataset from file(s).
        
        Args:
            path (str): Path to dataset file(s).
        """
        path = Path(path).absolute()
        self.dataset = []
        for label in ["pos", "neg"]:
            files_path = path / f"withRats_{label}"
            text_files = list(files_path.glob("*.txt"))
            text_files.sort(key=lambda x: x.name)
            for text_file in text_files:
                with open(text_file, 'r') as f:
                    text = f.read()
                self.dataset.append({
                    'text': text,
                    'label': label,
                })

    def treat_dataset(self):
        """Treat dataset."""
        for sample in self.dataset:
            tag_1 = "<POS>" if sample['label'] == "pos" else "<NEG>"
            tag_2 = "</POS>" if sample['label'] == "pos" else "</NEG>"

            tokens = []
            rationales = []
            rationale = 0
            for token in sample['text'].split():
                if token == tag_1:
                    assert not rationale, f"{tag_1} was not closed."
                    rationale = 1
                elif token == tag_2:
                    assert rationale, f"{tag_1} was not opened."
                    rationale = 0
                elif tag_1 in token or tag_2 in token:
                    raise ValueError("Invalid token.")
                else:
                    tokens.append(token)
                    rationales.append(rationale)
            assert not rationale, f"{tag_1} was not closed."
            
            sample['tokens'] = tokens
            sample['rationales'] = rationales
            del sample['text']
