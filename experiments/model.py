from abc import ABC, abstractmethod

class Model(ABC):
    """Generic model for explainability experiments."""
    @abstractmethod
    def fit(self, dataset, w1, w2):
        """Fit model on (a portion of the) dataset.

        Fit model on both original texts and rationale texts.

        Args:
            dataset (list of dict): List of dictionaries with 'tokens' (list of
                str), 'rationales' (list of int), 'label' (str), and
                'negative_rationales' (list of list of int).
            w1 (float): Weight for original samples.
            w2 (float): Weight for rationale samples.
        """
        pass
    
    @abstractmethod
    def losses(self):
        """Calculate and return loss for original samples and rationale
        samples.
        """
        pass

    @abstractmethod
    def predict(self, dataset):
        """Predict labels for a dataset.

        Args:
            dataset (list of dict): List of dictionaries with 'tokens' (list of
                str).
        """
        pass

    @abstractmethod
    def predict_proba(self, dataset):
        """Predict probabilities for a dataset.

        Args:
            dataset (list of dict): List of dictionaries with 'tokens' (list of
                str).
        """
        pass

    @abstractmethod
    def fill_missing_tokens(self, perturbed_tokens, original_tokens):
        """Fill missing tokens in a perturbed text.

        The model needs to inform how it would like a perturbed text (list of
        original tokens but with some of them as empty strings) to be filled
        in. Currently, this method is required by `LimeExplainer`, but it can
        be used for other explainers in the future.

        Args:
            perturbed_tokens (list of str): List of perturbed tokens.
            original_tokens (list of str): List of original tokens.

        Returns:
            list of str: List of filled in tokens.
        """
        pass

    @property
    @abstractmethod
    def classes(self):
        """Return list of classes."""
        pass

    @abstractmethod
    def copy(self):
        """Return a copy of the model."""
        pass
