import numpy as np
from copy import deepcopy
from joblib import Memory
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline

from ..model import Model
from .contrastive_logistic_regression import ContrastiveLogisticRegression

class TfidfLogisticRegression(Model):
    """TF-IDF + contrastive logistic regression model.

    Model for explainability experiments.
    """
    classes = None
    cache = None

    def __init__(
        self,
        tf_idf_ngram_range=(1, 1),
        t_svd_n_components=2,
        random_state=42,
        lr_tol=1e-4,
        lr_C=1.0,
        lr_max_iter=1e3,
        n_jobs=1,
        cross_val=False,
    ):
        """Init class.

        Args:
            tf_idf_ngram_range (tuple, optional): Range of ngrams for TF-IDF.
                Defaults to (1, 1).
            t_svd_n_components (int, optional): Number of components for
                Truncated Singular Value Decomposition. Defaults to 2.
            random_state (int, optional): Random state for random operations.
                Defaults to 42.
            lr_tol (float, optional): Tolerance of contrastive logistic
                regression. Defaults to 1e-4.
            lr_C (float, optional): Regularization parameter of contrastive
                logistic regression. Defaults to 1.0.
            lr_max_iter (int, optional): Maximum number of iterations of
                contrastive logistic regression. Defaults to 1e3.
            n_jobs (int, optional): Number of jobs for parallel processing.
                Defaults to 1.
            cross_val (bool, optional): Whether to use cross validation. Cross-
                -validation is only performed once, in the first fit. Defaults
                to False.
        """
        tf_idf = TfidfVectorizer(ngram_range=tf_idf_ngram_range)
        t_svd = TruncatedSVD(
            n_components=t_svd_n_components,
            random_state=random_state,
        )
        self.vectorizer = Pipeline([("tf_idf", tf_idf), ("t_svd", t_svd)])
        lr = ContrastiveLogisticRegression(
            tol=lr_tol,
            C=lr_C,
            random_state=random_state,
            max_iter=lr_max_iter,
            n_jobs=n_jobs,
        )
        self.clf = lr
        self.cross_val = cross_val

    def vectorizer_transform(self, X):
        """Transform list of texts into a vector representation using TF-IDF.

        Args:
            X (list of str): List of texts.

        Returns:
            np.array: Vector representation of texts.
        """
        return self.vectorizer.transform(X)

    def fit(self, dataset, w1, w2):
        """Fit model on the dataset with weights for original texts and
        rationale texts (must sum to 1).

        Args:
            dataset (list of dict): List of dictionaries with 'tokens' (list of
                str), 'rationales' (list of int), 'label' (str), and
                'negative_rationales' (list of list of int).
            w1 (float): Weight for original samples.
            w2 (float): Weight for rationale samples.
        """
        X, y, X_ratio, y_ratio = self.Xyz(dataset)
        self.vectorizer.fit(X)
        if self.cache is not None:
            X = self.cache(self.vectorizer_transform, X)
            X_ratio = np.array([
                self.cache(self.vectorizer_transform, rationales) \
                    for rationales in X_ratio
            ])
        else:
            X = self.vectorizer_transform(X)
            X_ratio = np.array([
                self.vectorizer_transform(rationales) for rationales in X_ratio
            ])
        if self.cross_val:
            self.fit_cross_val(X, y)
            self.cross_val = False
        self.clf.fit(X, y, X_ratio, y_ratio, w1, w2)

        self.classes = list(self.clf.classes_)

    def Xyz(self, dataset):
        """Transform dataset for training.
        
        Receive a list of dicts and return X, y, X_ratio, and y_ratio.
        
        Args:
            dataset (list of dict): List of dictionaries with 'tokens' (list of
                str), 'rationales' (list of int), 'label' (str), and
                'negative_rationales' (list of list of int).

        Returns:
            X (list of str): List of original texts.
            y (list of str): List of labels.
            X_ratio (list of list of str): List of rationale texts.
            y_ratio (list of str): List of rationale labels.
        """
        X, y, X_ratio, y_ratio = [], [], [], []
        for sample in dataset:
            tokens = sample['tokens']
            text = " ".join(tokens)
            X.append(text)

            label = sample['label']
            y.append(label)

            positive_rationales = sample['rationales']
            if len(positive_rationales) > 0:
                all_rationales = [positive_rationales] + \
                    sample['negative_rationales']
                all_rationales_text = []
                for rationales in all_rationales:
                    rationales = [
                        token for token, rationale in zip(tokens, rationales) \
                            if rationale
                    ]
                    rationales_text = " ".join(rationales)
                    all_rationales_text.append(rationales_text)
                X_ratio.append(all_rationales_text)
                y_ratio.append(label)

        return X, y, X_ratio, y_ratio

    def fit_cross_val(self, X, y):
        """Update `self.clf` with cross-validated `C` parameter."""            
        clf = LogisticRegressionCV(
            Cs=10,
            fit_intercept=True,
            cv=5,
            dual=False,  # n_samples > n_features
            penalty='l2',
            scoring='accuracy',
            solver='lbfgs',
            tol=self.clf.tol,
            max_iter=self.clf.max_iter,
            n_jobs=self.clf.n_jobs,
            verbose=0,
            refit=True,
            random_state=self.clf.random_state,
        )
        clf.fit(X, y)
        self.clf.C = clf.C_[0]

    def losses(self, dataset):
        """Get model losses (original and contrastive) for a dataset.

        Args:
            dataset (list of dict): List of dictionaries with 'tokens' (list of
                str), 'rationales' (list of int), 'label' (str), and
                'negative_rationales' (list of list of int).

        Returns:
            np.array: Losses for original and rationales texts.
        """
        X, y, X_ratio, y_ratio = self.Xyz(dataset)
        if self.cache is not None:
            X = self.cache(self.vectorizer_transform, X)
            X_ratio = np.array([
                self.cache(self.vectorizer_transform, rationales) \
                    for rationales in X_ratio
            ])
        else:
            X = self.vectorizer_transform(X)
            X_ratio = np.array([
                self.vectorizer_transform(rationales) for rationales in X_ratio
            ])
        return self.clf.losses(X, y, X_ratio, y_ratio)

    def predict(self, dataset):
        pass

    def predict_proba(self, dataset):
        """Get model probabilities for a dataset.

        Args:
            dataset (list of dict): List of dictionaries with 'tokens' (list of
                str).

        Returns:
            np.array: Model probabilities for a dataset.
        """
        X = self.X(dataset)
        if self.cache is not None:
            X = self.cache(self.vectorizer_transform, X)
        else:
            X = self.vectorizer_transform(X)
        return self.clf.predict_proba(X)

    def X(self, dataset):
        """Transform dataset for prediction.
        
        Receive a list of dicts and return X.
        
        Args:
            dataset (list of dict): List of dictionaries with 'tokens' (list of
                str).

        Returns:
            X (list of str): List of original texts.
        """
        X = []
        for sample in dataset:
            tokens = sample['tokens']
            text = " ".join(tokens)
            X.append(text)
        return X

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
        return perturbed_tokens

    def copy(self):
        """Return a copy of the model."""
        
        # Remove cache before copying
        cache = self.cache
        self.cache = None

        to_return = deepcopy(self)
        self.cache = cache
        return to_return
