import numpy as np
import torch
from copy import deepcopy
from sklearn.linear_model import LogisticRegressionCV
from transformers import AutoTokenizer, AutoModel

from ..model import Model
from .contrastive_logistic_regression import ContrastiveLogisticRegression

class DistilBertLogisticRegression(Model):
    """DistilBERT + contrastive logistic regression model.

    Model for explainability experiments.
    """
    classes = None
    cache = None

    def __init__(
        self,
        model_name="distilbert-base-uncased",
        device=0,
        batch_size=128,
        random_state=42,
        lr_tol=1e-4,
        lr_C=1.0,
        lr_max_iter=1e3,
        n_jobs=1,
        cross_val=False,
        use_auth_token=False,
        max_length=512,
        output="last_hidden_state",
    ):
        """Init class.

        Args:
            model_name (str): Name of the model to be used from Hugging Face.
                Defaults to "distilbert-base-uncased".
            device (str): Device to be used for model inference.
                Defaults to 0.
            batch_size (int): Batch size for model inference.
                Defaults to 128.
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
            use_auth_token (bool, optional): Whether to use auth token for
                Hugging Face authentication. Defaults to False.
            max_length (int, optional): Maximum length of input text. Defaults
                to 512. Used mainly for `bert-base-uncased` with HateXplain
                dataset (because this is what is done in the paper).
            output (str, optional): Output of model to be used. Defaults to
                "last_hidden_state". The other option is "pooler_output".
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_auth_token=use_auth_token,
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            use_auth_token=use_auth_token,
        )
        self.model.to(self.device)
        self.model.eval()

        lr = ContrastiveLogisticRegression(
            tol=lr_tol,
            C=lr_C,
            random_state=random_state,
            max_iter=lr_max_iter,
            n_jobs=n_jobs,
        )
        self.clf = lr
        self.cross_val = cross_val
        self.max_length = max_length
        self.output = output
        assert self.output in {"last_hidden_state", "pooler_output"}, \
            f"Invalid output: {self.output}."

    def vectorizer_transform(self, X):
        """Transform list of texts into a vector representation using
        DistilBERT.

        Args:
            X (list of str): List of texts.

        Returns:
            np.array: Vector representation of texts.
        """
        X_tokenized = self.tokenizer(
            X,
            add_special_tokens=True,
            padding='max_length',
            truncation='only_first',
            max_length=self.max_length,
            return_tensors='pt',
        )
        X_tokenized = X_tokenized.to(self.device)
        
        input_ids = X_tokenized['input_ids']
        attention_mask = X_tokenized['attention_mask']
        
        cls = []
        for i in range(0, len(X), self.batch_size):
            with torch.no_grad():
                outputs = self.model(
                    input_ids[i:i + self.batch_size],
                    attention_mask = attention_mask[i:i + self.batch_size]
                )
                if self.output == "last_hidden_state":
                    cls.append(outputs['last_hidden_state'][:, 0, :].cpu())
                elif self.output == "pooler_output":
                    cls.append(outputs['pooler_output'].cpu())
                else:
                    raise ValueError(f"Invalid output: {self.output}.")
                
        return torch.vstack(cls).numpy()

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

        if self.model_name == "prajjwal1/bert-mini" or \
            self.model_name == "visual-ds/bert-mini-hatexplain" or \
                self.model_name == "bert-base-uncased" or \
                    self.model_name == "bert-large-uncased":
            # In the specific case of `bert-mini`, we are simplifying the input
            # text to take advantage of truncation. We do not do this for
            # DistilBERT because of the already run experiments.
            # We also do this for `bert-base-uncased`. Ideally, this would be
            # done for all models, but we already have experiments that were
            # run without this simplification.
            X = [self.simplify(text) for text in X]
            X_ratio = [[self.simplify(text) for text in rationales] \
                for rationales in X_ratio]

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

    def simplify(self, text):
        """Simplify text by removing extra tokens.

        This is an heuristic to reduce the size of the input text, taking
        advantage of the fact text will be truncated anyway. It is specific to
        the `bert-mini` model but it could be used for DistilBERT too. We use
        the fact here that BERT tokenizer will always split at whitespace at
        least.

        Args:
            text (str): Text to be simplified.

        Returns:
            str: Simplified text.
        """
        tokens = text.split(" ")
        tokens = tokens[:550]  # 550 instead of 512 (or 510) to be safe
        return " ".join(tokens)

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
                    rationales_list = []
                    for token, rationale in zip(tokens, rationales):
                        if rationale:
                            rationales_list.append(token)
                        else:
                            tokenized = self.tokenizer(
                                token,
                                add_special_tokens=False,
                                return_attention_mask=False
                            )['input_ids']
                            for _ in range(len(tokenized)):
                                rationales_list.append(self.tokenizer.mask_token)
                    rationales_text = " ".join(rationales_list)
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

        if self.model_name == "prajjwal1/bert-mini" or \
            self.model_name == "visual-ds/bert-mini-hatexplain" or \
                self.model_name == "bert-base-uncased" or \
                    self.model_name == "bert-large-uncased":
            # In the specific case of `bert-mini`, we are simplifying the input
            # text to take advantage of truncation. We do not do this for
            # DistilBERT because of the already run experiments.
            # We also do this for `bert-base-uncased`. Ideally, this would be
            # done for all models, but we already have experiments that were
            # run without this simplification.
            X = [self.simplify(text) for text in X]
            X_ratio = [[self.simplify(text) for text in rationales] \
                for rationales in X_ratio]

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

    def predict_proba(self, dataset):
        """Get model probabilities for a dataset.

        Args:
            dataset (list of dict): List of dictionaries with 'tokens' (list of
                str).

        Returns:
            np.array: Model probabilities for a dataset.
        """
        X = self.X(dataset)

        if self.model_name == "prajjwal1/bert-mini" or \
            self.model_name == "visual-ds/bert-mini-hatexplain" or \
                self.model_name == "bert-base-uncased" or \
                    self.model_name == "bert-large-uncased":
            # In the specific case of `bert-mini`, we are simplifying the input
            # text to take advantage of truncation. We do not do this for
            # DistilBERT because of the already run experiments.
            # We also do this for `bert-base-uncased`. Ideally, this would be
            # done for all models, but we already have experiments that were
            # run without this simplification.
            X = [self.simplify(text) for text in X]

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
        final_tokens = []
        for token, perturbed_token in zip(original_tokens, perturbed_tokens):
            if perturbed_token != "":
                final_tokens.append(perturbed_token)
            else:
                tokenized = self.tokenizer(
                    token,
                    add_special_tokens=False,
                    return_attention_mask=False
                )['input_ids']
                for _ in range(len(tokenized)):
                    final_tokens.append(self.tokenizer.mask_token)
        return final_tokens

    def predict(self):
        pass

    def copy(self):
        """Return a copy of the model."""

        # Remove cache before copying
        cache = self.cache
        self.cache = None

        self.model.to('cpu')  # Send model to CPU before copying
        to_return = deepcopy(self)
        self.model.to(self.device)
        self.cache = cache
        return to_return
