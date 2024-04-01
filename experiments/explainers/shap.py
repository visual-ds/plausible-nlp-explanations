import numpy as np
import queue
import shap
import time
from tqdm.auto import tqdm

from ..explainer import Explainer

class DeterministicShapExplainer(shap.Explainer):
    """Deterministic version of SHAP explainer (in our application).

    The texts sent to the model are sorted. Therefore, they can be cached
    among multiple calls to the model.
    """
    def __init__(self, *args, **kwargs):

        # `shap.Explainer.__init__`` requires that
        # `self.__class__ == shap.Explainer``
        self.__class__ = shap.Explainer

        shap.Explainer.__init__(self, *args, **kwargs)

        # We change the model class to our inherited version of
        # `shap.models.Model.` This is because we modify the `__call__` method
        # of `shap.models.Model` to ensure reproducibility of the texts sent to
        # the model.
        if self.model.__class__ == shap.models.Model:
            self.model.__class__ = DeterministicShapModel

class DeterministicShapModel(shap.models.Model):
    def __call__(self, *args):
        """DeterministicShapModel.__call__.

        The texts are ordered before being sent to the model and this is undone
        after the model returns the predictions.
        """
        assert len(args) == 1, ("`args` is expected to be a tuple with only "
                                "one element which is an 2d array.")
        my_args = args[0]
        assert type(my_args) == np.ndarray, ("`args` is expected to be a tuple"
                                             " with only one element which is "
                                             "an 2d array.")
        my_args = my_args.tolist()
        my_args_sorted = sorted(my_args, key=lambda x: " ".join(x))
        my_args_sorted_np = np.array(my_args_sorted)
        
        to_return = shap.models.Model.__call__(self, my_args_sorted_np)
        assert type(to_return) == np.ndarray, ("`to_return` is expected to be "
                                               "a NumPy array.")
        
        # Undo sorting
        to_return_unsorted = [None]*len(to_return)
        for i, arg in enumerate(my_args):
            index = my_args_sorted.index(arg)
            my_args_sorted[index] = None
            to_return_unsorted[i] = to_return[index]

        return np.array(to_return_unsorted)

class ShapExplainer(Explainer):
    """SHAP explainer for explainability experiments."""
    def __init__(
        self,
        split_token="#",
    ):
        """Args:
            split_token (str): Token to use to split tokens. Default is "#".
        """
        self.split_token = split_token
        masker = shap.maskers.Text(
            self.custom_tokenizer,  # Use custom tokenizer to use the split
                                    # token
            mask_token=self.split_token,
            collapse_mask_token=False,
            output_type="array",  # The masker is able to output tokens instead
                                  # of a string. This is good, but it does not
                                  # solve the problem of SHAP not being able to
                                  # receive a list of tokens instead of a
                                  # complete string.
        )
        self.shap_explainer = DeterministicShapExplainer(
            self.model_fn,
            masker,
        )
    
    def custom_tokenizer(self, s, return_offsets_mapping=True):
        """Tokenize a string using split token.

        Because SHAP is not capable of receiving a list of tokens instead of a
        complete string, we need to tokenize the string using our method inside
        SHAP implementation. The following implementation of split and range is
        inspired by `shap.maskers._text.SimpleTokenizer`.
        """

        # SHAP assumes that an empty string should return nothing or there
        # should be special tokens. So we return nothing.
        if s == "":
            return {"input_ids": [], "offset_mapping": []}
        # Little trick to have split token in the final string passed to
        # self.model.
        elif s == self.split_token:
            return {
                "input_ids": [self.split_token],
                "offset_mapping": [(0, 1)]
            }
        # SHAP deals with the duality of tokens and their IDs. When SHAP
        # tokenizes only the split token, it is searching for its ID. However,
        # in my implementation, the ID is the token itself.
        
        tokenized_s = s.split(self.split_token)
        pos = 0
        offset_ranges = []
        for token in tokenized_s:
            offset_ranges.append((pos, pos + len(token)))
            pos += len(token) + len(self.split_token)
        out = {}
        out["input_ids"] = tokenized_s
        if return_offsets_mapping:
            out["offset_mapping"] = offset_ranges
        return out

    def model_fn(self, texts):
        """Model function for SHAP explainer.
        
        Args:
            texts (list of array): List of texts in array form (token by
                token).

        Returns:
            array-like of shape (n_samples, n_classes): Classifier
                probabilities.
        """
        texts = [list(text) for text in texts]
        dataset = []
        for text in texts:
            text = [token if token != self.split_token else "" \
                    for token in text]
            text = self.model.fill_missing_tokens(text, self.tokens)
            dataset.append({'tokens': text})
        return self.model.predict_proba(dataset)

    def explain_sample(self, model, sample):
        """Explain model on a single sample.
        
        Args:
            model (Model): Model to explain.
            sample (dict): Dictionary with 'tokens' (list of str) and 'label'
                (str).
                
        Returns:
            list of float: Explanation for the sample.
        """
        tokens = sample['tokens']
        text = self.split_token.join(tokens)
        label = sample['label']
        label = model.classes.index(label)

        # We need `tokens` and `model` to be available in `model_fn`.
        self.tokens = tokens
        self.model = model

        shap_values = self.shap_explainer(
            [text],
            batch_size=1000,
        )
        shap_values = shap_values.values[0]
        shap_values = shap_values[:, label]
        return list(shap_values)
