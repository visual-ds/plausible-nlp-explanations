from lime.lime_text import LimeTextExplainer

from ..explainer import Explainer

class LimeExplainer(Explainer):
    """Lime explainer for explainability experiments."""
    def __init__(
        self,
        num_samples=5000,
        model_regressor=None,
        random_state=42,
        split_token="#",
    ):
        """Args:
            num_samples (int): Number of samples to use in explanation. Default
                is 5000.
            model_regressor (sklearn regressor to use in explanation). Defaults
                to `scikit-learn`'s `Ridge`. Check `LimeTextExplainer` for more
                details.
            random_state (int): Random seed. Default is 42.
            split_token (str): Token to use to split tokens. Default is "#".
        """
        self.num_samples = num_samples
        self.model_regressor = model_regressor
        self.split_token = split_token
        self.lime_text_explainer = LimeTextExplainer(
            split_expression=lambda text: text.split(self.split_token),
            bow=False,
            mask_string="",
            random_state=random_state,
        )

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

        def classifier_fn(perturbed_texts):
            """Classifier function for Lime explainer.
            
            Args:
                perturbed_texts (list of str): List of perturbed texts.
                
            Returns:
                array-like of shape (n_samples, n_classes): Classifier
                    probabilities.
            """
            dataset = []
            for perturbed_text in perturbed_texts:
                perturbed_tokens = perturbed_text.split(self.split_token)
                final_tokens = model.fill_missing_tokens(perturbed_tokens, tokens)
                dataset.append({'tokens': final_tokens})
            return model.predict_proba(dataset)

        num_features = len(tokens)
        explanation = self.lime_text_explainer.explain_instance(
            text,
            classifier_fn,
            labels=(label,),
            num_features=num_features,
            num_samples=self.num_samples,
            model_regressor=self.model_regressor,
        )
        scores = self.lime_explanation_to_scores(explanation, label)
        return scores

    def lime_explanation_to_scores(self, explanation, label):
        """Convert Lime explanation to scores.
        
        Args:
            explanation (Lime explanation): Lime explanation.
            label (int): Label (integer) of the sample.
            
        Returns:
            list of float: Explanation for the sample.
        """
        ids_n_scores = explanation.local_exp[label]
        sorted_ids_n_scores = sorted(ids_n_scores, key=lambda x: x[0])
        scores = list(zip(*sorted_ids_n_scores))[1]
        return scores
