import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm

class Explainer(ABC):
    """Generic explainer for explainability experiments."""
    def explain(self, model, dataset, filter_no_rationales=False):
        """Explain model on (a portion of the) dataset.
        
        Args:
            model (Model): Model to explain.
            dataset (list of dict): List of dictionaries with 'tokens' (list of
                str) and 'label' (str).
            filter_no_rationales (bool): If True, filter samples with no
                rationales (those with rationales as an empty list). Defaults
                to False.

        Returns:
            list of list of float: Explanations for each sample in the dataset.
                Each list of float is the explanation for a single sample.
        """
        explanations = []
        for sample in tqdm(dataset, desc="Explaining dataset"):
            if filter_no_rationales and len(sample['rationales']) == 0:
                explanation = None
            else:
                explanation = self.explain_sample(model, sample)
            explanations.append(explanation)
        return explanations

    @abstractmethod
    def explain_sample(self, model, sample):
        """Explain model on a single sample.
        
        Args:
            model (Model): Model to explain.
            sample (dict): Dictionary with 'tokens' (list of str) and 'label'
                (str).
                
        Returns:
            list of float: Explanation for the sample.
        """
        pass

    def explanation_in_html(self, model, sample):
        scores = self.explain_sample(model, sample)
        max_abs_score = np.max(np.abs(scores))
        scores = scores/max_abs_score

        green = np.array([60, 179, 113])  # MediumSeaGreen
        white = np.array([255, 255, 255])  # White
        red = np.array([220, 20, 60])  # Crimson

        colors = []
        for score in scores:
            if score >= 0:
                color = score*green + (1 - score)*white
                colors.append(color)
            else:
                score *= -1
                color = score*red + (1 - score)*white
                colors.append(color)
        colors = [list(color) for color in colors]

        html = ""
        tokens = sample['tokens']
        for token, color in zip(tokens, colors):
            color = f"rgb({color[0]},{color[1]},{color[2]})"
            span_style = f"style=\"background-color:{color}\""
            html += f"<span {span_style}>{token}</span>"
            html += " "
        html = html[:-1]

        span_style = "style=\"color:Black;background-color:White;\""
        html = f"<span {span_style}>{html}</span>"
        return html
