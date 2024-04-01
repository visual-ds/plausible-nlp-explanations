import numpy as np
from .metrics import accuracy, naive_auprc

class Evaluator:
    """Evaluate a model on a dataset with an explainer."""
    def evaluate(
        self,
        model,
        dataset,
        explainer,
        performance_metrics=[('accuracy', accuracy)],
        explainability_metrics=[('naive_auprc', naive_auprc)],
        random=None,
    ):
        """Evaluate a model on a dataset with an explainer.

        Given a list of performance and explainability metrics, evaluate the
        model on the dataset with the explainer and return a dictionary of
        results.

        Args:
            model (Model): The model to evaluate.
            dataset (list of dict): List of dictionaries with 'tokens' (list of
                str), 'rationales' (list of int), and 'label' (str).
            explainer (Explainer): The explainer to use.
            performance_metrics (list of tuple of (str, function), optional):
                List of performance metric names and functions to use. Check
                `accuracy` metric as an example. Defaults to `[('accuracy',
                accuracy)]`.
            explainability_metrics (list of tuple of (str, function),
                optional): List of explainability metric names and functions to
                use. Check `naive_auprc` metric as an example. Defaults to
                `[('naive_auprc', naive_auprc)]`.
            random (np.random or None, optional): Random number generator to
                use. Defaults to `None`.

        Returns:
            dict: Dictionary of results.
        """
        if random is None:
            random = np.random

        if len(performance_metrics) > 0:
            performance_results = self.evaluate_performance(model, dataset, \
                performance_metrics)
        else:
            performance_results = dict()

        if len(explainability_metrics) > 0:
            explainability_results = self.evaluate_explainability(model, \
                dataset, explainer, explainability_metrics, random)
        else:
            explainability_results = dict()

        results = {
            'performance': performance_results,
            'explainability': explainability_results,
        }
        return results

    def evaluate_performance(self, model, dataset, metrics):
        """Evaluate performance of a model on a dataset with a metric.

        Args:
            model (Model): The model to evaluate.
            dataset (list of dict): List of dictionaries with 'tokens' (list of
                str) and 'label' (str).
            metrics (list of tuple of (str, function)): List of performance
                metric names and functions to use.

        Returns:
            dict: Dictionary of results in the form `{metric_name:
                metric_value}`.
        """
        probabilities = model.predict_proba(dataset)
        labels = [model.classes.index(sample['label']) for sample in dataset]
        results = dict()
        for metric_name, metric in metrics:
            result = metric(probabilities, labels)
            if type(result) == list:
                for i, value in enumerate(result):
                    results[metric_name + '_' + model.classes[i]] = value
            else:
                results[metric_name] = result
        return results

    def evaluate_explainability(self, model, dataset, explainer, metrics,
                                random):
        """Evaluate explainability of a model on a dataset with an explainer
        and a metric.

        Args:
            model (Model): The model to evaluate.
            dataset (list of dict): List of dictionaries with 'tokens' (list of
                str), 'rationales' (list of int), and 'label' (str).
            explainer (Explainer): The explainer to use.
            metrics (list of tuple of (str, function)): List of explainability
                metric names and functions to use.
            random (np.random): Random number generator to use.

        Returns:
            dict: Dictionary of results in the form `{metric_name:
                list of metric_value}`.
        """
        all_scores = explainer.explain(model, dataset, \
            filter_no_rationales=True)
        all_rationales = [sample['rationales'] for sample in dataset]
        all_tokens = [sample['tokens'] for sample in dataset]
        all_labels = [sample['label'] for sample in dataset]

        all_results = dict()
        for metric_name, metric in metrics:
            results = []
            for scores, rationales, tokens, label in zip(all_scores, \
                all_rationales, all_tokens, all_labels):
                if len(rationales) > 0:
                    result = metric(scores, rationales, tokens, label, model, \
                        random)
                else:
                    result = None
                results.append(result)
            all_results[metric_name] = results

        return all_results
