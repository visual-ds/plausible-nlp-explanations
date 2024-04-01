from pathlib import Path

from .datasets import HatEval
from .evaluator import Evaluator
from .experiments import Experiments
from .metrics import accuracy, recall
from .models import VectorizerLogisticRegressionCache

class MyEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        self.hateval = HatEval()
        super().__init__(*args, **kwargs)
        
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

        probabilities = model.predict_proba(self.hateval.dataset)
        labels = [model.classes.index(sample['label']) for sample in self.hateval.dataset]

        result = accuracy(probabilities, labels)
        results["accuracy_hateval"] = result

        result = recall(probabilities, labels)
        for i, value in enumerate(result):
            results["recall_hateval_" + model.classes[i]] = value

        return results

class MyExperiments(Experiments):
   def __init__(
        self,
        datasets,
        explainers,
        models,
        negative_rationales,
        experiments_path,
        performance_metrics,
        explainability_metrics,
        random_state,
        gpu_device,
        batch_size,
    ):
        """Init class.

        Args:
            datasets (list of str): List of datasets to run experiments on.
            explainers (list of str): List of explainers to run experiments
                with.
            models (list of str): List of models to run experiments with.
            negative_rationales (list of int): List of number of negative
                rationales to run experiments with the contrastive rationale
                loss. If None, it uses the negative rationale as the opposite
                of the positive one. If 0, it does not use negative rationales;
                it uses the sigmoid loss instead.
            experiments_path (str): Path to save experiments to.
            performance_metrics (list of str): List of performance metrics to
                evaluate models with.
            explainability_metrics (list of str): List of explainability
                metrics to evaluate models with.
            random_state (int or None): Random state to use.
            gpu_device (int or None): GPU device to use.
            batch_size (int): Batch size to use.
        """
        self.datasets = datasets
        self.explainers = explainers
        self.models = models
        self.negative_rationales = negative_rationales
        self.experiments_path = self.create_path(experiments_path)
        self.performance_metrics = performance_metrics
        self.explainability_metrics = explainability_metrics
        if random_state is not None:
            self.random_state = random_state
        else:
            self.random_state = 42
        if gpu_device is not None:
            self.device = gpu_device
        else:
            self.device = 'cpu'
        self.batch_size = batch_size

        self.save_args()

        self.cache = VectorizerLogisticRegressionCache()
        self.cache.clear()
        self.evaluator = MyEvaluator()
        self.run()


def main_out_of_distribution_experiments():
    experiments = MyExperiments(
        datasets=["hatexplain"],
        explainers=["lime"],
        models=["distilbert"],
        negative_rationales=[2],
        experiments_path=str(Path(__file__).parents[1] / "data" /
                             "experiments"),
        performance_metrics=["accuracy", "recall"],
        explainability_metrics=["auprc", "comprehensiveness", "sufficiency"],
        random_state=46,
        gpu_device=0,
        batch_size=128,
    )
