import argparse
import json
import numpy as np
from copy import deepcopy
from pprint import pprint
from time import time
from tqdm import tqdm

from .datasets import HateXplain
from .evaluator import Evaluator
from .experiments import Experiments
from .models import BertAttention

class Comparison(Experiments):
    """Run explainability comparison experiments with datasets, explainers,
    models, and losses.
    
    Mainly designed to compare our methodology with the HateXplain paper:
    B. Mathew, P. Saha, S. M. Yimam, C. Biemann, P. Goyal, and A. Mukherjee,
    “HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection,” in
    Proceedings of the AAAI Conference on Artificial Intelligence, Virtual:
    AAAI Press, May 2021, pp. 14867–14875. Accessed: Jul. 07, 2022. [Online].
    Available: https://ojs.aaai.org/index.php/AAAI/article/view/17745
    """
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
        n_jobs,
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
                it uses the sigmoid loss instead. Notice it may be the case the
                selected model(s) do(es) not support negative rationales, as is
                the case for `bert_attention`. In this case, this argument is
                ignored for the respective model(s).
            experiments_path (str): Path to save experiments to.
            performance_metrics (list of str): List of performance metrics to
                evaluate models with.
            explainability_metrics (list of str): List of explainability
                metrics to evaluate models with.
            random_state (int or None): Random state to use.
            gpu_device (int or None): GPU device to use.
            batch_size (int): Batch size to use. Notice it may be the case the
                selected model(s) do(es) not support batch size, as is the case
                for `bert_attention`. In this case, this argument is ignored
                for the respective model(s).
            n_jobs (int): Number of jobs to use.
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
        self.n_jobs = n_jobs

        self.save_args()

        self.evaluator = Evaluator()
        self.run()

    def save_args(self):
        """Save arguments to a JSON file."""
        args = {
            'datasets': self.datasets,
            'explainers': self.explainers,
            'models': self.models,
            'negative_rationales': self.negative_rationales,
            'experiments_path': str(self.experiments_path),
            'performance_metrics': self.performance_metrics,
            'explainability_metrics': self.explainability_metrics,
            'random_state': self.random_state,
            'gpu_device': self.device,
            'batch_size': self.batch_size,
            'n_jobs': self.n_jobs,
        }
        with open(self.experiments_path / "args.json", 'w') as f:
            json.dump(args, f, indent=4)

    def run_experiment(
        self,
        dataset,
        explainer,
        model,
        negative_rationales,
        performance_metrics,
        explainability_metrics,
    ):
        """Run explainability experiment for specific dataset, explainer,
        model, and number of negative rationales.

        Args:
            dataset (str): Dataset to run experiment on.
            explainer (str): Explainer to run experiment with.
            model (str): Model to run experiment with.
            negative_rationales (int): Number of negative rationales to run
                experiment with the contrastive rationale loss. If None, it
                uses the negative rationale as the opposite of the positive
                one. If 0, it does not use negative rationales; it uses the
                sigmoid loss instead. Notice it may be the case the selected
                model(s) do(es) not support negative rationales, as is the
                case for `bert_attention`. In this case, this argument is
                ignored for the respective model(s).
            performance_metrics (list of str): List of performance metrics to
                evaluate models with.
            explainability_metrics (list of str): List of explainability
                metrics to evaluate models with.
        """
        start = time()
        print(
            f"Running experiment with {dataset}, {explainer}, {model}, "
            f"{negative_rationales}."
        )
        path = self.experiments_path / (f"{dataset}-{explainer}-{model}-"
            f"{negative_rationales}")
        path.mkdir(parents=True, exist_ok=True)
        print(f"Saving results to {path}.")

        return_to_gpu = model in ["bert_attention"]
        model = self.select_model(model, dataset)
        dataset = self.select_dataset(dataset, negative_rationales)

        weights, models = self.explore_lambdas(model, dataset.train_dataset,
                                               return_to_gpu)
        if return_to_gpu:
            model.model.to('cpu')

        performance_metrics = \
            self.select_performance_metrics(performance_metrics)
        explainability_metrics = \
            self.select_explainability_metrics(explainability_metrics)
        all_results = []
        for model in tqdm(models):
            explainer_ = self.select_explainer(explainer, dataset)
            if return_to_gpu:
                model.model.to(model.device)
            results = self.evaluator.evaluate(
                model,
                dataset.test_dataset,
                explainer_,
                performance_metrics,
                explainability_metrics,
                np.random.RandomState(self.random_state),
            )
            all_results.append(results)
            if return_to_gpu:
                model.model.to('cpu')
        end = time()
        weights = weights[:, [1, 0]]  # Swap weights to use lambda in results
                                      # log.
        self.log_results(all_results, weights, path, end - start)

    def select_dataset(self, dataset, negative_rationales):
        """Select dataset.

        Args:
            dataset (str): Dataset to select.
            negative_rationales (int): Number of negative rationales to run
                experiment with the contrastive rationale loss. If None, it
                uses the negative rationale as the opposite of the positive
                one. If 0, it does not use negative rationales; it uses the
                sigmoid loss instead. Notice it may be the case the selected
                model(s) do(es) not support negative rationales, as is the
                case for `bert_attention`. In this case, this argument is
                ignored for the respective model(s).

        Returns:
            Dataset: Dataset object.
        """
        if dataset == "hatexplain_all":
            return HateXplain(
                negative_rationales=negative_rationales,
                shuffle=True,
                random_state=42,
                all_labels=True,
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}.")

    def select_model(self, model, dataset):
        """Select model.

        Args:
            model (str): Model to select.
            dataset (str): Dataset to select.

        Returns:
            Model: Model object.
        """
        if model == "bert_attention":
            if dataset == "hatexplain_all":
                num_labels = 3
            else:
                raise ValueError(f"Dataset {dataset} is not supported for "
                                  "model bert_attention yet (only "
                                  "hatexplain_all). It may be easy to add "
                                  "support for it though. It is necessary to "
                                  "indicate the number of labels through the "
                                  "variable `num_labels`. Please, check the "
                                  "code.")
            return BertAttention(
                device=self.device,
                num_labels=num_labels,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown model: {model}.")

    def explore_lambdas(self, model, dataset, return_to_gpu):
        """Explore lambdas.

        Args:
            model (Model): Model object.
            dataset (list of dict): List of dictionaries with 'tokens' (list of
                str), 'rationales' (list of int), 'label' (str), and
                'negative_rationales' (list of list of int).
            return_to_gpu (bool): Whether to return the model to the GPU after
                it is sent to CPU. It may happen when the model is copied.

        Returns:
            np.array: Weights of the experiment.
            list of Model: List of models of the experiment.
        """
        lambdas_list = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        weights, models = [], []
        for lambda_ in lambdas_list:
            model_copy = model.copy()
            if return_to_gpu:
                model_copy.model.to(model_copy.device)
            model_copy.fit_lambda(dataset, lambda_)
            if return_to_gpu:
                model_copy.model.to('cpu')
            weights.append([1.0, lambda_])
            models.append(model_copy)
        return np.array(weights), models

    def save_results(self, all_results, weights, path):
        """Save the results of the experiment.
        
        Args:
            all_results (list of dict): Results of the experiment.
            weights (np.array): Weights of the experiment.
            path (Path): Path to save the results to.
        """
        w1 = list(weights[:, 0])
        data = []
        for i in range(len(w1)):
            line = {
                'weight': self.safe_float(w1[i]),
                'performance': deepcopy(all_results[i]['performance']),
                'explainability': deepcopy(all_results[i]['explainability']),
            }
            for key, value in line['performance'].items():
                line['performance'][key] = self.safe_float(value)
            for key, value in line['explainability'].items():
                line['explainability'][key] = [self.safe_float(v)
                                               for v in value]
            data.append(line)
        with open(path / "results.jsonl", 'w') as f:
            for line in data:
                json.dump(line, f)
                f.write("\n")

    def safe_float(self, value):
        """Convert value to float.

        Convert value to float if it is a NumPy float. Does nothing otherwise.

        Args:
            value: Value to convert.

        Returns:
            Converted value.
        """
        if isinstance(value, np.float16):
            return float(value)
        elif isinstance(value, np.float32):
            return float(value)
        elif isinstance(value, np.float64):
            return float(value)
        elif isinstance(value, np.float128):
            return float(value)
        else:
            return value


def create_parser():
    parser = argparse.ArgumentParser(
        prog="Comparison experiments",
        description="Run comparison explainability experiments.",
    )

    parser.add_argument(
        "--datasets",
        choices=["hatexplain_all"],
        default="hatexplain_all",
        help="Datasets to run experiments on. \"all\" means all labels.",
        nargs="+",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--explainers",
        choices=["lime", "shap"],
        default="lime",
        help="Explainers to run experiments with.",
        nargs="+",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--models",
        choices=["bert_attention"],
        default="bert_attention",
        help="Models to run experiments with.",
        nargs="+",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--negative_rationales",
        help=(
            "Number of negative rationales to use for each sample through the "
            "contrastive rationale loss. If None, it uses the negative "
            "rationale as the opposite of the positive one. If 0, it does not "
            "use negative rationales; it uses the sigmoid loss instead. Notice"
            "it may be the case the selected model(s) do(es) not support "
            "negative rationales, as is the case for `bert_attention`. In "
            "this case, this argument is ignored for the respective model(s)."
        ),
        nargs="+",
        required=True,
        type=lambda value: None if value == "None" else int(value),
    )
    parser.add_argument(
        "--experiments_path",
        help="Path to save experiments.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--performance_metrics",
        choices=["accuracy", "recall"],
        help="Performance metrics to use.",
        nargs="*",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--explainability_metrics",
        choices=["auprc", "comprehensiveness", "sufficiency"],
        help="Explainability metrics to use.",
        nargs="*",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--random_state",
        help="Random state to use.",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--gpu_device",
        help="GPU device to use.",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--batch_size",
        help=(
            "Batch size to use. Notice it may be the case the selected "
            "model(s) do(es) not support batch size, as is the case for "
            "`bert_attention`. In this case, this argument is ignored for the "
            "respective model(s)."
        ),
        required=True,
        type=int,
    )
    parser.add_argument(
        "--n_jobs",
        help="Number of jobs to use.",
        default=1,
        required=False,
        type=int,
    )
    return parser

def main_comparison_experiments():
    parser = create_parser()
    args = parser.parse_args()
    print("Running comparison experiments with the following arguments:")
    pprint(vars(args))

    comparison = Comparison(
        datasets=args.datasets,
        explainers=args.explainers,
        models=args.models,
        negative_rationales=args.negative_rationales,
        experiments_path=args.experiments_path,
        performance_metrics=args.performance_metrics,
        explainability_metrics=args.explainability_metrics,
        random_state=args.random_state,
        gpu_device=args.gpu_device,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
    )
