import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timezone
from pathlib import Path
from pprint import pprint
from time import time
from tqdm import tqdm

from .datasets import HateXplain, MovieReviews, TweetSentimentExtraction
from .evaluator import Evaluator
from .explainers import LimeExplainer, ShapExplainer
from .metrics import accuracy, recall
from .metrics import alternative_auprc, naive_auprc
from .metrics import comprehensiveness_aopc, random_comprehensiveness_aopc
from .metrics import comprehensiveness_10, random_comprehensiveness_10
from .metrics import sufficiency_aopc, random_sufficiency_aopc
from .metrics import sufficiency_10, random_sufficiency_10
from .models import BertAttention, DistilBertLogisticRegression
from .models import TfidfLogisticRegression, VectorizerLogisticRegressionCache
from .nise import NISE

# To do:
# - Think about the order of the for loops
# - Improve results plotting

class Experiments:
    """Run explainability experiments with datasets, explainers, models, and
    losses."""
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
        self.evaluator = Evaluator()
        self.run()

    def create_path(self, path):
        """Create path.

        Args:
            path (str): Path to create.

        Returns:
            Path: Path object.
        """
        path = Path(path)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
        path = path / now
        path.mkdir(parents=True, exist_ok=True)
        return path

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
        }
        with open(self.experiments_path / 'args.json', 'w') as f:
            json.dump(args, f, indent=4)

    def run(self):
        """Run explainability experiments."""
        self.lr_C = 1.0
        self.cross_val = True
        for model in self.models:
            for dataset in self.datasets:
                for explainer in self.explainers:
                    for negative_rationales in self.negative_rationales:
                        self.run_experiment(
                            dataset,
                            explainer,
                            model,
                            negative_rationales,
                            self.performance_metrics,
                            self.explainability_metrics
                        )
                        self.cross_val = False
                self.cross_val = True

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
                sigmoid loss instead.
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

        return_to_gpu = model in ["distilbert", "bert_mini", "bert_128",
                                  "bert_128_fine_tuned", "bert_large"]
        model = self.select_model(model, dataset)
        dataset = self.select_dataset(dataset, negative_rationales)
        self.cache.clear()
        model.cache = self.cache

        nise = NISE(model, dataset.train_dataset, n_models=30)
        if return_to_gpu:
            model.model.to('cpu')
        solutions = nise.opt.solutionsList
        solutions = sorted(solutions, key=lambda x: x.w[0])
        models, losses, weights = [], [], []
        for solution in solutions:
            models.append(solution.x)
            losses.append(solution.objs)
            weights.append(solution.w)
        losses = np.array(losses)
        weights = np.array(weights)
        self.log_losses(losses, weights, path)

        self.lr_C = models[0].clf.C

        performance_metrics = self.select_performance_metrics(performance_metrics)
        explainability_metrics = self.select_explainability_metrics(explainability_metrics)
        all_results = []
        for model in tqdm(models):

            # Selecting the explainer every time is a little trick to decrease
            # running time. When the explainer is selected, it is "reset," so
            # the perturbations (in the case of LIME) are the same every time.
            # This way, because of cache, running time is decreased.
            explainer_ = self.select_explainer(explainer, dataset)

            if return_to_gpu:
                model.model.to(model.device)
            model.cache = self.cache
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
        self.cache.clear()
        end = time()
        self.log_results(all_results, weights, path, end - start)

    def select_dataset(self, dataset, negative_rationales):
        """Select dataset.

        Args:
            dataset (str): Dataset to select.
            negative_rationales (int): Number of negative rationales to run
                experiment with the contrastive rationale loss. If None, it
                uses the negative rationale as the opposite of the positive
                one. If 0, it does not use negative rationales; it uses the
                sigmoid loss instead.

        Returns:
            Dataset: Dataset object.
        """
        if dataset == "hatexplain":
            return HateXplain(
                negative_rationales=negative_rationales,
                shuffle=True,
                random_state=42,  # HateXplain has a fixed random state. This
                                  # is because of DistilBERT and bert-mini
                                  # fine-tuning.
            )
        elif dataset == "hatexplain_all":
            return HateXplain(
                negative_rationales=negative_rationales,
                shuffle=True,
                random_state=42,
                all_labels=True,
            )
        elif dataset == "movie_reviews":
            return MovieReviews(
                negative_rationales=negative_rationales,
                shuffle=True,
                random_state=42,  # We are keeping the data split
            )
        elif dataset == "tweet_sentiment_extraction":
            return TweetSentimentExtraction(
                negative_rationales=negative_rationales,
                shuffle=True,
                random_state=42,  # We are keeping the data split
            )
        elif dataset == "tweet_sentiment_extraction_all":
            return TweetSentimentExtraction(
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
        if model == "distilbert":
            if dataset == "hatexplain":
                model_name = "visual-ds/distilbert-base-uncased-hatexplain"
                use_auth_token = True
            else:
                model_name = "distilbert-base-uncased"
                use_auth_token = False
            return DistilBertLogisticRegression(
                model_name=model_name,
                device=self.device,
                batch_size=self.batch_size,
                random_state=self.random_state,
                lr_tol=1e-4,
                lr_C=self.lr_C,
                lr_max_iter=1e3,
                n_jobs=1,
                cross_val=self.cross_val,
                use_auth_token=use_auth_token,
            )
        elif model == "bert_mini":
            if dataset == "hatexplain":
                model_name = "visual-ds/bert-mini-hatexplain"
                use_auth_token = True
            else:
                model_name = "prajjwal1/bert-mini"
                use_auth_token = False
            return DistilBertLogisticRegression(
                model_name=model_name,
                device=self.device,
                batch_size=self.batch_size,
                random_state=self.random_state,
                lr_tol=1e-4,
                lr_C=self.lr_C,
                lr_max_iter=1e3,
                n_jobs=1,
                cross_val=self.cross_val,
                use_auth_token=use_auth_token,
            )
        elif model == "bert_128":
            model_name = "bert-base-uncased"
            use_auth_token = False
            max_length = 128
            output = "pooler_output"
            model =  DistilBertLogisticRegression(
                model_name=model_name,
                device=self.device,
                batch_size=self.batch_size,
                random_state=self.random_state,
                lr_tol=1e-4,
                lr_C=self.lr_C,
                lr_max_iter=1e3,
                n_jobs=1,
                cross_val=self.cross_val,
                use_auth_token=use_auth_token,
                max_length=max_length,
                output=output,
            )

            # As referenced by the HateXplain implementation
            # https://github.com/hate-alert/HateXplain
            # I do not think it makes any difference, but I am keeping it
            # here just in case.
            assert model.model.config.classifier_dropout is None \
                and model.model.config.hidden_dropout_prob == 0.1
            
            return model
        elif model == "bert_large":
            model_name = "bert-large-uncased"
            use_auth_token = False
            return DistilBertLogisticRegression(
                model_name=model_name,
                device=self.device,
                batch_size=self.batch_size,
                random_state=self.random_state,
                lr_tol=1e-4,
                lr_C=self.lr_C,
                lr_max_iter=1e3,
                n_jobs=1,
                cross_val=self.cross_val,
                use_auth_token=use_auth_token,
            )            
        elif model == "bert_128_fine_tuned":

            # Fine-tuning (classification) of BERT on HateXplain (all labels)
            if dataset == "hatexplain_all":
                num_labels = 3
                dataset = self.select_dataset(
                    "hatexplain_all",
                    negative_rationales=2,  # The number of negative rationales
                                            # does not matter here as it is not
                                            # used by `BertAttention`. And it
                                            # does not impact dataset train-
                                            # -test split.
                )
            else:
                raise ValueError(f"Dataset {dataset} not supported for model "
                                 f"{model}.")
            bert_attention = BertAttention(
                device=self.device,
                num_labels=num_labels,
                n_jobs=1,
                random_state=self.random_state,                
            )
            bert_attention.fit_lambda(dataset.train_dataset, 0)

            # Select model for the experiments
            model_name = "bert-base-uncased"
            use_auth_token = False
            max_length = 128
            output = "pooler_output"
            self.cross_val = True  # We cannot guarantee reproducibility of
                                   # `bert_attention`
            model =  DistilBertLogisticRegression(
                model_name=model_name,
                device=self.device,
                batch_size=self.batch_size,
                random_state=self.random_state,
                lr_tol=1e-4,
                lr_C=self.lr_C,
                lr_max_iter=1e3,
                n_jobs=1,
                cross_val=self.cross_val,
                use_auth_token=use_auth_token,
                max_length=max_length,
                output=output,
            )
            model.model.to('cpu')
            model.model = bert_attention.model.bert
            model.model.eval()

            # As referenced by the HateXplain implementation
            # https://github.com/hate-alert/HateXplain
            # I do not think it makes any difference, but I am keeping it
            # here just in case.
            assert model.model.config.classifier_dropout is None \
                and model.model.config.hidden_dropout_prob == 0.1
            
            return model
        elif model == "tf_idf":
            return TfidfLogisticRegression(
                tf_idf_ngram_range=(1, 1),
                t_svd_n_components=200,
                random_state=self.random_state,
                lr_tol=1e-4,
                lr_C=self.lr_C,
                lr_max_iter=1e3,
                n_jobs=1,
                cross_val=self.cross_val,
            )
        else:
            raise ValueError(f"Unknown model: {model}.")

    def select_explainer(self, explainer, dataset):
        """Select explainer.

        Args:
            explainer (str): Explainer to select.
            dataset (Dataset): Dataset object.

        Returns:
            Explainer: Explainer object.
        """
        if explainer == "lime":
            return LimeExplainer(
                num_samples=1000,
                random_state=self.random_state,
                split_token=dataset.split_token,
            )
        elif explainer == "shap":
            return ShapExplainer(split_token=dataset.split_token)
        else:
            raise ValueError(f"Unknown explainer: {explainer}.")

    def select_performance_metrics(self, performance_metrics):
        """Select performance metrics.

        Args:
            performance_metrics (list of str): List of performance metrics to
                evaluate models with.

        Returns:
            list of tuple of (str, function): List of tuple of metric name and
                metric function.
        """
        performance_metrics = list(set(performance_metrics))
        metrics_to_return = []
        for metric in performance_metrics:
            if metric == "accuracy":
                metrics_to_return.append(("accuracy", accuracy))
            elif metric == "recall":
                metrics_to_return.append(("recall", recall))
            else:
                raise ValueError(f"Unknown performance metric: {metric}.")
        return metrics_to_return

    def select_explainability_metrics(self, explainability_metrics):
        """Select explainability metrics.

        Args:
            explainability_metrics (list of str): List of explainability
                metrics to evaluate models with.

        Returns:
            list of tuple of (str, function): List of tuple of metric name and
                metric function.
        """
        explainability_metrics = list(set(explainability_metrics))
        metrics_to_return = []
        for metric in explainability_metrics:
            if metric == "auprc":
                metrics_to_return.append(("alternative_auprc", \
                    alternative_auprc))
                metrics_to_return.append(("naive_auprc", naive_auprc))
            elif metric == "comprehensiveness":
                metrics_to_return.append(("comprehensiveness_aopc", \
                    comprehensiveness_aopc))
                metrics_to_return.append(("random_comprehensiveness_aopc", \
                    random_comprehensiveness_aopc))
                metrics_to_return.append(("comprehensiveness_10", \
                    comprehensiveness_10))
                metrics_to_return.append(("random_comprehensiveness_10", \
                    random_comprehensiveness_10))
            elif metric == "sufficiency":
                metrics_to_return.append(("sufficiency_aopc", \
                    sufficiency_aopc))
                metrics_to_return.append(("random_sufficiency_aopc", \
                    random_sufficiency_aopc))
                metrics_to_return.append(("sufficiency_10", \
                    sufficiency_10))
                metrics_to_return.append(("random_sufficiency_10", \
                    random_sufficiency_10))
            else:
                raise ValueError(f"Unknown explainability metric: {metric}.")
        return metrics_to_return

    def log_losses(self, losses, weights, path):
        """Log the losses of the experiment.
        
        Args:
            losses (np.array): Losses of the experiment.
            weights (np.array): Weights of the experiment.
            path (Path): Path to save the losses to.
        """
        self.plot_losses(losses, weights, path)
        self.save_losses(losses, weights, path)

    def plot_losses(self, losses, weights, path):
        """Log the losses of the experiment.
        
        Args:
            losses (np.array): Losses of the experiment.
            weights (np.array): Weights of the experiment.
            path (Path): Path to save the losses to.
        """
        w1 = list(weights[:, 0])
        fig, ax = plt.subplots()
        sns.scatterplot(
            x="Cross-entropy loss",
            y="Contrastive loss",
            data=pd.DataFrame(
                losses,
                columns=["Cross-entropy loss", "Contrastive loss"]
            ),
            hue=w1,
            ax=ax,
        )
        plt.title("Cross-entropy loss vs. contrastive loss")
        ax.get_legend().remove()
        norm = plt.Normalize(min(w1), max(w1))
        cmap = sns.cubehelix_palette(as_cmap=True)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(
            sm,
            orientation="vertical",
            label="Cross-entropy loss weight",
        )
        plt.savefig(path / "losses.png")
        plt.close()

    def save_losses(self, losses, weights, path):
        """Log the losses of the experiment.
        
        Args:
            losses (np.array): Losses of the experiment.
            weights (np.array): Weights of the experiment.
            path (Path): Path to save the losses to.
        """
        w1 = list(weights[:, 0])
        data = []
        for i in range(len(w1)):
            data.append({
                'weight': w1[i],
                'cross_entropy_loss': losses[i, 0],
                'contrastive_loss': losses[i, 1],
            })
        with open(path / "losses.jsonl", 'w') as f:
            for line in data:
                json.dump(line, f)
                f.write('\n')

    def log_results(self, all_results, weights, path, time):
        """Log the results of the experiment.
        
        Args:
            all_results (dict): Results of the experiment.
            weights (np.array): Weights of the experiment.
            path (Path): Path to save the results to.
            time (float): Time taken to run the experiment.
        """
        self.plot_results(all_results, weights, path)
        self.save_results(all_results, weights, path)
        self.save_time(time, path)

    def plot_results(self, all_results, weights, path):
        """Plot the results of the experiment.
        
        Args:
            all_results (dict): Results of the experiment.
            weights (np.array): Weights of the experiment.
            path (Path): Path to save the results to.
        """
        w1 = list(weights[:, 0])
        for performance_metric in all_results[0]['performance'].keys():
            for explainability_metric in all_results[0]['explainability'].keys():
                x = []
                y = []
                for results in all_results:
                    x.append(results['performance'][performance_metric])
                    y.append(
                        np.mean(
                            self.filter_results(results['explainability'][explainability_metric])
                        )
                    )

                fig, ax = plt.subplots()
                sns.scatterplot(
                    x=performance_metric,
                    y=explainability_metric,
                    data=pd.DataFrame({
                        performance_metric: x,
                        explainability_metric: y,
                    }),
                    hue=w1,
                    ax=ax,
                )
                plt.title("Performance vs. explainability")
                ax.get_legend().remove()
                norm = plt.Normalize(min(w1), max(w1))
                cmap = sns.cubehelix_palette(as_cmap=True)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                fig.colorbar(
                    sm,
                    orientation="vertical",
                    label="Cross-entropy loss weight",
                )
                plt.savefig(path / f"{performance_metric}_{explainability_metric}.png")
                plt.close()

    def filter_results(self, results):
        """Filter the results.

        Remove the None and NaN values from the results.

        Args:
            results (list of float): Results to filter.
        
        Returns:
            list of float: Filtered results.
        """
        return [result for result in results \
            if result is not None and not np.isnan(result)]

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
            data.append({
                'weight': w1[i],
                'performance': all_results[i]['performance'],
                'explainability': all_results[i]['explainability'],
            })
        with open(path / "results.jsonl", 'w') as f:
            for line in data:
                json.dump(line, f)
                f.write("\n")        

    def save_time(self, time, path):
        """Save the time taken to run the experiment.
        
        Args:
            time (float): Time taken to run the experiment.
            path (Path): Path to save the time to.
        """
        with open(path / "time.txt", 'w') as f:
            f.write(str(time))     

def create_parser():
    parser = argparse.ArgumentParser(
        prog="Explainability experiments",
        description="Run explainability experiments.",
    )

    parser.add_argument(
        "--datasets",
        choices=["hatexplain", "hatexplain_all", "movie_reviews",
                 "tweet_sentiment_extraction",
                 "tweet_sentiment_extraction_all"],
        help="Datasets to run experiments on. \"all\" means all labels.",
        nargs="+",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--explainers",
        choices=["lime", "shap"],
        help="Explainers to run experiments with.",
        nargs="+",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--models",
        choices=["distilbert", "bert_mini", "bert_128", "bert_128_fine_tuned",
                 "tf_idf", "bert_large"],
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
            "use negative rationales; it uses the sigmoid loss instead."
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
        help="Batch size to use.",
        required=True,
        type=int,
    )

    # Things to consider giving access through command line arguments:
    # - paths (HateXplain, MovieReviews, VectorizerLogisticRegressionCache)
    # - n_models

    return parser

def main_experiments():
    parser = create_parser()
    args = parser.parse_args()
    print("Running experiments with the following arguments:")
    pprint(vars(args))

    experiments = Experiments(
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
    )
