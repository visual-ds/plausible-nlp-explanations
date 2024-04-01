import numpy as np
from sklearn.metrics import auc, average_precision_score
from sklearn.metrics import precision_recall_curve

def accuracy(probabilities, labels):
    """Compute accuracy.

    Args:
        probabilities (array-like of shape (n_samples, n_classes)): Classifier
            probabilities.
        labels (list of int): Index labels.

    Returns:
        float: Accuracy.
    """
    predictions = np.argmax(probabilities, axis=1)
    labels = np.array(labels)
    return np.mean(predictions == labels)

def recall(probabilities, labels):
    """Compute recall per class.
    
    Args:
        probabilities (array-like of shape (n_samples, n_classes)): Classifier
            probabilities.
        labels (list of int): Index labels.

    Returns:
        list of float: Recall per class.
    """
    predictions = np.argmax(probabilities, axis=1)
    labels = np.array(labels)
    n_classes = probabilities.shape[1]
    recalls = []
    for i in range(n_classes):
        positives = np.sum(labels == i)
        true_positives = np.sum((labels == i) & (predictions == i))
        recalls.append(true_positives/positives)
    return recalls

def naive_auprc(scores, rationales, tokens, label, model, random):
    """Compute the area under the precision-recall curve.

    Naive implementation with Scikit-learn's `precision_recall_curve` and
    `auc`.

    Args:
        scores (list of float): Explanation scores.
        rationales (list of int): Index labels.
        tokens (list of str): List of tokens.
        label (str): Label.
        model (Model): The model to evaluate.
        random (np.random): Random number generator to use.

    Returns:
        float: Area under the precision-recall curve.
    """
    if sum(rationales) == 0:
        # Recall is undefined if there are no positive samples.
        return None
    scores = np.array(scores)
    scores[scores < 0] = 0
    max_score = np.max(scores)
    if max_score > 0:
        scores /= max_score
    precision, recall, _ = precision_recall_curve(rationales, scores)
    return auc(recall, precision)

def alternative_auprc(scores, rationales, tokens, label, model, random):
    """Compute the area under the precision-recall curve.

    Alternative implementation with Scikit-learn's `average_precision_score`.

    Args:
        scores (list of float): Explanation scores.
        rationales (list of int): Index labels.
        tokens (list of str): List of tokens.
        label (str): Label.
        model (Model): The model to evaluate.
        random (np.random): Random number generator to use.

    Returns:
        float: Area under the precision-recall curve.
    """
    if sum(rationales) == 0:
        # Recall is undefined if there are no positive samples.
        return None
    scores = np.array(scores)
    scores[scores < 0] = 0
    max_score = np.max(scores)
    if max_score > 0:
        scores /= max_score
    return average_precision_score(rationales, scores)

def comprehensiveness_aopc(scores, rationales, tokens, label, model, random):
    """Comprehensiveness measure of the explanations.
    
    Comprehensiveness is a measure of faithfulness. It measures how much the
    output of the model changes after removing the explanations. That is, if
    the change is big, it means the explanations are really important for the
    model prediction.

    Args:
        scores (list of float): Explanation scores.
        rationales (list of int): Index labels.
        tokens (list of str): List of tokens.
        label (str): Label.
        model (Model): The model to evaluate.
        random (np.random): Random number generator to use.

    Returns:
        float: Comprehensiveness.
    """
    length = len(tokens)
    label = model.classes.index(label)
    all_comprehensiveness = []
    for top in [1, 5, 10, 20, 50]:
        number_to_remove = int(np.ceil(length*top/100))
        indices_to_remove = set(np.argsort(scores)[::-1][:number_to_remove])
        filtered_tokens = [token if i not in indices_to_remove else "" \
            for i, token in enumerate(tokens)]
        filtered_tokens = model.fill_missing_tokens(filtered_tokens, tokens)
        probs = model.predict_proba([
            {'tokens': tokens},
            {'tokens': filtered_tokens},
        ])
        comprehensiveness = probs[0][label] - probs[1][label]
        all_comprehensiveness.append(comprehensiveness)
    return np.mean(all_comprehensiveness)

def random_comprehensiveness_aopc(scores, rationales, tokens, label, model,
                                  random):
    """Random comprehensiveness.
    
    Measure the comprehensiveness of random scores as a baseline.
    """
    if random is None:
        random = np.random
    scores = list(random.permutation(scores))
    return comprehensiveness_aopc(scores, rationales, tokens, label, model,
                                  random)

def comprehensiveness_10(scores, rationales, tokens, label, model, random):
    """Comprehensiveness measure of the explanations.
    
    Comprehensiveness is a measure of faithfulness. It measures how much the
    output of the model changes after removing the explanations. That is, if
    the change is big, it means the explanations are really important for the
    model prediction. Instead of taking the mean of comprehensiveness among
    different numbers of top tokens, we select 10 tokens."""
    length = len(tokens)
    label = model.classes.index(label)
    top = 10
    number_to_remove = int(np.ceil(length*top/100))
    indices_to_remove = set(np.argsort(scores)[::-1][:number_to_remove])
    filtered_tokens = [token if i not in indices_to_remove else "" \
        for i, token in enumerate(tokens)]
    filtered_tokens = model.fill_missing_tokens(filtered_tokens, tokens)
    probs = model.predict_proba([
        {'tokens': tokens},
        {'tokens': filtered_tokens},
    ])
    comprehensiveness = probs[0][label] - probs[1][label]
    return comprehensiveness

def random_comprehensiveness_10(scores, rationales, tokens, label, model,
                                random):
    """Random comprehensiveness.
    
    Measure the comprehensiveness of random scores as a baseline.
    """
    if random is None:
        random = np.random
    scores = list(random.permutation(scores))
    return comprehensiveness_10(scores, rationales, tokens, label, model,
                                random)

def sufficiency_aopc(scores, rationales, tokens, label, model, random):
    """Sufficiency measure of the explanations.
    
    Sufficiency is a measure of faithfulness. It measures how much the
    output of the model changes after keeping only the explanations. That is,
    if the change is low, it means the explanations are really important for
    the model prediction.

    Args:
        scores (list of float): Explanation scores.
        rationales (list of int): Index labels.
        tokens (list of str): List of tokens.
        label (str): Label.
        model (Model): The model to evaluate.
        random (np.random): Random number generator to use.

    Returns:
        float: Sufficiency.
    """
    length = len(tokens)
    label = model.classes.index(label)
    sufficiencies = []
    for top in [1, 5, 10, 20, 50]:
        number_to_keep = int(np.ceil(length*top/100))
        indices_to_keep = set(np.argsort(scores)[::-1][:number_to_keep])
        filtered_tokens = [token if i in indices_to_keep else "" \
            for i, token in enumerate(tokens)]
        filtered_tokens = model.fill_missing_tokens(filtered_tokens, tokens)
        probs = model.predict_proba([
            {'tokens': tokens},
            {'tokens': filtered_tokens},
        ])
        sufficiency = probs[0][label] - probs[1][label]
        sufficiencies.append(sufficiency)
    return np.mean(sufficiencies)

def random_sufficiency_aopc(scores, rationales, tokens, label, model, random):
    """Random sufficiency.
    
    Measure the sufficiency of random scores as a baseline.
    """
    if random is None:
        random = np.random
    scores = list(random.permutation(scores))
    return sufficiency_aopc(scores, rationales, tokens, label, model, random)

def sufficiency_10(scores, rationales, tokens, label, model, random):
    """Sufficiency measure of the explanations.
    
    Sufficiency is a measure of faithfulness. It measures how much the
    output of the model changes after keeping only the explanations. That is,
    if the change is low, it means the explanations are really important for
    the model prediction. Instead of taking the mean of sufficiency among
    different numbers of top tokens, we select 10 tokens."""
    length = len(tokens)
    label = model.classes.index(label)
    top = 10
    number_to_keep = int(np.ceil(length*top/100))
    indices_to_keep = set(np.argsort(scores)[::-1][:number_to_keep])
    filtered_tokens = [token if i in indices_to_keep else "" \
        for i, token in enumerate(tokens)]
    filtered_tokens = model.fill_missing_tokens(filtered_tokens, tokens)
    probs = model.predict_proba([
        {'tokens': tokens},
        {'tokens': filtered_tokens},
    ])
    sufficiency = probs[0][label] - probs[1][label]
    return sufficiency

def random_sufficiency_10(scores, rationales, tokens, label, model, random):
    """Random sufficiency.
    
    Measure the sufficiency of random scores as a baseline.
    """
    if random is None:
        random = np.random
    scores = list(random.permutation(scores))
    return sufficiency_10(scores, rationales, tokens, label, model, random)
