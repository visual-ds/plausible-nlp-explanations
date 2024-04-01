import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AdamW, BertForSequenceClassification, BertTokenizer

from ..model import Model


class BertAttention(Model):
    """Bert attention model for comparison experiments.

    Reproduction of "BERT-HateXplain":
    B. Mathew, P. Saha, S. M. Yimam, C. Biemann, P. Goyal, and A. Mukherjee,
    “HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection,” in
    Proceedings of the AAAI Conference on Artificial Intelligence, Virtual:
    AAAI Press, May 2021, pp. 14867–14875. Accessed: Jul. 07, 2022. [Online].
    Available: https://ojs.aaai.org/index.php/AAAI/article/view/17745
    """
    classes = None

    def __init__(
            self,
            device=0,
            num_labels=2,
            n_jobs=1,
            random_state=42,
        ):
        """Args:
            device (str): GPU device to be used for model training. Defaults to
                0.
            num_labels (int): Number of labels of the dataset to be used to]
                train. Defaults to 2.
            n_jobs (int): Number of jobs for data loading. Defaults to 1.
            random_state (int): Random state for reproducibility. Defaults to
                42.
        """
        self.device = device
        self.n_jobs = n_jobs
        self.random_state = random_state


        # As referenced by the HateXplain paper
        model_name = "bert-base-uncased"
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        self.attention_loss_fn = torch.nn.CrossEntropyLoss()
        self.classification_loss_fn = torch.nn.CrossEntropyLoss()
        self.lr = 2e-5
        self.max_length = 128
        
        
        # As referenced by the HateXplain implementation
        # https://github.com/hate-alert/HateXplain
        self.softmax_temp = 1/5.0  # Softmax temperature is the inverse of
                                   # softmax variance
        assert self.model.config.classifier_dropout is None \
            and self.model.config.hidden_dropout_prob == 0.1
        self.num_heads = 6
        self.batch_size = 16
        self.epochs = 20
        self.eps = 1e-8
        # Assert number of BERT layers is 12
        # Authors use the 12th layer attention for supervision, we use the last
        assert len(self.model.bert.encoder.layer) == 12


        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        assert self.model.config.use_return_dict is True

    def fit_lambda(self, dataset, lambda_):
        """Fit model on (a portion of the) dataset.

        Fit model on the original texts giving more weight to the rationales in
        the attention(s) of the last layer. $\lambda$, as referenced by
        HateXplain, is calculated by $w_2/w_1$. Negative rationales are ignored
        by this method. Fit can only be made once (because of `create_classes`
        method).

        Args:
            dataset (list of dict): List of dictionaries with 'tokens' (list of
                str), 'rationales' (list of int), and 'label' (str).
            lambda_ (float): Weight for attention.
        """
        w1 = 1/(1 + lambda_)
        w2 = lambda_/(1 + lambda_)
        return self.fit(dataset, w1, w2)

    def fit(self, dataset, w1, w2):
        """Fit model on (a portion of the) dataset.

        Fit model on the original texts giving more weight to the rationales in
        the attention(s) of the last layer. $\lambda$, as referenced by
        HateXplain, is calculated by $w_2/w_1$. Negative rationales are ignored
        by this method. Fit can only be made once (because of `create_classes`
        method).

        Args:
            dataset (list of dict): List of dictionaries with 'tokens' (list of
                str), 'rationales' (list of int), and 'label' (str).
            w1 (float): Weight for original samples.
            w2 (float): Weight for attention.
        """
        X, y, rationale_attention_mask = self.Xya(dataset)

        input_ids, tokenization_attention_mask, token_type_ids = \
            self.tokenize_texts(X)

        self.create_classes(y)
        labels = [self.classes.index(label) for label in y]
        labels = torch.tensor(labels)

        rationale_attention_mask = \
            self.fix_attention_mask(rationale_attention_mask)
        
        dataset = TensorDataset(
            input_ids,
            tokenization_attention_mask,
            token_type_ids,
            labels,
            rationale_attention_mask,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_jobs,
            generator=torch.Generator().manual_seed(self.random_state),
        )
        lambda_ = w2/w1
        self.model.train()
        self.train_loop(dataloader, lambda_)

    def train_loop(self, dataloader, lambda_):
        """Train loop for the model.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for the
                dataset.
            lambda_ (float): Weight for attention.
        """

        # Model optimizer should be created inside this method and not as a
        # model attribute. This is because, otherwise, the optimizer would
        # accumulate memory on the GPU, and we would run out of memory after
        # several iterations.
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            eps=self.eps,
        )

        for epoch in tqdm(range(self.epochs), desc="Epochs of model training"):
            for batch in dataloader:
                batch = [item.to(self.device) for item in batch]

                input_ids_batch, tokenization_attention_mask_batch, \
                    token_type_ids_batch, labels_batch, \
                        rationale_attention_mask_batch = batch
                
                optimizer.zero_grad()                
                outputs = self.model(
                    input_ids=input_ids_batch,
                    attention_mask=tokenization_attention_mask_batch,
                    token_type_ids=token_type_ids_batch,
                    output_attentions=True,
                )
                logits = outputs.logits  # (batch_size, num_labels)
                classification_loss = self.classification_loss_fn(
                    logits,  # (batch_size, num_labels)
                    labels_batch,  # (batch_size)
                )

                # Last layer attention
                output_attentions = outputs.attentions[-1]  # (batch_size,
                                                            # num_heads,
                                                            # sequence_length,
                                                            # sequence_length)

                attention_loss = 0
                for i in range(self.num_heads):
                    attention_loss += self.attention_loss_fn(
                        output_attentions[:, i, 0, :],   # (batch_size,
                                                         # sequence_length)
                        rationale_attention_mask_batch,  # (batch_size,
                                                         # sequence_length)
                    )
                attention_loss /= self.num_heads  # As referenced by the paper
                                                  # (Appendix, "Attention
                                                  # supervision in BERT", 2nd
                                                  # paragraph). Their
                                                  # implementation, though, may
                                                  # be different:
                                                  # https://github.com/hate-alert/HateXplain/blob/01d742279dac941981f53806154481c0e15ee686/Models/bertModels.py#L57

                loss = classification_loss + lambda_*attention_loss
                loss.backward()
                optimizer.step()

    def Xya(self, dataset):
        """Transform dataset for training.
        
        Receive a list of dicts and return X, y, attention_mask.
        
        Args:
            dataset (list of dict): List of dictionaries with 'tokens' (list of
                str), 'rationales' (list of int), and 'label' (str).

        Returns:
            X (list of str): List of original texts.
            y (list of str): List of labels.
            attention_mask (list of list of int): List of attention masks.
        """
        X, y, attention_mask = [], [], []
        for sample in dataset:
            tokens = sample['tokens']
            text = " ".join(tokens)
            X.append(text)

            label = sample['label']
            y.append(label)

            rationales = sample['rationales']  # rationales can be a list of
                                               # 0s and 1s or an empty list.
            if len(rationales) > 0:  # If rationales is not empty
                rationales_list = []
                for token, rationale in zip(tokens, rationales):
                    assert rationale in {0, 1}, "Rationale must be 0 or 1."
                    tokenized = self.tokenizer(
                        token,
                        add_special_tokens=False,
                        return_token_type_ids=False,
                        return_attention_mask=False,
                    )['input_ids']
                    for _ in range(len(tokenized)):
                        rationales_list.append(rationale)
            else:
                tokenized = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    return_token_type_ids=False,
                    return_attention_mask=False,
                )['input_ids']
                count = len(tokenized)
                assert count > 0, "Tokenized text must not be empty."
                rationales_list = [0]*count

            # Apply softmax to rationales_list, as in HateXplain paper
            rationales_list = np.array(rationales_list)/self.softmax_temp
            rationales_list = np.exp(rationales_list)
            rationales_list = list(rationales_list/rationales_list.sum())

            attention_mask.append(rationales_list)

        return X, y, attention_mask

    def tokenize_texts(self, texts):
        """Tokenize a list of texts.

        Args:
            X (list of str): List of texts.

        Returns:
            input_ids (torch.Tensor): Tensor of input ids of shape (n_samples,
                max_length).
            attention_mask (torch.Tensor): Tensor of attention masks of shape
                (n_samples, max_length).
            token_type_ids (torch.Tensor): Tensor of token type ids of shape
                (n_samples, max_length).
        """
        tokenized_texts = self.tokenizer(
            text=texts,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        input_ids = tokenized_texts['input_ids']
        attention_mask = tokenized_texts['attention_mask']
        token_type_ids = tokenized_texts['token_type_ids']
        return input_ids, attention_mask, token_type_ids
    
    def create_classes(self, y):
        """Create classes from y.

        Args:
            y (list of str): List of labels.
        """
        assert self.classes is None, "Classes already created."
        self.classes = sorted(list(set(y)))

    def fix_attention_mask(self, attention_mask):
        """Truncate and pad attention masks to max_length.

        Args:
            attention_mask (list of list of int): List of attention masks.

        Returns:
            torch.Tensor: Tensor of attention masks of shape (n_samples,
                max_length).
        """
        attention_mask = [mask[:self.max_length - 2]   # We subtract two because
                          for mask in attention_mask]  # because of the special
                                                       # tokens
        attention_mask = [[0] + mask + [0]             # We add 0s because of
                          for mask in attention_mask]  # the special tokens
        attention_mask = [
            mask + [0]*(self.max_length - len(mask)) for mask in attention_mask
        ]
        attention_mask = torch.tensor(attention_mask)
        return attention_mask
        
    def predict_proba(self, dataset):
        """Predict probabilities for a dataset.

        Args:
            dataset (list of dict): List of dictionaries with 'tokens' (list of
                str).

        Returns:
            np.array: Model probabilities for a dataset.
        """
        X = self.X(dataset)
        input_ids, attention_mask, token_type_ids = self.tokenize_texts(X)
        dataset = TensorDataset(input_ids, attention_mask, token_type_ids)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_jobs,
        )

        self.model.eval()
        return self.eval_loop(dataloader)

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

    def eval_loop(self, dataloader):
        """Evaluation loop.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader.
            
        Returns:
            np.array: Model probabilities for a dataset.
        """
        logitss = []
        with torch.no_grad():
            for batch in dataloader:
                batch = [item.to(self.device) for item in batch]

                input_ids_batch, attention_mask_batch, token_type_ids_batch = \
                    batch
                
                outputs = self.model(
                    input_ids=input_ids_batch,
                    attention_mask=attention_mask_batch,
                    token_type_ids=token_type_ids_batch,
                )
                logits = outputs.logits
                logits = logits.detach().cpu().numpy()
                logitss.append(logits)
        logitss = np.concatenate(logitss)
        logitss = np.exp(logitss)  # (n_samples, n_classes)

        probs = logitss/logitss.sum(axis=1, keepdims=True)
        # (n_samples, n_classes)/(n_samples, 1) = (n_samples, n_classes)

        return probs

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
    
    def losses(self):
        raise NotImplementedError

    def predict(self, dataset):
        raise NotImplementedError

    def copy(self):
        """Return a copy of the model."""
        self.model.to('cpu')  # Send model to CPU before copying
        to_return = deepcopy(self)
        self.model.to(self.device)
        return to_return
