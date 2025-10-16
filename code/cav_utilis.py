############################################################################################################
# This code is adapted from the TCAV implementation by Been Kim et al.:
# @misc{kim2018interpretabilityfeatureattributionquantitative,
#       title={Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)}, 
#       author={Been Kim and Martin Wattenberg and Justin Gilmer and Carrie Cai and James Wexler and Fernanda Viegas and Rory Sayres},
#       year={2018},
#       eprint={1711.11279},
#       archivePrefix={arXiv},
#       primaryClass={stat.ML},
#       url={https://arxiv.org/abs/1711.11279}, 
# }
#############################################################################################################

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
import math
from interpreto import ModelWithSplitPoints


def dataset_to_activations(
    dataset,
    splitted_model,
    tokenizer,
    split_points,
    batch_size: int = 32,
):
    """
    Converts a dataset of text samples into model activations at specified split points.
    """
    texts = [example["text"] for example in dataset]
    labels = [example["label"] for example in dataset]
    activations = {sp: [] for sp in split_points}
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        # inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        batch_activations = splitted_model.get_activations(batch_texts, activation_granularity=ModelWithSplitPoints.activation_granularities.SAMPLE)
        for sp in split_points:
            activations[sp].append(batch_activations[sp].detach().cpu())
    for sp in split_points:
        activations[sp] = torch.cat(activations[sp], dim=0)
    labels = torch.tensor(labels)
    return activations, labels

def build_dataset(concept_type: str, num_samples: int, seed: int = 42, split: str = "train") -> dict:
    if concept_type == "semantic":
        dataset_name = "glue"
        subset_name = "sst2"
        text_column = "sentence"
    elif concept_type == "syntactic":
        dataset_name = "glue"
        subset_name = "cola"
        text_column = "sentence"
    else:
        raise ValueError("concept_type must be 'semantic' or 'syntactic'")
    if split not in ["train", "validation"]:
        raise ValueError("split must be 'train' or 'validation'")
    dataset = load_dataset(dataset_name, subset_name, split=split)
    shuffled_dataset = dataset.shuffle(seed=seed)

    label_0 = [example for example in shuffled_dataset if example["label"] == 0]
    label_1 = [example for example in shuffled_dataset if example["label"] == 1]

    # Number of samples per label
    num_per_label = num_samples // 2

    # Randomly sample from each label group
    np.random.seed(seed)
    sampled_label_0 = np.random.choice(label_0, size=min(num_per_label, len(label_0)), replace=False)
    sampled_label_1 = np.random.choice(label_1, size=min(num_per_label, len(label_1)), replace=False)

    sampled_dataset = list(sampled_label_0) + list(sampled_label_1)
    np.random.shuffle(sampled_dataset)

    texts = [example[text_column] for example in sampled_dataset]
    labels = [example["label"] for example in sampled_dataset]

    return {"texts": texts, "labels": labels}

def train_cav(
    activations: torch.Tensor,
    labels: torch.Tensor,
    l2_reg: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> torch.Tensor:
    """
    Trains a Concept Activation Vector (CAV) using logistic regression.

    Args:
        activations (torch.Tensor): The activations from the model.
        labels (torch.Tensor): The binary labels for the concept.
        l2_reg (float): L2 regularization strength.
        max_iter (int): Maximum number of iterations for optimization.
        tol (float): Tolerance for stopping criteria.

    Returns:
        The trained CAV as a torch.Tensor.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    activations = activations.to(device)
    labels = labels.to(device)

    n_samples, n_features = activations.shape
    weights = torch.zeros(n_features, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([weights], lr=0.01)

    prev_loss = float('inf')
    for iteration in range(max_iter):
        optimizer.zero_grad()
        logits = activations @ weights
        loss = F.binary_cross_entropy_with_logits(logits, labels.float()) + l2_reg * torch.sum(weights ** 2)
        loss.backward()
        optimizer.step()

        if abs(prev_loss - loss.item()) < tol:
            break
        prev_loss = loss.item()

    return weights.cpu().detach()


def compute_gradients_for_tcav(
    dataset,
    splitted_model,
    tokenizer,
    split_points: list,
    batch_size: int = 32,
    targets: list[int] | None = None,
) -> dict:
    """
    Computes gradients of model outputs with respect to activations at split points.
    This is used for proper TCAV computation.
    
    Automatically detects if the model is a classification model and uses appropriate
    granularity: CLS_TOKEN for classification models, TOKEN for others.

    Args:
        dataset: A dataset object containing text samples and labels.
        splitted_model: A ModelWithSplitPoints that can compute gradients at specified split points.
        tokenizer: A tokenizer compatible with the model.
        split_points: List of layer indices where gradients are to be computed.
        batch_size: Number of samples to process in each batch.
        targets: List of target class indices. If None, uses all classes.

    Returns:
        A tuple containing:
            - A dictionary mapping each split point to its corresponding gradients.
            - A tensor of labels corresponding to the dataset samples.
    """
    texts = [example["text"] for example in dataset]
    labels = [example["label"] for example in dataset]
    
    def identity_encode(activations):
        return activations
    
    def identity_decode(concepts):
        return concepts
    
    # If targets not specified and it's a classification model, use all unique labels
    if targets is None:
        unique_labels = list(set(labels))
        if len(unique_labels) <= 10: 
            targets = unique_labels
    
    gradients = {sp: [] for sp in split_points}
    
    for sp in split_points:
        model_name = splitted_model._model.__class__.__name__
        if "ForSequenceClassification" in model_name:
            activation_granularity = splitted_model.activation_granularities.CLS_TOKEN
        else:
            # For non-classification models, use TOKEN granularity
            activation_granularity = splitted_model.activation_granularities.TOKEN
            
        gradient_list = splitted_model._get_concept_output_gradients(
            inputs=texts[:len(texts)],  # Process all texts
            encode_activations=identity_encode,
            decode_concepts=identity_decode,
            targets=targets,
            split_point=sp,
            activation_granularity=activation_granularity,
            aggregation_strategy=splitted_model.aggregation_strategies.MEAN,
            concepts_x_gradients=False,
            batch_size=batch_size
        )
        
        # Concatenate gradients from all samples
        # Each element in gradient_list has shape (t, g, d) where t=targets, g=granularity, d=features
        concatenated_gradients = []
        for grad_tensor in gradient_list:
            # grad_tensor shape depends on granularity:
            # - CLS_TOKEN: (t, 1, d) where g=1 (one CLS token per sample)
            # - TOKEN: (t, g, d) where g=number of tokens (excluding special tokens)
            
            if activation_granularity == splitted_model.activation_granularities.CLS_TOKEN:
                # For CLS_TOKEN granularity, g=1, so we squeeze that dimension
                if targets is None or len(targets) == 1:
                    grad_for_target = grad_tensor[0, 0, :] if grad_tensor.dim() == 3 else grad_tensor[0, :]
                else:
                    # If multiple targets, take the mean across targets
                    grad_for_target = grad_tensor.mean(dim=0).squeeze(0)
            else:
                # For TOKEN granularity, we need to aggregate across tokens
                if targets is None or len(targets) == 1:
                    # Take first target and aggregate across tokens (g dimension)
                    grad_for_target = grad_tensor[0, :, :].mean(dim=0)  # (g, d) -> (d,)
                else:
                    # If multiple targets, take mean across targets then across tokens
                    grad_for_target = grad_tensor.mean(dim=0).mean(dim=0)  # (t, g, d) -> (g, d) -> (d,)
            
            concatenated_gradients.append(grad_for_target)
        
        gradients[sp] = torch.stack(concatenated_gradients, dim=0)
    
    return gradients, torch.tensor(labels)

def compute_tcav_score(
    cav: torch.Tensor,
    gradients: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Computes the TCAV score for a given CAV and gradients.
    
    TCAV score measures the sensitivity of the model output to the concept direction.
    It computes the directional derivative of the output along the CAV direction.

    Args:
        cav (torch.Tensor): The Concept Activation Vector.
        gradients (torch.Tensor): The gradients of model output w.r.t. activations.
        labels (torch.Tensor): The binary labels for the concept.

    Returns:
        The TCAV score as a float.
    """
    # Compute directional derivatives along the CAV direction
    directional_derivatives = gradients @ cav
    
    # Select positive and negative examples
    positive_derivatives = directional_derivatives[labels == 1]
    negative_derivatives = directional_derivatives[labels == 0]

    if len(positive_derivatives) == 0 or len(negative_derivatives) == 0:
        return float('nan')

    # TCAV score is the fraction of examples where the directional derivative is positive
    tcav_score = (positive_derivatives > 0).float().mean().item()
    return float(tcav_score)

def cav_pipeline(
    concept_type: str,
    num_samples: int,
    splitted_model,
    tokenizer,
    split_points: list,
    batch_size: int = 32,
    l2_reg: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-4,
    test_size: float = 0.2,
    seed: int = 42,
    targets: list[int] | None = None,
    verbose: bool = False,
) -> dict:
    """
    Full pipeline to compute CAVs and TCAV scores for a given concept using proper gradient-based TCAV.

    Args:
        concept_type (str): The type of concept ("semantic" or "syntactic").
        num_samples (int): The number of samples to select from the dataset.
        splitted_model: A ModelWithSplitPoints that can compute gradients at specified split points.
        tokenizer: A tokenizer compatible with the model.
        split_points (list): List of layer indices where activations are to be extracted.
        batch_size (int): Number of samples to process in each batch.
        l2_reg (float): L2 regularization strength for CAV training.
        max_iter (int): Maximum number of iterations for CAV optimization.
        tol (float): Tolerance for stopping criteria in CAV training.
        test_size (float): Proportion of the dataset to include in the test split.
        seed (int): Random seed for reproducibility.
        targets (list[int] | None): Target class indices for gradient computation. If None, uses all classes.
        verbose (bool): Whether to print progress information.

    Returns:
        A dictionary containing the following keys:
            - "cavs": A dictionary of trained CAVs for each split point.
            - "tcav_scores": A dictionary of TCAV scores corresponding to each CAV.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Step 1: Data 
    dataset_train = build_dataset(concept_type, num_samples, seed=seed)
    dataset_test = build_dataset(concept_type, int(num_samples * test_size), seed=seed, split="validation")

    train_data = [{"text": t, "label": l} for t, l in zip(dataset_train["texts"], dataset_train["labels"])]
    test_data = [{"text": t, "label": l} for t, l in zip(dataset_test["texts"], dataset_test["labels"])]

    # Step 2: Feature Extraction for CAV Training (using activations)
    train_activations, train_labels = dataset_to_activations(
        train_data, splitted_model, tokenizer, split_points, batch_size
    )
    
    # Step 3: Gradient Computation for TCAV Score (using gradients)
    test_gradients, test_labels = compute_gradients_for_tcav(
        test_data, splitted_model, tokenizer, split_points, batch_size, targets
    )

    # Step 4: CAV Training and TCAV Score Computation
    cavs = {}
    tcav_scores = {}
    if verbose:
        print(f"Training CAVs for concept '{concept_type}' with {num_samples} samples...")
        print("Using gradient-based TCAV computation...")
        
    for sp in split_points:
        # Train CAV using activations
        cav = train_cav(
            train_activations[sp],
            train_labels,
            l2_reg=l2_reg,
            max_iter=max_iter,
            tol=tol,
        )
        
        # Compute TCAV score using gradients
        tcav_score = compute_tcav_score(
            cav, test_gradients[sp], test_labels
        )
        
        cavs[sp] = cav.cpu().numpy()
        tcav_scores[sp] = np.array(tcav_score)
        
        if verbose:
            print(f"Split Point {sp}: TCAV Score = {tcav_score:.4f}")
    
    return {
        "cavs": cavs,
        "tcav_scores": tcav_scores,
    }