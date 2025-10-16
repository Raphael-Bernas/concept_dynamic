from main import opts, steps_list, get_dataset, retrieve_split_points, retrieve_metrics
from eurobert_XAI import Base_decomposer, Learning_process_decomposer, Higher_decomposer, available_base_decomposer
from datasets import load_dataset
import torch
import torch.nn.functional as F
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

if __name__ == "__main__":
    args = opts()

    print("Device:", args.device)
    if "SAE" in args.decomposer:
        args.device = "cpu"
    print("Model Name:", args.model_name)
    print("Decomposer:", args.decomposer)
    print("Split Point:", args.split_point)
    print("Test Name:", args.test_name)

    dataset = get_dataset(args.dataset, args.nb_dataset_samples*(1+int(args.higher_decomposition)))
    split_point = retrieve_split_points(args.split_point)
    metrics = retrieve_metrics(args.metrics)
    reference_step = steps_list([args.reference_steps], reference_setter=True)[0]
    if args.reference_split_point == 0:
        print("A reference split point is required in order to use a split point list, using default value 0 (can be changed using --reference_split_point)")

    if args.decomposer in available_base_decomposer() and not args.higher_decomposition:
        DEC = Base_decomposer(device=args.device, eurobert_repo_id=args.model_name, decomposer=args.decomposer, decomposition_method=args.decomposition_method, split_point=split_point, test_name=args.test_name, metrics=metrics, reference_step=reference_step, reference_split_point=args.reference_split_point, shared=args.unique_plot, verbose=args.verbose, gradient_plot=args.gradient_plot, log_scale=args.log_scale, kwargs=args.kwargs, forced_file=args.forced_file)
    elif args.decomposer == "LearningProcess":
        DEC = Learning_process_decomposer(device=args.device, eurobert_repo_id=args.model_name, split_point=split_point, test_name=args.test_name, decomposition_method=args.decomposition_method, reference_split_point=args.reference_split_point, shared=args.unique_plot, verbose=args.verbose, convergence=args.convergence, gradient_plot=args.gradient_plot, log_scale=args.log_scale, kwargs=args.kwargs, forced_file=args.forced_file)
    elif args.decomposer in available_base_decomposer() and args.higher_decomposition:
        DEC = Higher_decomposer(device=args.device, eurobert_repo_id=args.model_name, decomposer=args.decomposer, decomposition_method=args.decomposition_method, split_point=split_point, test_name=args.test_name, train_dataset=dataset[args.nb_dataset_samples:], metrics=metrics, reference_step=reference_step, reference_split_point=args.reference_split_point, shared=args.unique_plot, verbose=args.verbose, gradient_plot=args.gradient_plot, log_scale=args.log_scale, kwargs=args.kwargs, forced_file=args.forced_file)
    else:
        raise ValueError(f"Unsupported decomposer: {args.decomposer}. Supported decomposers are: NMF, SemiNMF, ConvexNMF, PCA, SparsePCA, ICA, SVD, KMeans, DictionaryLearning, LearningProcess.")
    
    DEC.load_dict()
    DEC.plot()

