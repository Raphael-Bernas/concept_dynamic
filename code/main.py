import argparse
import json
import torch
import time
from datasets import load_dataset, get_dataset_split_names, get_dataset_config_names
from eurobert_XAI import Base_decomposer, Learning_process_decomposer, Higher_decomposer, available_base_decomposer
import importlib.util
from pathlib import Path


def load_auto_metric(metric_path):
    if metric_path is None:
        return None

    path = Path(metric_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {metric_path}")

    spec = importlib.util.spec_from_file_location("user_metric", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {metric_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "auto_metric"):
        raise AttributeError(f"No function `auto_metric` found in {metric_path}")

    auto_metric = getattr(module, "auto_metric")

    if not callable(auto_metric):
        raise TypeError(f"`auto_metric` in {metric_path} is not callable")

    return lambda D, R, S, P, M: auto_metric(D, R, S, P, M)


def parse_kwargs(kwargs_str):
    if not kwargs_str:
        return {}
    try:
        parsed = json.loads(kwargs_str)
        if not isinstance(parsed, dict):
            raise argparse.ArgumentTypeError(f"kwargs must be a JSON object (dictionary), got {type(parsed).__name__}")
        return parsed
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON format for kwargs: {e}")

def opts() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EuroBERT XAI script")
    parser.add_argument(
        "--model_name",
        type=str,
        default="EuroBERT/EuroBERT-210m",
        metavar="MOD",
        help="Name of the model for initialization (default: EuroBERT/EuroBERT-210m)",
    )
    parser.add_argument(
        "--decomposer",
        type=str,
        default="NMF",
        metavar="DEC",
        help="Decomposer to use (default: NMF)",
    )
    parser.add_argument(
        "--higher_decomposition",
        action="store_true",
        help="Enable more complex decomposition (default: False)",
    )
    parser.add_argument(
        "--decomposition_method",
        type=str,
        default="default",
        metavar="DM",
        help="Decomposition method to use inside the decomposer (default: default)",
    )
    parser.add_argument(
        "--split_point",
        type=str,
        default="model.layers.1.mlp",
        metavar="SP",
        help="Split point(s) for the model (default: 'model.layers.1.mlp'). Provide a comma-separated string for multiple split points.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        metavar="DEV",
        help="Device to use for computation (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--test_name",
        type=str,
        default="test_210m" + time.strftime("%Y%m%d-%H%M%S"),
        metavar="TN",
        help="Name of the test (default: test_210m + timestamp)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=[i+1 for i in range(48)],
        nargs='+', 
        metavar="STEPS",
        help="List of steps to test (e.g. --steps 1 2 3 45 46 48) (default: [1, ..., 48])",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cornell-movie-review-data/rotten_tomatoes",
        metavar="DS",
        help="Dataset to use for testing (default: cornell-movie-review-data/rotten_tomatoes)",
    )
    parser.add_argument(
        "--nb_concepts",
        type=int,
        default=10,
        metavar="NC",
        help="Number of concepts to use from the dataset (default: 10)",
    )
    parser.add_argument(
        "--nb_dataset_samples",
        type=int,
        default=1000,
        metavar="NDS",
        help="Number of samples to use from the dataset (default: 1000)",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        metavar="SPLIT",
        help="Dataset split to use (default: train). Common options: train, test, validation",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        metavar="FIELD",
        help="Field name containing the text in the dataset (default: text). Common options: text, content, sentence, passage",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="default",
        metavar="MET",
        help="Metrics to use for evaluation (default: 'default'). Provide a comma-separated string for multiple metrics.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (default: False)",
    )
    parser.add_argument(
        "--gradient_plot",
        action="store_true",
        help="Enable gradient plot (default: False)",
    )
    parser.add_argument(
        "--log_scale",
        action="store_true",
        help="Enable log scale for plots (default: False)",
    )
    parser.add_argument(
        "--reference_split_point",
        type=int,
        default=0,
        metavar="RSP",
        help="Provide a int to decide which split point should be used as reference (default: 0)",
    )
    parser.add_argument(
        "--reference_steps",
        type=int,
        default=0,
        metavar="RS",
        help="Provide a int to decide which step should be used as reference (default: 0). If 0, the last step will be used as reference.",
    )
    parser.add_argument(
        "--visualization",
        action="store_true",
        help="Enable visualization (default: False)",
    )
    parser.add_argument(
        "--unique_plot",
        action="store_true",
        help="Enable unique plot (default: False)",
    )
    parser.add_argument(
        "--freeshold",
        type=int,
        default=5,
        metavar="FRE",
        help="Number of steps after which the dictionary is freed (default: 5)",
    )
    parser.add_argument(
        "--convergence",
        action="store_true",
        help="Force a comparison with the final value (default: False)",
    )
    parser.add_argument(
        "--metric_path",
        type=str,
        default=None,
        metavar="MP",
        help="Path to a py file containing auto metrics method (default: None)",
    )
    parser.add_argument(
        "--kwargs",
        type=parse_kwargs,
        default={},
        metavar="KWARGS",
        help="Additional parameters in JSON format (default: {}). Example: --kwargs '{\"param1\": \"value1\", \"param2\": \"value2\"}'",
    )
    parser.add_argument(
        "--forced_file",
        type=str,
        default=None,
        metavar="FF",
        help="Force the use of a specific file name for saving/loading (default: None)",
    )
    parser.add_argument(
        "--sample_max_len",
        type=int,
        default=None,
        metavar="SML",
        help="Maximum length for each text sample. Texts longer than this will be truncated at the nearest space (default: None - no truncation)",
    )
    args = parser.parse_args()
    return args

def steps_list(list, reference_setter=False):
    if max(list) > 48:
        raise ValueError("List values must be under 48.")
    steps = []
    error = 0
    for i in list:
        if i == 0:
            steps.append("final")
        elif i < 0:
            if not reference_setter:
                raise ValueError("Using a negative value not in reference mode is wrong.")
            error += 1
            steps.append(f"reference{i}")
        else:
            steps.append(f"step{int(i*10000)}")
    if error > 1:
        raise ValueError(f"Negative value are accepted only for reference, you have {error} negative number.")
    return steps

def get_dataset(dataset_name, nb_samples, split="train", text_field="text", sample_max_len=None):
    try:
        dataset_config = None
        try:
            available_splits = get_dataset_split_names(dataset_name)
        except ValueError as e:
            if "Config name is missing" in str(e):
                print(f"Warning: Dataset '{dataset_name}' requires a configuration.")
                available_configs = get_dataset_config_names(dataset_name)
                
                # Try to find a suitable default config
                for config_candidate in ['en', 'default'] + [c for c in available_configs if 'en' in c]:
                    if config_candidate in available_configs:
                        dataset_config = config_candidate
                        break
                
                if not dataset_config and available_configs:
                    dataset_config = available_configs[0]

                if dataset_config:
                    print(f"Attempting to use configuration: '{dataset_config}'")
                    available_splits = get_dataset_split_names(dataset_name, dataset_config)
                else:
                    raise ValueError(f"Could not find any configurations for dataset '{dataset_name}'.") from e
            else:
                raise

        # Try to find the best split if the specified one doesn't exist
        if split not in available_splits:
            print(f"Warning: Split '{split}' not found. Available splits: {available_splits}")
            # Common split names in order of preference
            for fallback_split in ["train", "training", "test", "validation", "val"]:
                if fallback_split in available_splits:
                    split = fallback_split
                    print(f"Using split '{split}' instead.")
                    break
            else:
                if available_splits:
                    split = available_splits[0]
                    print(f"Using first available split '{split}'.")
                else:
                    raise ValueError(f"No splits found for dataset '{dataset_name}'.")

        split_data = load_dataset(dataset_name, name=dataset_config, split=split)
        
        # Try to find the text field if the specified one doesn't exist
        available_fields = list(split_data.column_names)
        if text_field not in available_fields:
            print(f"Warning: Text field '{text_field}' not found. Available fields: {available_fields}")
            # Common text field names
            for fallback_field in ["text", "content", "sentence", "passage", "document", "article", "body"]:
                if fallback_field in available_fields:
                    text_field = fallback_field
                    print(f"Using text field '{text_field}' instead.")
                    break
            else:
                # Search for field that might contain text (string type)
                for field in available_fields:
                    if field in split_data.features and isinstance(split_data[0][field], str):
                        text_field = field
                        print(f"Using detected text field '{text_field}'.")
                        break
                else:
                    raise ValueError(f"No suitable text field found in dataset. Available fields: {available_fields}")
        
        texts = split_data[text_field]
        
        if nb_samples > len(texts):
            print(f"Warning: Requested {nb_samples} samples, but dataset only has {len(texts)} samples. Using all available samples.")
            nb_samples = len(texts)
        
        # Get the subset of texts
        selected_texts = texts[:nb_samples]
        
        if sample_max_len is not None:
            truncated_texts = []
            total_original_length = 0
            total_final_length = 0
            
            for text in selected_texts:
                total_original_length += len(text)
                
                if len(text) <= sample_max_len:
                    truncated_texts.append(text)
                    total_final_length += len(text)
                else:
                    truncate_pos = sample_max_len
                    while truncate_pos > 0 and text[truncate_pos] != ' ':
                        truncate_pos -= 1
                    
                    if truncate_pos == 0:
                        truncate_pos = sample_max_len
                    
                    truncated_text = text[:truncate_pos]
                    truncated_texts.append(truncated_text)
                    total_final_length += len(truncated_text)
            
            print(f"Dataset length statistics:")
            print(f"  - Original total length: {total_original_length:,} characters")
            print(f"  - Final total length: {total_final_length:,} characters") 
            print(f"  - Kept {len(truncated_texts)} elements in dataset")
            print(f"  - Average original length: {total_original_length/len(selected_texts):.1f} characters")
            print(f"  - Average final length: {total_final_length/len(truncated_texts):.1f} characters")
            
            return truncated_texts
        else:
            return selected_texts
        
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}")
        print("Common dataset examples:")
        print("- Wikipedia: 'wikimedia/wikipedia' (config e.g., '20231101.en', text_field='text')")
        print("- Rotten Tomatoes: 'rotten_tomatoes' (use text_field='text')")
        print("- BookCorpus: 'bookcorpus' (use text_field='text')")
        print("- C4: 'c4' (config 'en', use text_field='text', split='train')")
        raise

def retrieve_split_points(split_point_str):
    if "," not in split_point_str:
        return [split_point_str.strip()]
    split_points = split_point_str.split(',')
    return [sp.strip() for sp in split_points]

def retrieve_metrics(metrics_str):
    return [metric.strip() for metric in metrics_str.split(',')]

if __name__ == "__main__":
    args = opts()

    auto_metric = load_auto_metric(args.metric_path)

    print("Device:", args.device)
    if "SAE" in args.decomposer:
        args.device = "cpu"
    print("Model Name:", args.model_name)
    print("Decomposer:", args.decomposer)
    print("Split Point:", args.split_point)
    print("Test Name:", args.test_name)

    dataset = get_dataset(args.dataset, args.nb_dataset_samples*(1+int(args.higher_decomposition)), args.dataset_split, args.text_field, args.sample_max_len)
    split_point = retrieve_split_points(args.split_point)
    metrics = retrieve_metrics(args.metrics)
    reference_step = steps_list([args.reference_steps], reference_setter=True)[0]
    if args.reference_split_point == 0:
        print("A reference split point is required in order to use a split point list, using default value 0 (can be changed using --reference_split_point)")

    if args.decomposer in available_base_decomposer() and not args.higher_decomposition:
        DEC = Base_decomposer(device=args.device, eurobert_repo_id=args.model_name, decomposer=args.decomposer, nb_concepts=args.nb_concepts, decomposition_method=args.decomposition_method, split_point=split_point, test_name=args.test_name, metrics=metrics, reference_step=reference_step, reference_split_point=args.reference_split_point, shared=args.unique_plot, verbose=args.verbose, gradient_plot=args.gradient_plot, log_scale=args.log_scale, kwargs=args.kwargs, forced_file=args.forced_file)
    elif args.decomposer == "LearningProcess":
        DEC = Learning_process_decomposer(device=args.device, eurobert_repo_id=args.model_name, split_point=split_point, test_name=args.test_name, decomposition_method=args.decomposition_method, reference_split_point=args.reference_split_point, shared=args.unique_plot, verbose=args.verbose, convergence=args.convergence, gradient_plot=args.gradient_plot, log_scale=args.log_scale, kwargs=args.kwargs, forced_file=args.forced_file)
    elif args.decomposer in available_base_decomposer() and args.higher_decomposition:
        DEC = Higher_decomposer(device=args.device, eurobert_repo_id=args.model_name, decomposer=args.decomposer, nb_concepts=args.nb_concepts, decomposition_method=args.decomposition_method, split_point=split_point, test_name=args.test_name, train_dataset=dataset[args.nb_dataset_samples:], metrics=metrics, reference_step=reference_step, reference_split_point=args.reference_split_point, shared=args.unique_plot, verbose=args.verbose, gradient_plot=args.gradient_plot, log_scale=args.log_scale, kwargs=args.kwargs, forced_file=args.forced_file)
    else:
        raise ValueError(f"Unsupported decomposer: {args.decomposer}. Supported decomposers are: NMF, SemiNMF, ConvexNMF, PCA, SparsePCA, ICA, SVD, KMeans, DictionaryLearning, LearningProcess.")
    DEC.initialize(dataset[:args.nb_dataset_samples])
    steps = steps_list(args.steps)
    if args.verbose:
        print("Steps to be tested:", steps)
        print("Freeshold for freeing dictionary:", args.freeshold)
    idx = 0
    for step in steps:
        print(f"Running {args.decomposer} for {step}...")
        if step == "final":
            DEC()
        else:
            DEC(steps=step)
        if args.verbose:
            DEC.print_memory()
        if idx % args.freeshold == 0:
            DEC.update_json()
            DEC.free_dict()
        idx += 1
    
    DEC.update_json()
    DEC.plot()
    if args.visualization:
        DEC.visualize()

