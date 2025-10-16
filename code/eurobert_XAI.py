import copy
import io
import requests
from interpreto import ModelWithSplitPoints
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoConfig, PreTrainedTokenizerFast
import interpreto as preto
import interpreto.concepts as pretoconcept
import interpreto.concepts.metrics as pretometric
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import BatchEncoding
from utils import util_metric, solve_with_norm, available_metrics, columnwise_correlation
from sklearn.decomposition import PCA
from cav_utilis import cav_pipeline

decomposer_mapping = {
            "NMF": preto.concepts.NMFConcepts,
            "SemiNMF": preto.concepts.SemiNMFConcepts,
            "ConvexNMF": preto.concepts.ConvexNMFConcepts,
            "PCA": preto.concepts.PCAConcepts,
            "SparsePCA": preto.concepts.SparsePCAConcepts,
            "ICA": preto.concepts.ICAConcepts,
            "SVD": preto.concepts.SVDConcepts,
            "KMeans": preto.concepts.KMeansConcepts,
            "DictionaryLearning": preto.concepts.DictionaryLearningConcepts,
            "VanillaSAE": preto.concepts.VanillaSAEConcepts,
            "TopKSAE": preto.concepts.TopKSAEConcepts,
            "BatchTopKSAE": preto.concepts.BatchTopKSAEConcepts,
            "JumpReLUSAE": preto.concepts.JumpReLUSAEConcepts
        }

def EuroBERT_load_checkpoint_in_memory(repo_id: str, revision: str, checkpoint_file: str, device="cpu"):
    print(f"Loading model from {repo_id} at revision {revision}...")
    config = AutoConfig.from_pretrained(repo_id, revision=revision, trust_remote_code=True)
    weights_url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{checkpoint_file}"
    response = requests.get(weights_url, stream=True)
    response.raise_for_status()
    buffer = io.BytesIO(response.content)
    model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)
    state_dict = torch.load(buffer, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"Model loaded successfully from {repo_id} at revision {revision}.")
    return model

def EuroBERT_load_tokenizer_in_memory(repo_id: str, revision: str, tokenizer_file: str):
    tokenizer_url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{tokenizer_file}"
    response = requests.get(tokenizer_url, stream=True)
    response.raise_for_status()
    buffer = io.BytesIO(response.content)
    temp_file_path = "temp_tokenizer.json"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(buffer.getvalue())
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=temp_file_path)
    os.remove(temp_file_path)
    tokenizer.model_input_names = [name for name in tokenizer.model_input_names if name != "token_type_ids"]
    return tokenizer

class EuroBERT_XAI:
    def __init__(self, model=None, tokenizer=None, device="cpu", eurobert_repo_id="EuroBERT/EuroBERT-210m"):
        self.eurobert_repo_id = eurobert_repo_id
        self.model = model or EuroBERT_load_checkpoint_in_memory(self.eurobert_repo_id, "main", "pytorch_model.bin", device=device)
        self.tokenizer = tokenizer or EuroBERT_load_tokenizer_in_memory(self.eurobert_repo_id, "main", "tokenizer.json")
        self.steps = None
        self.device = device
        self.steps_memory = ["main"]
        self.available_steps = [f"step{int((i+1)*10000)}" for i in range(48)]
        self.available_steps.append("final")

    def delete_past_model(self):
        del self.model, self.tokenizer
        torch.cuda.empty_cache()

    def load_steps(self, steps, no_load=False):
        if no_load:
            if steps is None or steps == "main":
                model = EuroBERT_load_checkpoint_in_memory(self.eurobert_repo_id, "main", "pytorch_model.bin", device=self.device)
                tokenizer = EuroBERT_load_tokenizer_in_memory(self.eurobert_repo_id, "main", "tokenizer.json")
                return model, tokenizer
            else:
                model = EuroBERT_load_checkpoint_in_memory(self.eurobert_repo_id, steps, "pytorch_model.bin", device=self.device)
                tokenizer = EuroBERT_load_tokenizer_in_memory(self.eurobert_repo_id, steps, "tokenizer.json")
                return model, tokenizer
        else:
            if steps == self.steps:
                print(f"Steps {steps} already loaded. No need to reload.")
            elif steps is not None and isinstance(steps, str) and steps != "main":
                self.delete_past_model()
                self.steps = steps
                self.steps_memory.append(steps)
                self.model = EuroBERT_load_checkpoint_in_memory(self.eurobert_repo_id, steps, "pytorch_model.bin", device=self.device)
                self.tokenizer = EuroBERT_load_tokenizer_in_memory(self.eurobert_repo_id, steps, "tokenizer.json")
            else:
                self.delete_past_model()
                self.steps = None
                self.model = EuroBERT_load_checkpoint_in_memory(self.eurobert_repo_id, "main", "pytorch_model.bin", device=self.device)
                self.tokenizer = EuroBERT_load_tokenizer_in_memory(self.eurobert_repo_id, "main", "tokenizer.json")

    def get_steps(self):
        return self.steps

    def get_int_steps(self):
        if self.steps is None or self.steps=="final":
            return 47
        return int(self.steps.split("p")[1])//10000 - 1


class Concept_decomposer(EuroBERT_XAI):
    def __init__(self, split_point=None, test_name="test", dataset=None, model=None, tokenizer=None, nb_concepts=10, device="cpu", eurobert_repo_id="EuroBERT/EuroBERT-210m", reference_split_point=0, decomposer=None, shared=False, forced_file=None):
        super().__init__(model=model, tokenizer=tokenizer, device=device, eurobert_repo_id=eurobert_repo_id)
        self.test_name = forced_file or test_name
        self.splitted_model = None
        self.split_point = split_point # ex "model.layers.1.mlp"
        self.decomposer_dict = {}
        self.decomposer_dict["final"] = {}
        self.nb_concepts = nb_concepts
        self.reference_splitted_model = None
        self.reference_splitted_content = {"tokenizer": None, "split_point": None}
        self.reference_split_point = reference_split_point
        self.dataset = dataset
        self.decomposer_name = decomposer
        self.shared = shared

    def set_split_point(self, split_point):
        self.split_point = split_point

    def delete_splitted_model(self):
        if self.splitted_model is not None:
            del self.splitted_model
            torch.cuda.empty_cache()
            self.splitted_model = None
    
    def delete_reference_splitted_model(self):
        if self.reference_splitted_model is not None:
            del self.reference_splitted_model
            del self.reference_splitted_content["tokenizer"]
            del self.reference_splitted_content["split_point"]
            torch.cuda.empty_cache()
        self.reference_splitted_model = None
        self.reference_splitted_content = {"tokenizer": None, "split_point": None}

    def load_splitted_model(self, steps=None):
        self.load_steps(steps)
        self.delete_splitted_model()
        if self.split_point is not None:
            self.splitted_model = preto.ModelWithSplitPoints(model_or_repo_id=self.model,
                                                            split_points=self.split_point,
                                                            tokenizer=self.tokenizer
                                                            )
        else:
            raise ValueError("Split point must be set before loading the splitted model. Use set_split_point() method.")

    def save_reference_splitted_model(self, steps=None):
        if steps is None:
            steps = self.steps
        model, tokenizer = self.load_steps(steps, no_load=True)
        self.reference_splitted_model = preto.ModelWithSplitPoints(model_or_repo_id=model,
                                                                   split_points=self.split_point,
                                                                   tokenizer=tokenizer
                                                                   )
        self.reference_splitted_content["tokenizer"] = tokenizer
        self.reference_splitted_content["split_point"] = str(self.split_point)

    def make_key(self, key):
        # if key is a string
        if isinstance(key, str):
            if self.steps is not None:
                self.decomposer_dict[self.steps][key] = []
            else:
                self.decomposer_dict["final"][key] = []
        # if key is a list
        elif isinstance(key, list):
            for k in key:
                self.make_key(k)
        else:
            raise ValueError("Key must be a string or a list of strings.")
        
    def tojson_encoder(self, o):
        if isinstance(o, np.ndarray):
            return {
                "__ndarray__": o.tolist(),
                "dtype": str(o.dtype),
                "shape": o.shape
            }
        raise TypeError(f"Object of type {type(o)} is not JSON serializable. Please use numpy arrays or lists.")

    def tojson_decoder(self, d):
        if "__ndarray__" in d:
            arr = np.array(d["__ndarray__"], dtype=d["dtype"])
            arr = arr.reshape(d["shape"])
            return arr
        return d
        
    def save_dict(self, key, value):
        if self.steps is not None:
            if self.steps not in self.decomposer_dict:
                self.decomposer_dict[self.steps] = {}
            if key not in self.decomposer_dict[self.steps]:
                self.make_key(key)
            self.decomposer_dict[self.steps][key] = value
        else:
            if key not in self.decomposer_dict["final"]:
                self.make_key(key)
            self.decomposer_dict["final"][key] = value
    
    def dict_update(self, inter_dict):
        # here key is the step and sub_key is the data name
        for key in self.decomposer_dict:
            if key not in inter_dict:
                inter_dict[key] = {}
            for sub_key in self.decomposer_dict[key]:
                if sub_key not in inter_dict[key]:
                    inter_dict[key][sub_key] = self.decomposer_dict[key][sub_key]
        return inter_dict
    
    def free_dict(self):
        for key in list(self.decomposer_dict.keys()):
            if key != "final":
                del self.decomposer_dict[key]

    def update_json(self):
        file_path = self.test_name + ".json"
        if not os.path.exists(file_path):
            with open(file_path, 'w') as json_file:
                json.dump(self.decomposer_dict, json_file, default=self.tojson_encoder, indent=4)
        else:
            with open(file_path, 'r') as json_file:
                inter_dict = json.load(json_file, object_hook=self.tojson_decoder)
            final_dict = self.dict_update(inter_dict)
            with open(file_path, 'w') as json_file:
                json.dump(final_dict, json_file, default=self.tojson_encoder, indent=4)
    
    def load_json(self):
        file_path = self.test_name + ".json"
        if os.path.exists(file_path):
            with open(file_path, 'r') as json_file:
                ndict = json.load(json_file, object_hook=self.tojson_decoder)
            return ndict
        else:
            raise FileNotFoundError(f"{file_path} does not exist. Please run the decomposer first to create the file.")
    
    def load_dict(self):
        self.update_json()
        ndict = self.load_json()
        self.decomposer_dict = ndict

    def print_json_size(self):
        file_path = self.test_name + ".json"
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"Size of {file_path}: {size / (1024 * 1024):.2f} MB")
        else:
            print(f"{file_path} does not exist.")

    def set_dataset(self, dataset):
        self.dataset = dataset
    
    def tokenize_dataset(self, dataset: list[str]) -> BatchEncoding:
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be set before tokenizing the dataset. Use set_tokenizer() method.")
        if not isinstance(dataset, list):
            raise ValueError("Dataset must be a list of strings.")
        
        token_dataset = self.tokenizer(
            dataset,
            return_tensors="pt",
            padding=True,
        )
        
        allowed_keys = {"input_ids", "attention_mask"}
        for key in list(token_dataset.keys()):
            if key not in allowed_keys:
                token_dataset.pop(key)
        
        token_dataset = BatchEncoding(
            {k: v.to(self.device) for k, v in token_dataset.items()}
        )
        
        return token_dataset

    def plotter(self, sub_key, split_point=None, shared=False, gradient=False, title=None, log_scale=False):
        if split_point is None:
            split_point = self.split_point if isinstance(self.split_point, list) else [self.split_point]
        elif isinstance(split_point, str):
            split_point = [split_point]
        elif not isinstance(split_point, list):
            raise ValueError("Split point must be a string or a list of strings.")
        elif len(split_point) == 0:
            raise ValueError("Split point list cannot be empty.")
        elif not all(isinstance(sp, str) for sp in split_point):
            raise ValueError("All elements in split point list must be strings.")
        elif any(sp not in self.split_point for sp in split_point):
            raise ValueError("One or more elements in split_point are not in the current split_point.")
        
        final_dict = self.load_json()
        if shared:
            plt.figure(figsize=(10, 6))
        
        # Set up gradient colors if requested
        if gradient and len(split_point) > 1:
            cmap = plt.cm.viridis  # You can change this to other colormaps like 'plasma', 'coolwarm', etc.
            colors = [cmap(i / (len(split_point) - 1)) for i in range(len(split_point))]
        else:
            colors = None
            
        for idx, sp in enumerate(split_point):
            sub_key_list = []
            for key in final_dict:
                if key.startswith("step"):
                    number = int(key.replace("step", ""))
                else:
                    number = 490000
                if sub_key in final_dict[key] and sp in final_dict[key][sub_key]:
                    sub_key_list.append([number, final_dict[key][sub_key][sp]])
                # plot sub_key_list
            sub_key_list = sorted(sub_key_list, key=lambda x: x[0])
            x = [i[0] for i in sub_key_list]
            y = [i[1] for i in sub_key_list]
            
            # Use gradient color if enabled
            color = colors[idx] if colors else None
            
            if shared:
                plt.plot(x, y, label=f"{sub_key} ({sp})", color=color)
            else:
                plt.plot(x, y, label=f"{sub_key} ({sp})", color=color)
                plt.xlabel("Steps")
                plt.ylabel(sub_key)
                if log_scale:
                    plt.xscale('log')
                if title is not None:
                    plt.title(title)
                else:
                    plt.title(f"{self.decomposer_name}: {sub_key} over steps ({sp})")
                plt.legend()
                plt.grid()
                plt.savefig(f"{self.test_name}_{sub_key}_{sp}.png")
                plt.clf()
        
        if shared:
            plt.xlabel("Steps")
            plt.ylabel(sub_key)
            if log_scale:
                plt.xscale('log')
            method = self.decomposition_method.split("_")[0]
            if title is not None:
                plt.title(title)
            else:
                plt.title(f"{self.decomposer_name}: {sub_key} over steps ({method})")
            # Create gradient legend if enabled
            if gradient and len(split_point) > 1:
                # Create a colorbar to show the gradient
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(split_point)-1))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.6)
                
                # Set custom labels for the colorbar
                if len(split_point) >= 3:
                    cbar.set_ticks([0, (len(split_point)-1)/2, len(split_point)-1])
                    cbar.set_ticklabels([split_point[0], split_point[len(split_point)//2], split_point[-1]])
                else:
                    cbar.set_ticks([0, len(split_point)-1])
                    cbar.set_ticklabels([split_point[0], split_point[-1]])
                cbar.set_label('Split Points')
            else:
                plt.legend()
            
            plt.grid()
            plt.savefig(f"{self.test_name}_{sub_key}_{method}.png")
            plt.clf()
    
    def visualization(self, sub_key, split_point=None, gradient=False, title=None):
        if split_point is None:
            split_point = self.split_point if isinstance(self.split_point, list) else [self.split_point]
        elif isinstance(split_point, str):
            split_point = [split_point]
        elif not isinstance(split_point, list):
            raise ValueError("Split point must be a string or a list of strings.")
        elif len(split_point) == 0:
            raise ValueError("Split point list cannot be empty.")
        elif not all(isinstance(sp, str) for sp in split_point):
            raise ValueError("All elements in split point list must be strings.")
        elif any(sp not in self.split_point for sp in split_point):
            raise ValueError("One or more elements in split_point are not in the current split_point.")
        
        final_dict = self.load_json()
        
        
        # Create one plot per split point
        for sp in split_point:
            # Collect all data for this split point
            all_data = []
            step_labels = []
            step_numbers = []
            
            for key in final_dict:
                if key.startswith("step"):
                    number = int(key.replace("step", ""))
                else:
                    number = 490000
                    
                if sub_key in final_dict[key] and sp in final_dict[key][sub_key]:
                    data = final_dict[key][sub_key][sp]
                    
                    # Convert to numpy array if needed
                    if not isinstance(data, np.ndarray):
                        data = np.array(data)
                    
                    # Handle dimensions: ensure 2D, take mean over first dimension if 3D
                    if data.ndim == 3:
                        data = np.mean(data, axis=0)
                    elif data.ndim == 1:
                        data = data.reshape(1, -1)
                    elif data.ndim > 3:
                        raise ValueError(f"Data has too many dimensions: {data.ndim}")
                    
                    # Transpose so that points are rows (second dimension becomes first)
                    data = data.T
                    
                    all_data.append(data)
                    step_labels.extend([key] * data.shape[0])
                    step_numbers.extend([number] * data.shape[0])
            
            if not all_data:
                print(f"No data found for split point {sp}")
                continue
                
            # Concatenate all data
            merged_data = np.vstack(all_data)
            
            # Apply PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(merged_data)
            
            # Create the plot
            plt.figure(figsize=(10, 8))
            
            # Get unique steps and assign colors
            unique_steps = sorted(set(step_labels), key=lambda x: step_numbers[step_labels.index(x)])
            
            if gradient and len(unique_steps) > 1:
                cmap = plt.cm.viridis
                colors = [cmap(i / (len(unique_steps) - 1)) for i in range(len(unique_steps))]
                color_map = dict(zip(unique_steps, colors))
            else:
                color_map = {}
            
            # Plot points colored by step
            plotted_labels = set()
            for i, (step_label, step_num) in enumerate(zip(step_labels, step_numbers)):
                color = color_map.get(step_label) if color_map else None
                plt.scatter(pca_result[i, 0], pca_result[i, 1], 
                           c=[color] if color is not None else None,
                           label=step_label if step_label not in plotted_labels else "",
                           alpha=0.7)
                plotted_labels.add(step_label)
            
            plt.xlabel(f'First Principal Component (explained variance: {pca.explained_variance_ratio_[0]:.2%})')
            plt.ylabel(f'Second Principal Component (explained variance: {pca.explained_variance_ratio_[1]:.2%})')
            
            if title is not None:
                plt.title(f"{title} - {sp}")
            else:
                plt.title(f"{self.decomposer_name}: PCA visualization of {sub_key} ({sp})")
            
            # Create legend
            if gradient and len(unique_steps) > 1:
                # Create a colorbar for gradient
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(unique_steps)-1))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.6)
                
                if len(unique_steps) >= 3:
                    cbar.set_ticks([0, (len(unique_steps)-1)/2, len(unique_steps)-1])
                    cbar.set_ticklabels([unique_steps[0], unique_steps[len(unique_steps)//2], unique_steps[-1]])
                else:
                    cbar.set_ticks([0, len(unique_steps)-1])
                    cbar.set_ticklabels([unique_steps[0], unique_steps[-1]])
                cbar.set_label('Training Steps')
            else:
                # Remove duplicate labels and create regular legend
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys())
            
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{self.test_name}_{sub_key}_pca_{sp}.png", dpi=300, bbox_inches='tight')
            plt.clf()

    def print_memory(self):
        if self.device == "cpu":
            print("Memory usage on CPU is not available.")
        else:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
            print(f"CUDA device {self.device} — Free: {free_bytes / 1024**3:.2f} GB, Total: {total_bytes / 1024**3:.2f} GB")

def available_base_decomposer():
    return ["default", "NMF", "SemiNMF", "ConvexNMF", "PCA", "SparsePCA", "ICA", "SVD", "KMeans", "DictionaryLearning", "VanillaSAE", "TopKSAE", "BatchTopKSAE", "JumpReLUSAE"]

class Base_decomposer(Concept_decomposer):
    def __init__(self, decomposer="NMF", decomposition_method="codes_retrieval", force_relu=True, split_point=None, test_name="test", dataset=None, model=None, tokenizer=None, nb_concepts=10, device="cpu", eurobert_repo_id="EuroBERT/EuroBERT-210m", reference_split_point=0, reference_step="final", metrics="default", shared=False, verbose=True, gradient_plot=False, log_scale=False, kwargs=None, forced_file=None):
        super().__init__(split_point=split_point, test_name=test_name, dataset=None, model=model, tokenizer=tokenizer, nb_concepts=nb_concepts, device=device, eurobert_repo_id=eurobert_repo_id, reference_split_point=reference_split_point, decomposer=decomposer, shared=shared, forced_file=forced_file)
        self.available_decomposer = available_base_decomposer()
        if decomposer in self.available_decomposer:
            if decomposer == "default":
                self.decomposer_type = "NMF"
            else:
                self.decomposer_type = decomposer
        else:
            raise ValueError("Decomposer must be one of the following: " + ", ".join(self.available_decomposer))
        self.kwargs = kwargs if kwargs is not None else {}
        if "plot_title" in self.kwargs:
            self.plot_title = self.kwargs["plot_title"]
            self.kwargs.pop("plot_title")
        else:
            self.plot_title = None
        self.gradient_plot = gradient_plot
        self.log_scale = log_scale
        self.force_relu = force_relu
        self.kwargs_scheduler()
        self.decomposer_mapping = decomposer_mapping
        if decomposition_method not in ["default", "codes_retrieval", "concepts_retrieval", "mixed_retrieval"]:
            raise ValueError("Decomposition method must be 'codes_retrieval' or 'concepts_retrieval'.")
        elif decomposition_method == "default":
            self.decomposition_method = "codes_retrieval"
        else:    
            self.decomposition_method = decomposition_method
        self.decomposer = []
        self.reference_decomposer = []
        self.reference_step = reference_step
        self.verbose = verbose
        if "default" in metrics:
            if self.decomposition_method == "codes_retrieval":
                self.metrics = ["MSE", "FID"]
                print("Using default metrics: MSE and FID.")
            else:
                self.metrics = ["COR"]
                print("Using default metrics: COR.")
        else:
            self.metrics = metrics
        if "all" in metrics:
            self.metrics = available_metrics
            if self.decomposition_method != "mixed_retrieval":
                self.metrics.remove("weightCOR")

    def kwargs_scheduler(self):
        if self.kwargs == {}:
            if self.decomposer_type == "NMF":
                self.kwargs["force_relu"] = self.force_relu
            # elif self.decomposer_type == "ICA":
            #     self.kwargs["tol"] = 1e-3
            #     self.kwargs["max_iter"] = 1000

    def initialize(self, dataset, steps=None, save_reference=False):
        if self.decomposition_method == "mixed_retrieval":
            save_reference = True
        if self.steps is not None:
            print("Attention : Initialization fix the decomposition reference to the current steps. This means that you want to compare the other steps to the current steps.")
            print(f"Using steps: {self.steps}")
        if self.split_point is None:
            raise ValueError("Split point must be set before initializing the decomposer. Use set_split_point() method.")
        self.load_splitted_model(steps=steps)
        if save_reference:
            self.delete_reference_splitted_model()
            self.save_reference_splitted_model()
            token_dataset = self.tokenize_dataset(dataset)
            activations = self.reference_splitted_model.get_activations(token_dataset, activation_granularity=ModelWithSplitPoints.activation_granularities.ALL_TOKENS)
            self.fit(activations, save_reference=True)
            del token_dataset
        if self.verbose:
            print(f"Using {self.decomposer_type} decomposer with {self.nb_concepts} concepts.")
        self.set_dataset(dataset)
        if self.decomposition_method == "codes_retrieval":
            if save_reference:
                self.decomposer = self.reference_decomposer
            else:
                token_dataset = self.tokenize_dataset(dataset)
                activations = self.splitted_model.get_activations(token_dataset, activation_granularity=ModelWithSplitPoints.activation_granularities.ALL_TOKENS)
                self.fit(activations)
                del token_dataset
        torch.cuda.empty_cache()
            
        
    def fit(self, activations, save_reference=False):
        if save_reference:
            splitted_model = self.reference_splitted_model
            self.reference_decomposer = []  # Reset reference decomposer list
        else:
            splitted_model = self.splitted_model
            self.decomposer = []  # Reset decomposer list
        for split_point in self.split_point:
            if self.decomposer_type in self.available_decomposer:
                decomposer_class = self.decomposer_mapping[self.decomposer_type]
                decomposer_instance = decomposer_class(splitted_model, nb_concepts=self.nb_concepts, device=self.device, split_point=split_point, **self.kwargs)
            else:
                raise ValueError("Invalid decomposer type. Available options are: " + ", ".join(self.available_decomposer))
            decomposer_instance.fit(activations[split_point])
            if save_reference:
                self.reference_decomposer.append(decomposer_instance)
            else:
                self.decomposer.append(decomposer_instance)

    def retrieve_reference(self):
        if "reference" in self.reference_step:
            ref_value = int(self.reference_step.split('-')[1])
            if self.steps is None or self.steps=="final":
                ref_step = -2
            else:
                ref_step = self.get_int_steps() - ref_value
            reference_step = self.available_steps[ref_step]
        else:
            reference_step = self.reference_step
        return reference_step
        
    def metric(self, activations=None, auto_metric=None):
        reference_step = self.retrieve_reference()
        if (not self.decomposer or self.decomposer == []) and self.decomposition_method == "codes_retrieval":
            raise ValueError("Decomposer must be initialized before calling the method. Use initialize() method (do not forget the dataset as input) or fit() method.")
        if len(self.split_point) <= self.reference_split_point:
            raise ValueError(f"Reference split point {self.reference_split_point} is out of range for the split points {self.split_point}.")
        self.load_dict()
        if auto_metric is None:
            if "MSE" in self.metrics or "FID" in self.metrics:
                if self.decomposition_method == "codes_retrieval":
                    if activations is None:
                        activations = self.get_activation()
                    ref_decomposer = self.decomposer[self.reference_split_point]
            for metric_name in self.metrics:
                if metric_name == "MSE":
                    if self.decomposition_method == "codes_retrieval":
                        metric_MSE = pretometric.MSE(ref_decomposer)
                    score = {}
                    for split_point in self.split_point:
                        if self.decomposition_method == "codes_retrieval":
                            score[split_point] = metric_MSE.compute(activations[split_point].to(self.device))
                        else:
                            score[split_point] = F.mse_loss(torch.tensor(self.decomposer_dict[reference_step]["extracted_concepts"][split_point]), torch.tensor(self.decomposer_dict[self.steps]["extracted_concepts"][split_point] if self.steps is not None else self.decomposer_dict["final"]["extracted_concepts"][split_point])).item()
                elif metric_name == "FID":
                    if self.decomposition_method == "codes_retrieval":
                        metric_FID = pretometric.FID(ref_decomposer)
                    else :
                        raise ValueError("FID metric is only available for codes retrieval method.") 
                    score = {}
                    for split_point in self.split_point:
                        score[split_point] = metric_FID.compute(activations[split_point].to(self.device))
                elif metric_name in available_metrics:
                    if self.decomposer_dict[reference_step]== {}:
                        raise ValueError("COR metric make a correlation between the concepts of the decomposer and the concepts of the reference decomposer. Please retrieve the final decomposition first (steps = None).")
                    if self.nb_concepts < 2:
                        raise ValueError("COR metric requires at least 2 concepts to compute the correlation.")
                    if self.decomposition_method == "codes_retrieval":
                        vect_name = "encoded_activations"
                    elif self.decomposition_method in ["concepts_retrieval", "mixed_retrieval"]:
                        vect_name = "extracted_concepts"
                    else:
                        raise ValueError("Decomposition method must be 'codes_retrieval', 'concepts_retrieval' or 'mixed_retrieval'.")
                    if vect_name not in self.decomposer_dict[reference_step]:
                        raise ValueError(f"Vector {vect_name} is not available in the final decomposition. Please run the decomposition method first.")
                    score = {}
                    for split_point in self.split_point:
                        final_decomp = self.decomposer_dict[reference_step][vect_name][split_point]
                        if self.steps is not None:
                            current_decomp = self.decomposer_dict[self.steps][vect_name][split_point]
                        else:
                            current_decomp = self.decomposer_dict[reference_step][vect_name][split_point]
                        if self.decomposition_method == "mixed_retrieval":
                            current_weight = self.decomposer_dict[self.steps]["encoded_activations"][split_point] if self.steps is not None else self.decomposer_dict[reference_step]["encoded_activations"][split_point]
                        else:
                            current_weight = None
                        metric_value = util_metric(current_decomp, final_decomp, metric_name=metric_name, W=current_weight)
                        score[split_point] = metric_value
                else:
                    raise ValueError(f"Metric {metric_name} is not implemented.")
                self.save_dict(f"score_{metric_name}", score)
        else:
            metric, keys = auto_metric(self.decomposer_dict, reference_step, self.steps, self.split_point, self.decomposition_method)
            for key in keys:
                self.save_dict(f"score_{key}", metric[key])

    def get_activation(self):
        token_dataset = self.tokenize_dataset(self.dataset)
        activations = self.splitted_model.get_activations(token_dataset, activation_granularity=ModelWithSplitPoints.activation_granularities.ALL_TOKENS)
        del token_dataset
        torch.cuda.empty_cache()
        return activations

    def codes_retriever(self, activations=None, reference=False):
        if self.decomposer == [] and self.reference_decomposer == []:
            raise ValueError("Decomposer/ref decomposer must be initialized before calling the method. Use initialize() method (do not forget the dataset as input).")
        if activations is None:
            activations = self.get_activation()
        
        encoded_activations = {}
        if reference:
            decomposer_list = self.reference_decomposer
        else:
            decomposer_list = self.decomposer
        for split_point, decomposer_instance in zip(self.split_point, decomposer_list):
            encoded_activations[split_point] = decomposer_instance.encode_activations(activations[split_point]).cpu().detach().numpy()
        self.save_dict("encoded_activations", encoded_activations)
    
    def concepts_retriever(self, activations=None, reference=False):
        if activations is None:
            activations = self.get_activation()
        self.fit(activations)
        extracted_concepts = {}
        if reference:
            if self.reference_decomposer == []:
                raise ValueError("Reference decomposer must be initialized before calling the method. Use initialize() method (do not forget the dataset as input).")
            decomposer_list = self.reference_decomposer
        else:
            decomposer_list = self.decomposer
        for split_point, decomposer_instance in zip(self.split_point, decomposer_list):
            dictionary = decomposer_instance.get_dictionary()
            if dictionary.dim() == 3:
                extracted_concepts[split_point] = dictionary.permute(1, 2, 0).cpu().detach().numpy()
            elif dictionary.dim() == 2:
                extracted_concepts[split_point] = dictionary.T.cpu().detach().numpy()
            else:
                raise RuntimeError(f"Unexpected tensor dimensions: {dictionary.dim()}")
        self.save_dict("extracted_concepts", extracted_concepts)

    def codes_retrieval_method(self):
        activations = self.get_activation()
        self.codes_retriever(activations)
        self.metric(activations=activations)

    def concepts_retrieval_method(self):
        activations = self.get_activation()
        self.concepts_retriever(activations)
        self.metric(activations=activations)
    
    def mixed_retrieval_method(self):
        activations = self.get_activation()
        self.codes_retriever(activations, reference=True)
        self.concepts_retriever(activations)    
        self.metric(activations=activations)

    def __call__(self, steps=None):
        self.load_splitted_model(steps=steps)
        if self.decomposition_method == "codes_retrieval":
            self.codes_retrieval_method()
        elif self.decomposition_method == "concepts_retrieval":
            self.concepts_retrieval_method()
        elif self.decomposition_method == "mixed_retrieval":
            self.mixed_retrieval_method()
        else:
            raise ValueError("Decomposition method must be 'codes_retrieval', 'concepts_retrieval' or 'mixed_retrieval'.")
        if self.verbose:
            print(f"Decomposition method {self.decomposition_method} done.")
    
    def plot(self, split_point=None):
        if self.decomposition_method in ["codes_retrieval", "concepts_retrieval", "mixed_retrieval"]:
            for metric_name in self.metrics:
                self.plotter(f"score_{metric_name}", split_point=split_point, shared=self.shared, gradient=self.gradient_plot, title=self.plot_title, log_scale=self.log_scale)
        else:
            raise ValueError("Decomposition method must be 'codes_retrieval', 'concepts_retrieval' or 'mixed_retrieval'.")
        print("Plots saved.")

    def visualize(self, split_point=None):
        if self.decomposition_method in ["codes_retrieval", "concepts_retrieval", "mixed_retrieval"]:
            if self.decomposition_method == "codes_retrieval":
                vect_name = "encoded_activations"
            elif self.decomposition_method in ["concepts_retrieval", "mixed_retrieval"]:
                vect_name = "extracted_concepts"
            else:
                raise ValueError("Decomposition method must be 'codes_retrieval', 'concepts_retrieval' or 'mixed_retrieval'.")
            self.visualization(vect_name, split_point=split_point, gradient=self.gradient_plot, title=self.plot_title)
        else:
            raise ValueError("Decomposition method must be 'codes_retrieval', 'concepts_retrieval' or 'mixed_retrieval'.")
        print("Visualizations saved.")

class Learning_process_decomposer(Concept_decomposer):
    def __init__(self, decomposition_method="Weight", split_point=None, test_name="test", dataset=None, model=None, tokenizer=None, nb_concepts=10, device="cpu", eurobert_repo_id="EuroBERT/EuroBERT-210m", reference_split_point=0, shared=False, verbose=True, convergence=True, gradient_plot=False, log_scale=False, kwargs=None, forced_file=None):
        self.decomposer = None
        self.kwargs = kwargs if kwargs is not None else {}
        if "plot_title" in self.kwargs:
            self.plot_title = self.kwargs["plot_title"]
            self.kwargs.pop("plot_title")
        else:
            self.plot_title = None
        self.gradient_plot = gradient_plot
        self.log_scale = log_scale
        if decomposition_method not in ["default", "Weight", "Activation"]:
            raise ValueError("Decomposition method must be 'Weight' or 'Activation'.")
        elif decomposition_method == "default":
            self.decomposition_method = "Weight"
        else:
            self.decomposition_method = decomposition_method
        if self.decomposition_method == "Weight":
            if split_point != "model.layers.1.mlp":
                print("For weight decay decomposition, split point is useless, but still require a proper one to go through the full pipeline.")
                split_point = "model.layers.1.mlp"
        super().__init__(split_point=split_point, test_name=test_name, dataset=dataset, model=model, tokenizer=tokenizer, nb_concepts=nb_concepts, device=device, eurobert_repo_id=eurobert_repo_id, reference_split_point=reference_split_point, decomposer="LearningProcess", shared=shared, forced_file=forced_file)
        self.verbose = verbose
        self.initial_point = None
        self.convergence = convergence
        
    def initialize(self, dataset, steps=None, save_reference=False):
        if self.decomposition_method == "Weight":
            if self.steps is not None:
                print("Attention : Initialization fix the decomposition reference to the current steps. This means that you want to compare the other steps to the current steps.")
                print(f"Using steps: {self.steps}")
            self.load_splitted_model(steps=steps)
            if save_reference:
                self.delete_reference_splitted_model()
                self.save_reference_splitted_model()
            self.set_dataset(dataset)
            if self.convergence:
                self.initial_point = {}
                for name, param in self.model.named_parameters():
                    if "weight" in name and param.requires_grad:
                        self.initial_point[name] = param.detach().clone()
            else:
                self.initial_point = None

        elif self.decomposition_method == "Activation":
            if self.steps is not None:
                print("Attention : Initialization fix the decomposition reference to the current steps. This means that you want to compare the other steps to the current steps.")
                print(f"Using steps: {self.steps}")
            if self.split_point is None:
                raise ValueError("Split point must be set before initializing the decomposer. Use set_split_point() method.")
            self.load_splitted_model(steps=steps)
            if save_reference:
                self.delete_reference_splitted_model()
                self.save_reference_splitted_model()
            self.set_dataset(dataset)
            token_dataset = self.tokenize_dataset(dataset)
            activations = self.splitted_model.get_activations(token_dataset, activation_granularity=ModelWithSplitPoints.activation_granularities.ALL_TOKENS)
            del token_dataset
            torch.cuda.empty_cache()
            if self.convergence:
                self.initial_point = activations
            else:
                self.initial_point = {}
                for split_point in self.split_point:
                    self.initial_point[split_point] = torch.zeros_like(activations[split_point])
        else:
            raise ValueError("Decomposition method must be 'Weight' or 'Activation'.")

    def weight_decay(self):
        if self.model is None:
            raise ValueError("Model must be loaded before computing weight decay.")
        weight_norms = []
        num_layers = 0
        if self.convergence:
            if self.initial_point is None:
                raise ValueError("Initial weights must be set before computing weight decay. Use initialize() method.")
            for name, param in self.model.named_parameters():
                if "weight" in name and param.requires_grad and name in self.initial_point:
                    diff = param - self.initial_point[name]
                    squared_norm = torch.norm(diff, p=2).item() ** 2
                    dimension = param.numel()
                    weight_norms.append(squared_norm / dimension)
                    num_layers += 1
        else:
            for name, param in self.model.named_parameters():
                if "weight" in name and param.requires_grad:
                    squared_norm = torch.norm(param, p=2).item() ** 2
                    dimension = param.numel()
                    weight_norms.append(squared_norm / dimension)
                    num_layers += 1
        if num_layers == 0:
            raise ValueError("No weight matrices found in the model.")
        weight_decay_score = sum(weight_norms) / num_layers
        weight_decay = {}
        weight_decay["model.layers.1.mlp"] = weight_decay_score
        self.save_dict("weight_decay_score", weight_decay)
        

    def activation_decay(self):
        if self.initial_point is None:
            raise ValueError("Initial activations must be set before computing activation decay. Use initialize() method.")
        if self.splitted_model is None:
            raise ValueError("Splitted model must be loaded before computing activation decay.")
        
        activation_norms = {}
        token_dataset = self.tokenize_dataset(self.dataset)
        activations = self.splitted_model.get_activations(token_dataset, activation_granularity=ModelWithSplitPoints.activation_granularities.ALL_TOKENS)
        for split_point in self.split_point:
            activation = activations[split_point]
            initial_activation = self.initial_point[split_point]
            squared_norms = torch.norm(activation - initial_activation, p=2, dim=-1) ** 2
            dimension = activation.numel() / activation.shape[-1]
            activation_norms[split_point] = squared_norms.mean().item() / dimension
        
        self.save_dict("activation_decay_score", activation_norms)

    def __call__(self, steps=None):
        self.load_splitted_model(steps=steps)
        if self.decomposition_method == "Weight":
            self.weight_decay()
        elif self.decomposition_method == "Activation":
            self.activation_decay()
        else:
            raise ValueError("Decomposition method must be 'Weight' or 'Activation'.")
        if self.verbose:
            print(f"Decomposition method {self.decomposition_method} done.")

    def plot(self, split_point=None):
        if self.decomposition_method == "Weight":
            self.plotter("weight_decay_score", split_point=split_point, shared=self.shared, gradient=self.gradient_plot, title=self.plot_title, log_scale=self.log_scale)
        elif self.decomposition_method == "Activation":
            self.plotter("activation_decay_score", split_point=split_point, shared=self.shared, gradient=self.gradient_plot, title=self.plot_title, log_scale=self.log_scale)
        else:
            raise ValueError("Decomposition method must be 'Weight' or 'Activation'.")
        print("Plots saved.")

    def visualize(self, split_point=None):
        pass

class Higher_decomposer(Concept_decomposer):
    def __init__(self, decomposer="NMF", decomposition_method="codes_retrieval", force_relu=True, split_point=None, test_name="test", train_dataset=None, test_dataset=None, model=None, tokenizer=None, nb_concepts=10, device="cpu", eurobert_repo_id="EuroBERT/EuroBERT-210m", reference_split_point=0, reference_step="final", metrics="default", shared=False, verbose=True, gradient_plot=False, log_scale=False, kwargs=None, forced_file=None):
        super().__init__(split_point=split_point, test_name=test_name, dataset=train_dataset, model=model, tokenizer=tokenizer, nb_concepts=nb_concepts, device=device, eurobert_repo_id=eurobert_repo_id, reference_split_point=reference_split_point, decomposer=decomposer, shared=shared, forced_file=forced_file)
        self.available_decomposer = available_base_decomposer()
        if decomposer in self.available_decomposer:
            if decomposer == "default":
                self.decomposer_type = "NMF"
            else:
                self.decomposer_type = decomposer
        else:
            raise ValueError("Decomposer must be one of the following: " + ", ".join(self.available_decomposer))
        self.kwargs = kwargs if kwargs is not None else {}
        if "plot_title" in self.kwargs:
            self.plot_title = self.kwargs["plot_title"]
            self.kwargs.pop("plot_title")
        else:
            self.plot_title = None
        self.gradient_plot = gradient_plot
        self.log_scale = log_scale
        if "solver" in self.kwargs:
            self.solver_type = self.kwargs["solver"]
            self.kwargs.pop("solver")
        else:
            self.solver_type = "fro"
        if self.solver_type not in ["fro", "1", "2", "linf", "nuc"]:
            raise ValueError("Solver must be one of the following: 'fro', '1', '2', 'linf', 'nuc'.")
        self.force_relu = force_relu
        self.kwargs_scheduler()
        self.decomposer_mapping = decomposer_mapping
        if decomposition_method not in ["default", "codes_retrieval", "concepts_retrieval", "mixed_retrieval", "cav_retrieval"]:
            raise ValueError("Decomposition method must be 'codes_retrieval', 'concepts_retrieval', 'mixed_retrieval' or 'cav_retrieval'.")
        elif decomposition_method == "default":
            self.decomposition_method = "codes_retrieval"
        else:    
            self.decomposition_method = decomposition_method
        self.decomposer = []
        self.reference_decomposer = []
        self.reference_step = reference_step
        self.verbose = verbose
        if "default" in metrics:
            self.metrics = ["MSE"]
            print("Using default metrics: MSE.")
        else:
            self.metrics = metrics
        if train_dataset is None:
            raise ValueError("Train dataset must be provided.")
        self.set_dataset(train_dataset)
        self.test_dataset = None

    def kwargs_scheduler(self):
        # if self.kwargs == {}:
        #     if self.decomposer_type == "ICA":
        #         self.kwargs["tol"] = 1e-3
        #         self.kwargs["max_iter"] = 1000
        if self.decomposer_type == "NMF":
            self.kwargs["force_relu"] = self.force_relu
            

    def initialize(self, test_dataset):
        if self.test_dataset is None and test_dataset is None:
            raise ValueError("Test dataset must be provided for the first call to initialize the true encoded activations.")
        if self.test_dataset is None:
            self.test_dataset = test_dataset
        ref_step = self.retrieve_reference()
        true_step = self.steps
        self.steps = ref_step
        print(f"Init - Using steps: {ref_step if ref_step is not None else 'final'}")
        if self.split_point is None:
            raise ValueError("Split point must be set before initializing the decomposer. Use set_split_point() method.")
        self.load_dict()
        if self.check_true_presence(ref_step):
            if self.verbose:
                print(f"True encoded activations already present for step {ref_step}. Skipping reference decomposer fitting.")
            return
        self.delete_reference_splitted_model()
        self.save_reference_splitted_model(steps=ref_step)
        if self.verbose:
            print(f"Using {self.decomposer_type} decomposer with {self.nb_concepts} concepts.")
        if self.decomposition_method == "codes_retrieval":
            token_dataset = self.tokenize_dataset(self.dataset)
            activations = self.reference_splitted_model.get_activations(token_dataset, activation_granularity=ModelWithSplitPoints.activation_granularities.ALL_TOKENS)
            self.fit(activations, save_reference=True)
            self.codes_retriever(activations=activations, reference=True, init=True)
            token_dataset = self.tokenize_dataset(test_dataset)
            activations = self.reference_splitted_model.get_activations(token_dataset, activation_granularity=ModelWithSplitPoints.activation_granularities.ALL_TOKENS)
            self.fit(activations, splitted=self.reference_splitted_model)
            self.codes_retriever(activations=activations, init=True)
            del token_dataset
            del activations
        elif self.decomposition_method == "concepts_retrieval":
            token_dataset = self.tokenize_dataset(self.dataset)
            activations = self.reference_splitted_model.get_activations(token_dataset, activation_granularity=ModelWithSplitPoints.activation_granularities.ALL_TOKENS)
            self.concepts_retriever(activations=activations, reference=True, add="train")
            token_dataset = self.tokenize_dataset(test_dataset)
            activations = self.reference_splitted_model.get_activations(token_dataset, activation_granularity=ModelWithSplitPoints.activation_granularities.ALL_TOKENS)
            self.concepts_retriever(activations=activations, reference=True, add="test")
            del token_dataset
            del activations
        self.steps = true_step
        torch.cuda.empty_cache()
            
        
    def fit(self, activations, save_reference=False, splitted=None):
        if save_reference:
            splitted_model = self.reference_splitted_model
            self.reference_decomposer = []  # Reset reference decomposer list
        else:
            splitted_model = self.splitted_model
            self.decomposer = []  # Reset decomposer list
        if splitted is not None:
            splitted_model = splitted
        for split_point in self.split_point:
            if self.decomposer_type in self.available_decomposer:
                decomposer_class = self.decomposer_mapping[self.decomposer_type]
                decomposer_instance = decomposer_class(splitted_model, nb_concepts=self.nb_concepts, device=self.device, split_point=split_point, **self.kwargs)
            else:
                raise ValueError("Invalid decomposer type. Available options are: " + ", ".join(self.available_decomposer))
            decomposer_instance.fit(activations[split_point])
            if save_reference:
                self.reference_decomposer.append(decomposer_instance)
            else:
                self.decomposer.append(decomposer_instance)

    def retrieve_reference(self):
        if "reference" in self.reference_step:
            ref_value = int(self.reference_step.split('-')[1])
            if self.steps is None or self.steps=="final":
                ref_step = -2
            else:
                ref_step = self.get_int_steps() - ref_value
            reference_step = self.available_steps[ref_step]
        else:
            reference_step = self.reference_step
        return reference_step
        
    def metric(self, auto_metric=None):
        reference_step = self.retrieve_reference()
        if len(self.split_point) <= self.reference_split_point:
            raise ValueError(f"Reference split point {self.reference_split_point} is out of range for the split points {self.split_point}.")
        self.load_dict()
        if auto_metric is None:
            if self.decomposition_method == "codes_retrieval":
                V_F_te = self.decomposer_dict[reference_step]["true_test_encoded_activations"]
                if self.steps is not None:
                    V_te = self.decomposer_dict[self.steps]["false_test_encoded_activations"]
                else:
                    V_te = self.decomposer_dict["final"]["false_test_encoded_activations"]
            elif self.decomposition_method == "concepts_retrieval":
                V_F_te = self.decomposer_dict[reference_step]["test_extracted_concepts"]
                if self.steps is not None:
                    _V_te = self.decomposer_dict[self.steps]["test_extracted_concepts"]
                else:
                    _V_te = self.decomposer_dict["final"]["test_extracted_concepts"]
                Transform = self.decomposer_dict[self.steps]["trained_transform"]
                V_te = self.matrix_transform(_V_te, Transform)
            for metric_name in self.metrics:
                score = {}
                for split_point in self.split_point:
                    v_te = torch.tensor(V_te[split_point])
                    v_f_te = torch.tensor(V_F_te[split_point])
                    if len(v_te.shape) == 2:
                        v_te = v_te.unsqueeze(0)
                    if len(v_f_te.shape) == 2:
                        v_f_te = v_f_te.unsqueeze(0)
                    if metric_name == "MSE":
                            score[split_point] = F.mse_loss(v_te, v_f_te).item()
                    elif metric_name == "CE":
                            score[split_point] = F.cross_entropy(v_te, v_f_te).item()
                    elif metric_name == "KL":
                            v_te_prob = F.log_softmax(v_te.flatten(), dim=0)
                            v_f_te_prob = F.softmax(v_f_te.flatten(), dim=0)
                            score[split_point] = F.kl_div(v_te_prob, v_f_te_prob, reduction="batchmean").item()
                    elif metric_name == "COR":
                            score[split_point] = columnwise_correlation(np.array(V_te[split_point]), np.array(V_F_te[split_point]))
                    else:
                        raise ValueError(f"Metric {metric_name} is not implemented.")
            self.save_dict(f"score_{metric_name}", score)
        else:
            metric_dict, keys = auto_metric(self.decomposer_dict, reference_step, self.steps, self.split_point, self.decomposition_method)
            for key in keys:
                self.save_dict(f"score_{key}", metric_dict[key])

    def check_true_presence(self, steps):
        # check that a given steps has true_train_encoded_activation inside dict
        if steps in self.decomposer_dict and "true_train_encoded_activations" in self.decomposer_dict[steps] and "true_test_encoded_activations" in self.decomposer_dict[steps]:
            return True
        return False

    def get_activation(self, test=False):
        if test and self.test_dataset is None:
            raise ValueError("Test dataset must be initialized before calling the method.")
        if test:
            token_dataset = self.tokenize_dataset(self.test_dataset)
        else:
            token_dataset = self.tokenize_dataset(self.dataset)
        activations = self.splitted_model.get_activations(token_dataset, activation_granularity=ModelWithSplitPoints.activation_granularities.ALL_TOKENS)
        del token_dataset
        torch.cuda.empty_cache()
        return activations
    
    def solver(self, U_F_tr, activations):
        E_tr = {}
        for split_point in self.split_point:
            if split_point not in U_F_tr:
                raise ValueError(f"Split point {split_point} not found in true_train_encoded_activations.")
            if split_point not in activations:
                raise ValueError(f"Split point {split_point} not found in activations.")
            if len(activations[split_point].shape) == 2:
                activations[split_point] = np.expand_dims(activations[split_point], axis=0)
            if len(U_F_tr[split_point].shape) == 2:
                U_F_tr[split_point] = np.expand_dims(U_F_tr[split_point], axis=0)
            if len(U_F_tr[split_point].shape) != len(activations[split_point].shape) or U_F_tr[split_point].shape[1] != activations[split_point].shape[1]:
                raise ValueError(f"Shape mismatch between true_train_encoded_activations and activations for split point {split_point}. Activations shape: {activations[split_point].shape}, true_train_encoded_activations shape: {U_F_tr[split_point].shape}.")
            E_tr[split_point] = np.zeros((activations[split_point].shape[0], activations[split_point].shape[2], U_F_tr[split_point].shape[2]))
            for i in range(activations[split_point].shape[0]):
                E_tr[split_point][i] = solve_with_norm(U_F_tr[split_point][i], activations[split_point][i], norm=self.solver_type)
        return E_tr

    def encoder_transform(self, activations, E_tr):
        U_tr = {}
        squeeze = False
        for split_point in self.split_point:
            if split_point not in E_tr:
                raise ValueError(f"Split point {split_point} not found in encoder.")
            if split_point not in activations:
                raise ValueError(f"Split point {split_point} not found in activations.")
            if len(activations[split_point].shape) == 2:
                squeeze = True
                activations[split_point] = np.expand_dims(activations[split_point], axis=0)
            if len(E_tr[split_point].shape) != len(activations[split_point].shape):
                raise ValueError(f"Shape mismatch between encoder and activations for split point {split_point}. Activations shape: {activations[split_point].shape}, encoder shape: {E_tr[split_point].shape}.")
            if E_tr[split_point].shape[0] != activations[split_point].shape[0] and E_tr[split_point].shape[1] != activations[split_point].shape[1]:
                raise ValueError(f"Number of samples mismatch between encoder and activations for split point {split_point}. Expected ({activations[split_point].shape[0]};{activations[split_point].shape[1]}) samples, got ({E_tr[split_point].shape[0]};{E_tr[split_point].shape[1]}) samples.")
            U_tr[split_point] = np.zeros((activations[split_point].shape[0], activations[split_point].shape[1], E_tr[split_point].shape[2]))
            for i in range(activations[split_point].shape[0]):
                U_tr[split_point][i] = np.matmul(activations[split_point][i], E_tr[split_point][i])
            if squeeze:
                U_tr[split_point] = np.squeeze(U_tr[split_point], axis=0)
                squeeze = False
        return U_tr

    def encoder_solver(self, activations, reference=False):
        ref_steps = self.retrieve_reference()
        U_F_tr = self.decomposer_dict[ref_steps]["true_train_encoded_activations"]
        if reference:
            train_or_test = "false_train"
            E_tr = self.solver(U_F_tr, activations)
            U_tr = self.encoder_transform(activations, E_tr)
            self.save_dict("train_encoder", E_tr)
            return U_tr
        else:
            train_or_test = "false_test"
            if "train_encoder" not in self.decomposer_dict[ref_steps]:
                raise RuntimeError("Encoder not initialized: 'train_encoder' missing. Please run with reference=True first to initialize.")
            E_tr = self.decomposer_dict[ref_steps]["train_encoder"]
            U_te = self.encoder_transform(activations, E_tr)
            return U_te

    def codes_retriever(self, activations=None, reference=False, init=False):
        if self.decomposer == [] and self.reference_decomposer == [] and not self.check_true_presence(self.retrieve_reference()):
            raise ValueError("Decomposer/ref decomposer must be initialized before calling the method. Use initialize() method (do not forget the dataset as input).")
        if activations is None:
            activations = self.get_activation(test=not reference)
        
        encoded_activations = {}
        if reference:
            train_or_test = "_train"
            decomposer_list = self.reference_decomposer
        else:
            train_or_test = "_test"
            decomposer_list = self.decomposer
        if init:
            train_or_test = "true" + train_or_test
            for split_point, decomposer_instance in zip(self.split_point, decomposer_list):
                encoded_activations[split_point] = decomposer_instance.encode_activations(activations[split_point]).cpu().detach().numpy()
        else:
            train_or_test = "false" + train_or_test
            encoded_activations = self.encoder_solver(activations, reference=reference)
        self.save_dict(train_or_test + "_encoded_activations", encoded_activations)

    def codes_retrieval_method(self):
        self.codes_retriever(reference=True)
        self.codes_retriever(reference=False)
        self.metric()


    def concepts_retriever(self, activations=None, reference=False, add=None):
        if activations is None and not reference:
            activations = self.get_activation()
        if activations is None and reference:
            raise ValueError("Activations must be provided for concepts retrieval when reference is True.")
        self.fit(activations, save_reference=reference)
        extracted_concepts = {}
        true_step = self.steps
        if reference:
            if self.reference_decomposer == []:
                raise ValueError("Reference decomposer must be initialized before calling the method. Use initialize() method (do not forget the dataset as input).")
            ref_step = self.retrieve_reference()
            self.steps = ref_step
            decomposer_list = self.reference_decomposer
        else:
            if self.decomposer == []:
                raise ValueError("Decomposer must be initialized before calling the method. Use initialize() method (do not forget the dataset as input).")
            decomposer_list = self.decomposer
        for split_point, decomposer_instance in zip(self.split_point, decomposer_list):
            dictionary = decomposer_instance.get_dictionary()
            if dictionary.dim() == 3:
                print("Dictionary dim is 3 :" + str(dictionary.shape))
                extracted_concepts[split_point] = dictionary.permute(1, 2, 0).cpu().detach().numpy()
            elif dictionary.dim() == 2:
                extracted_concepts[split_point] = dictionary.T.cpu().detach().numpy()
            else:
                raise RuntimeError(f"Unexpected tensor dimensions: {dictionary.dim()}")
        if add is not None:
            if add not in ["train", "test"]:
                raise ValueError("Add must be 'train' or 'test'.")
            name_concept = add + "_extracted_concepts"
        else:
            name_concept = "train_extracted_concepts"
        self.save_dict(name_concept, extracted_concepts)
        self.steps = true_step
    
    def retrieve_transform(self, C1, C2):
        T = {}
        for split_point in self.split_point:
            A = C1[split_point]
            B = C2[split_point]
            if len(A.shape) == 2:
                A = np.expand_dims(A, axis=0)
            if len(B.shape) == 2:
                B = np.expand_dims(B, axis=0)
            if len(A.shape) != len(B.shape):
                raise ValueError(f"Shape mismatch between A and B. A shape: {A.shape}, B shape: {B.shape}.")
            if A.shape[0] != B.shape[0] or A.shape[1] != B.shape[1] or A.shape[2] != B.shape[2]:
                raise ValueError(f"Shape mismatch between A and B. A shape: {A.shape}, B shape: {B.shape}.")
            T[split_point] = np.zeros((A.shape[0], A.shape[2], B.shape[2]))
            for i in range(A.shape[0]):
                T[split_point][i] = solve_with_norm(A[i], B[i], norm=self.solver_type)
        return T
    
    def matrix_transform(self, C, Transform):
        T_C = {}
        for split_point in self.split_point:
            if len(C[split_point].shape) == 2:
                C[split_point] = np.expand_dims(C[split_point], axis=0)
            if len(Transform[split_point].shape) == 2:
                Transform[split_point] = np.expand_dims(Transform[split_point], axis=0)
            T_C[split_point] = np.zeros((C[split_point].shape[0], C[split_point].shape[1], Transform[split_point].shape[2]))
            for i in range(C[split_point].shape[0]):
                T_C[split_point][i] = np.matmul(C[split_point][i], Transform[split_point][i])
        return T_C
            
    def concepts_transform(self):
        ref_steps = self.retrieve_reference()
        if "train_extracted_concepts" not in self.decomposer_dict[ref_steps]:
            raise RuntimeError("Train concepts not initialized: 'train_extracted_concepts' missing. Please run with reference=True first to initialize.")
        if "train_extracted_concepts" not in self.decomposer_dict[self.steps if self.steps is not None else "final"]:
            raise RuntimeError("Train concepts not initialized: 'train_extracted_concepts' missing. Please run with reference=False first to initialize.")
        C1 = self.decomposer_dict[ref_steps]["train_extracted_concepts"]
        C2 = self.decomposer_dict[self.steps if self.steps is not None else "final"]["train_extracted_concepts"]
        Transform = self.retrieve_transform(C1, C2)
        self.save_dict("trained_transform", Transform)

    def concepts_retrieval_method(self):
        activations = self.get_activation()
        self.concepts_retriever(add="train")
        self.concepts_transform()
        activations = self.get_activation(test=True)
        self.concepts_retriever(add="test")
        self.metric()
    
    def cav_retrieval_method(self):
        print(f"CAV - Using steps: {self.steps if self.steps is not None else 'final'}")
        if self.split_point is None:
            raise ValueError("Split point must be set before initializing the decomposer. Use set_split_point() method.")
        CAV = cav_pipeline(self.kwargs.get("concept_type", "semantic"), self.kwargs.get("num_samples", 500), self.splitted_model, self.tokenizer, self.split_point,
                   batch_size=self.kwargs.get("batch_size", 32),
                   l2_reg=self.kwargs.get("l2_reg", 0.1),
                   max_iter=self.kwargs.get("max_iter", 500),
                   tol=self.kwargs.get("tol", 1e-4),
                   test_size=self.kwargs.get("test_size", 0.9),
                   seed=self.kwargs.get("seed", 42),
                   verbose=self.verbose)
        self.save_dict("CAV", CAV["cavs"])
        self.save_dict("CAV_accuracy", CAV["tcav_scores"])

    def mixed_retrieval_method(self):
        pass

    def __call__(self, steps=None):
        self.load_splitted_model(steps=steps)
        ref_step = self.retrieve_reference()
        if not self.check_true_presence(ref_step):
            if self.test_dataset is None:
                raise ValueError("Test dataset must be provided for the first call to initialize the true encoded activations.")
            self.initialize(self.test_dataset)
        if self.decomposition_method == "codes_retrieval":
            self.codes_retrieval_method()
        elif self.decomposition_method == "concepts_retrieval":
            self.concepts_retrieval_method()
        elif self.decomposition_method == "mixed_retrieval":
            self.mixed_retrieval_method()
        elif self.decomposition_method == "cav_retrieval":
            self.cav_retrieval_method()
        else:
            raise ValueError("Decomposition method must be 'codes_retrieval', 'concepts_retrieval' or 'mixed_retrieval'.")
        if self.verbose:
            print(f"Decomposition method {self.decomposition_method} done.")
    
    def plot(self, split_point=None):
        if self.decomposition_method in ["codes_retrieval", "concepts_retrieval", "mixed_retrieval"]:
            for metric_name in self.metrics:
                self.plotter(f"score_{metric_name}", split_point=split_point, shared=self.shared, gradient=self.gradient_plot, title=self.plot_title, log_scale=self.log_scale)
        elif self.decomposition_method == "cav_retrieval":
            self.plotter("CAV_accuracy", split_point=split_point, shared=self.shared, gradient=self.gradient_plot, title=self.plot_title, log_scale=self.log_scale)  
        else:
            raise ValueError("Decomposition method must be 'codes_retrieval', 'concepts_retrieval' or 'mixed_retrieval'.")
        print("Plots saved.")

    def visualize(self, split_point=None):
        pass

    def create_transform_json(self):
        transform_dict = {}
        for step in self.available_steps:
            if step != "final":
                if "trained_transform" not in self.decomposer_dict[step]:
                    raise RuntimeError(f"Transform not found for step {step}. Please run concepts retrieval method first.")
                for split_point in self.split_point:
                    if split_point not in transform_dict:
                        transform_dict[split_point] = []
                    # Convert numpy array to list of lists before appending
                    arr = self.decomposer_dict[step]["trained_transform"][split_point]
                    arr_list = arr.tolist() if isinstance(arr, np.ndarray) else arr
                    transform_dict[split_point].append(arr_list)
        with open(f"{self.test_name}_transform.json", "w") as f:
            json.dump(transform_dict, f)
        print(f"Transform json file saved as {self.test_name}_transform.json")