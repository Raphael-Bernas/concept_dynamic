#############################################################################################################################################
# How to use :
# 1. Make sure you have installed Interpreto from pip install
#    pip install interpreto
# 2. Make sure you have installed the required packages from Interpreto_requirements.txt (it should be automatic with pip install interpreto)
# 3. Make sure you have the correct path to the code in the script
# 4. Use bash main_decomposition.sh to run the script
#############################################################################################################################################

# You can modify the parameters below to suit your needs :

# Python environment configuration
USE_UV=false  # Set to true to use 'uv run python', false to use regular 'python'

# Set Python command based on USE_UV parameter
if [ "$USE_UV" = true ]; then
    PYTHON_CMD="uv run python"
else
    PYTHON_CMD="python"
fi

# Choose which script to run: "main" or "process"
SCRIPT_CHOICE="main"  # Set to "main" or "process"
if [ "$SCRIPT_CHOICE" = "main" ]; then
    SCRIPT_PATH="concept_dynamic/code/main.py"
else
    SCRIPT_PATH="concept_dynamic/code/process.py"
fi

model_size="210m" # "210m", "610m", "2.1B"
model_name="EuroBERT/EuroBERT-${model_size}"
timestamp=$(date +"%Y%m%d-%H%M%S")
test_name="test_${model_size}${timestamp}"
decomposer="SemiNMF"
decomposition_method="concepts_retrieval"
split_point="model.layers.1.mlp, model.layers.2.mlp, model.layers.3.mlp, model.layers.4.mlp, model.layers.5.mlp, model.layers.6.mlp, model.layers.7.mlp, model.layers.8.mlp, model.layers.9.mlp, model.layers.10.mlp, model.layers.11.mlp" # 210m
# split_point="model.layers.1.mlp, model.layers.2.mlp, model.layers.3.mlp, model.layers.4.mlp, model.layers.5.mlp, model.layers.6.mlp, model.layers.7.mlp, model.layers.8.mlp, model.layers.9.mlp, model.layers.10.mlp, model.layers.11.mlp, model.layers.12.mlp, model.layers.13.mlp, model.layers.14.mlp, model.layers.15.mlp, model.layers.16.mlp, model.layers.17.mlp, model.layers.18.mlp, model.layers.19.mlp, model.layers.20.mlp, model.layers.21.mlp, model.layers.22.mlp, model.layers.23.mlp, model.layers.24.mlp, model.layers.25.mlp" # 610m
# split_point="model.layers.1.mlp, model.layers.2.mlp, model.layers.3.mlp, model.layers.4.mlp, model.layers.5.mlp, model.layers.6.mlp, model.layers.7.mlp, model.layers.8.mlp, model.layers.9.mlp, model.layers.10.mlp, model.layers.11.mlp, model.layers.12.mlp, model.layers.13.mlp, model.layers.14.mlp, model.layers.15.mlp, model.layers.16.mlp, model.layers.17.mlp, model.layers.18.mlp, model.layers.19.mlp, model.layers.20.mlp, model.layers.21.mlp, model.layers.22.mlp, model.layers.23.mlp, model.layers.24.mlp, model.layers.25.mlp, model.layers.26.mlp, model.layers.27.mlp, model.layers.28.mlp, model.layers.29.mlp, model.layers.30.mlp, model.layers.31.mlp" # 2.1B
metrics="all" # "default", "MSE", "FID", "COR", "oCOR", "interCOR", "MSE, FID, COR, oCOR, interCOR", "weightCOR", "all"
freeshold=1
nb_concepts=100

# DATASET LOADING
dataset_name="cornell-movie-review-data/rotten_tomatoes" # "wikimedia/wikipedia", "cornell-movie-review-data/rotten_tomatoes"
nb_samples=250
dataset_split="train" # "train", "test", "20231101.en"
text_field="text" # "text", "review"
sample_max_len=1000 # Set to a positive integer to limit text samples to that many characters, or 0 for no limit

# If you do not want a unique plot, you can remove the --unique_plot argument
# If you do not want to print information, you can remove the --verbose argument

# Do the first step with the reference step
$PYTHON_CMD $SCRIPT_PATH --model_name $model_name --steps 48 --verbose --test_name $test_name --decomposer $decomposer --nb_concepts $nb_concepts --decomposition_method $decomposition_method --split_point "$split_point" --metrics "$metrics" --unique_plot --freeshold $freeshold --nb_dataset_samples $nb_samples --reference_steps 48 --gradient_plot --log_scale --dataset $dataset_name --dataset_split $dataset_split --text_field $text_field --sample_max_len $sample_max_len #--kwargs '{"max_iter": 1500}'

for i in {1..24}; do
    step1=$((2 * i - 1))
    step2=$((2 * i))
    $PYTHON_CMD $SCRIPT_PATH --model_name $model_name --steps $step1 $step2 --verbose --test_name $test_name --decomposer $decomposer --nb_concepts $nb_concepts --decomposition_method $decomposition_method --split_point "$split_point" --metrics "$metrics" --unique_plot --freeshold $freeshold --nb_dataset_samples $nb_samples --reference_steps 48 --gradient_plot --log_scale --visualization --dataset $dataset_name --dataset_split $dataset_split --text_field $text_field --sample_max_len $sample_max_len #--kwargs '{"max_iter": 1500}'
done

# Prepare and save files
mkdir "$test_name${decomposer}_${decomposition_method}"
mv "${test_name}_"* "$test_name${decomposer}_${decomposition_method}" 2>/dev/null
mv "${test_name}.json" "$test_name${decomposer}_${decomposition_method}" 2>/dev/null
zip -r "${test_name}.zip" "$test_name${decomposer}_${decomposition_method}"

