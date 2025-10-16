#############################################################################################################################################
# How to use :
# 1. Make sure you have installed Interpreto from pip install
#    pip install interpreto
# 2. Make sure you have installed the required packages from Interpreto_requirements.txt (it should be automatic with pip install interpreto)
# 3. Make sure you have the correct path to the code in the script
# 4. Use bash main_learning.sh to run the script
#############################################################################################################################################

# You can modify the parameters below to suit your needs :

model_size="210m" # "210m", "610m", "2.1B"
model_name="EuroBERT/EuroBERT-${model_size}"
timestamp=$(date +"%Y%m%d-%H%M%S")
test_name="test_${model_size}${timestamp}"
decomposer="LearningProcess"
split_point="model.layers.1.mlp, model.layers.2.mlp, model.layers.3.mlp, model.layers.4.mlp, model.layers.5.mlp, model.layers.6.mlp, model.layers.7.mlp, model.layers.8.mlp, model.layers.9.mlp, model.layers.10.mlp, model.layers.11.mlp"
freeshold=3
decomposition_method="Activation" # "Weight", "Activation"

# DATASET LOADING
dataset_name="wikimedia/wikipedia" # "wikipedia", "cornell-movie-review-data/rotten_tomatoes"
nb_samples=2000
dataset_split="20231101.en" # "train", "test"
text_field="text" # "text", "review"

# If you do not want a unique plot, you can remove the --unique_plot argument
# If you do not want to print information, you can remove the --verbose argument

for i in {1..24}; do
    step1=$((2 * i - 1))
    step2=$((2 * i))
    python concept_dynamic/code/main.py --model_name $model_name --steps $step1 $step2 --verbose --test_name $test_name --decomposer $decomposer --split_point "$split_point" --unique_plot --freeshold $freeshold --nb_dataset_samples $nb_samples --convergence --decomposition_method $decomposition_method --gradient_plot --log_scale --visualization --dataset $dataset_name --dataset_split $dataset_split --text_field $text_field
done

# Prepare and save files
mkdir "$test_name${decomposer}"
mv "${test_name}_"* "$test_name${decomposer}" 2>/dev/null
mv "${test_name}.json" "$test_name${decomposer}" 2>/dev/null
zip -r "${test_name}.zip" "$test_name${decomposer}"