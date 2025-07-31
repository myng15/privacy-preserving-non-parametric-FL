SEEDS=(3407 12345 202502)
client_generation_seed=202502

s_frac=4600

backbone="base_patch14_dinov2" #"small_patch14_dinov2 "small_patch16_dino" "small_patch16_augreg"

val_frac=0.1 
knn_metric="euclidean" #"cosine"
knn_weights="gaussian_kernel"
global_sampling="random" 
n_neighbors=3 
device="cuda:1" 

classifiers=("linear") #"knn" "mlp"
local_epochs=100 # if classifier = "linear"/"mlp"
classifier_optimizer="adam" #"sgd" 

anonymizers=("cvae_fedavg") # "cgan_fedavg" 
cvae_layers=3 
n_fedavg_rounds_values=(50)
anonymizer_lr_values=(0.001) 
total_generated_factor=1.0
augmentation_strategy="inverse" #"replicate" #"uniform" #"uniform_all_classes"

dp="dp" #"non-dp" 
max_grad_norm=1.5
noise_multiplier=1.0 
epsilons=(1.0) 
delta=1e-4


for anonymizer in "${anonymizers[@]}"; do
    for anonymizer_lr in "${anonymizer_lr_values[@]}"; do
        for n_fedavg_rounds in "${n_fedavg_rounds_values[@]}"; do
            for classifier in "${classifiers[@]}"; do
                for epsilon in "${epsilons[@]}"; do
                    # Get the current date and time
                    current_time=$(date +"%Y_%m_%d_%H_%M")

                    # DP:
                    base_chkpts_dir="chkpts/camelyon17/$backbone/s_frac=${s_frac}/gen_seed=${client_generation_seed}/val_frac=${val_frac}/${classifier}_${anonymizer}/${cvae_layers}_layer_CVAE/anonymizer_lr=${anonymizer_lr}/${n_fedavg_rounds}_fedavg_rounds/${dp}/mgn=${max_grad_norm}-eps=${epsilon}-delta=${delta}/${current_time}" 
                    base_results_dir="results/camelyon17/$backbone/s_frac=${s_frac}/gen_seed=${client_generation_seed}/val_frac=${val_frac}/global_sampling=${global_sampling}/${classifier}_${anonymizer}/${cvae_layers}_layer_CVAE/knn_metric=${knn_metric}/${knn_weights}/n_neighbors=${n_neighbors}/classifier_optimizer=${classifier_optimizer}${local_epochs}_local_epochs/anonymizer_lr=${anonymizer_lr}/${n_fedavg_rounds}_fedavg_rounds/${dp}/mgn=${max_grad_norm}-eps=${epsilon}-delta=${delta}/total_generated_factor=${total_generated_factor}/augmentation=${augmentation_strategy}/${current_time}"
                    
                    # Non-DP:
                    # base_chkpts_dir="chkpts/camelyon17/$backbone/s_frac=${s_frac}/gen_seed=${client_generation_seed}/val_frac=${val_frac}/${classifier}_${anonymizer}/anonymizer_lr=${anonymizer_lr}/${n_fedavg_rounds}_fedavg_rounds/${dp}/${current_time}" 
                    # base_results_dir="results/camelyon17/$backbone/s_frac=${s_frac}/gen_seed=${client_generation_seed}/val_frac=${val_frac}/global_sampling=${global_sampling}/${classifier}_${anonymizer}/knn_metric=${knn_metric}/${knn_weights}/n_neighbors=${n_neighbors}/${local_epochs}_local_epochs/anonymizer_lr=${anonymizer_lr}/${n_fedavg_rounds}_fedavg_rounds/${dp}/total_generated_factor=${total_generated_factor}/augmentation=${augmentation_strategy}/${current_time}"
                    
                    results_all_seeds_file="${base_results_dir}/results_all_seeds.txt"
                    echo "Experiment Results Log - Generated on ${current_time}" > "$results_all_seeds_file"

                    for SEED in "${SEEDS[@]}"; do
                        echo "=> Generate data with seed $client_generation_seed"
                        
                        rm -rf data/camelyon17/all_clients_data
                        
                        python data/camelyon17/generate_data.py \
                            --s_frac $s_frac \
                            --n_clients 5 \
                            --test_clients_frac 0.0 \
                            --test_data_frac 0.2 \
                            --val_data_frac $val_frac \
                            --seed $client_generation_seed

                        current_seed_time=$(date +"%Y_%m_%d_%H_%M")
                        chkpts_path="${base_chkpts_dir}/${current_seed_time}"  
                        results_dir="${base_results_dir}/seed=${SEED}-chkpt=${current_seed_time}"
                        
                        python eval_ours.py \
                            camelyon17 \
                            random \
                            --aggregator_type fedknn \
                            --client_type fedknn \
                            --bz 128 \
                            --backbone $backbone \
                            --chkpts_path $chkpts_path \
                            --val_frac $val_frac \
                            --n_neighbors $n_neighbors \
                            --classifier $classifier \
                            --classifier_optimizer $classifier_optimizer \
                            --anonymizer $anonymizer \
                            --n_fedavg_rounds $n_fedavg_rounds \
                            --anonymizer_lr $anonymizer_lr \
                            --total_generated_factor $total_generated_factor \
                            --augmentation_strategy $augmentation_strategy \
                            --global_sampling $global_sampling \
                            --capacities_grid_resolution 1.0 \
                            --weights_grid_resolution 0.1 \
                            --knn_metric $knn_metric \
                            --knn_weights $knn_weights \
                            --gaussian_kernel_scale 1.0 \
                            --local_epochs $local_epochs \
                            --device $device \
                            --results_dir $results_dir \
                            --seed $SEED \
                            --verbose 1 \
                            --enable_dp \
                            --max_grad_norm $max_grad_norm \
                            --noise_multiplier $noise_multiplier \
                            --epsilon $epsilon \
                            --delta $delta \
                        
                        echo "----------------------------------------------------"

                        # Extract best accuracy and best weight, save to results
                        BEST_ACC=$(grep "Best Accuracy in Grid-Search evaluation:" "${results_dir}/results.log" | awk '{print $NF}')
                        BEST_BALANCED_ACC=$(grep "Best Balanced Accuracy in Grid-Search evaluation:" "${results_dir}/results.log" | awk '{print $NF}')
                        BEST_AUC=$(grep "Best ROC-AUC in Grid-Search evaluation:" "${results_dir}/results.log" | awk '{print $NF}')
                        BEST_F1_MACRO=$(grep "Best F1 Score (Macro) in Grid-Search evaluation:" "${results_dir}/results.log" | awk '{print $NF}') 
                        BEST_F1_WEIGHTED=$(grep "Best F1 Score (Weighted) in Grid-Search evaluation:" "${results_dir}/results.log" | awk '{print $NF}') 
                        BEST_WEIGHT=$(grep "Optimal weight:" "${results_dir}/results.log" | awk '{print $NF}')
                        BEST_WEIGHT_BALANCED=$(grep "Optimal weight for Balanced Accuracy:" "${results_dir}/results.log" | awk '{print $NF}')
                        BEST_WEIGHT_AUC=$(grep "Optimal weight for ROC-AUC:" "${results_dir}/results.log" | awk '{print $NF}')
                        BEST_WEIGHT_F1_MACRO=$(grep "Optimal weight for F1 Score (Macro):" "${results_dir}/results.log" | awk '{print $NF}') 
                        BEST_WEIGHT_F1_WEIGHTED=$(grep "Optimal weight for F1 Score (Weighted):" "${results_dir}/results.log" | awk '{print $NF}') 
                        AVG_TEST_ACC=$(grep "Test Accuracy using Optimal weight" "${results_dir}/results.log" | awk -F': ' '{print $2}' | awk '{print $1}')
                        AVG_TEST_BALANCED_ACC=$(grep "Test Balanced Accuracy using Optimal weight" "${results_dir}/results.log" | awk -F': ' '{print $2}' | awk '{print $1}')
                        AVG_TEST_AUC=$(grep "Test ROC-AUC using Optimal weight" "${results_dir}/results.log" | awk -F': ' '{print $2}' | awk '{print $1}')
                        AVG_TEST_F1_MACRO=$(grep "Test F1 Score (Macro) using Optimal weight" "${results_dir}/results.log" | awk -F': ' '{print $2}' | awk '{print $1}') 
                        AVG_TEST_F1_WEIGHTED=$(grep "Test F1 Score (Weighted) using Optimal weight" "${results_dir}/results.log" | awk -F': ' '{print $2}' | awk '{print $1}') 

                        echo "Best Accuracy in Grid-Search evaluation for seed $SEED: $BEST_ACC %"
                        echo "Best Balanced Accuracy in Grid-Search evaluation for seed $SEED: $BEST_BALANCED_ACC %"
                        echo "Best ROC-AUC in Grid-Search evaluation for seed $SEED: $BEST_AUC"
                        echo "Best F1 Score (Macro) in Grid-Search evaluation for seed $SEED: $BEST_F1_MACRO" # F1
                        echo "Best F1 Score (Weighted) in Grid-Search evaluation for seed $SEED: $BEST_F1_WEIGHTED" # F1 
                        echo "Optimal weight for seed $SEED: $BEST_WEIGHT"
                        echo "Optimal weight for Balanced Accuracy for seed $SEED: $BEST_WEIGHT_BALANCED"
                        echo "Optimal weight for ROC-AUC for seed $SEED: $BEST_WEIGHT_AUC"
                        echo "Optimal weight for F1 Score (Macro) for seed $SEED: $BEST_WEIGHT_F1_MACRO" # F1
                        echo "Optimal weight for F1 Score (Weighted) for seed $SEED: $BEST_WEIGHT_F1_WEIGHTED" # F1
                        echo "Average test accuracy for seed $SEED: $AVG_TEST_ACC %"
                        echo "Average test balanced accuracy for seed $SEED: $AVG_TEST_BALANCED_ACC %"
                        echo "Average test ROC-AUC for seed $SEED: $AVG_TEST_AUC"
                        echo "Average test F1 Score (Macro) for seed $SEED: $AVG_TEST_F1_MACRO" # F1
                        echo "Average test F1 Score (Weighted) for seed $SEED: $AVG_TEST_F1_WEIGHTED" # F1

                        # Save both metrics to the global metrics file
                        echo "$BEST_ACC $BEST_WEIGHT $BEST_BALANCED_ACC $BEST_WEIGHT_BALANCED $BEST_AUC $BEST_WEIGHT_AUC $BEST_F1_MACRO $BEST_WEIGHT_F1_MACRO $BEST_F1_WEIGHTED $BEST_WEIGHT_F1_WEIGHTED $AVG_TEST_ACC $AVG_TEST_BALANCED_ACC $AVG_TEST_AUC $AVG_TEST_F1_MACRO $AVG_TEST_F1_WEIGHTED" >> "$results_all_seeds_file" #"${base_results_dir}/results_all_seeds.txt" # F1

                        echo "----------------------------------------------------"
                    
                    done
                    
                    # Aggregate results after running all three seeds
                    echo "=> Aggregating results across all seeds:"
                    
                    python utils_ours.py \
                        --results_dir "${base_results_dir}" \
                        --seeds ${SEEDS[@]} >> "$results_all_seeds_file"
                done
            done
        done
    done
done


