SEEDS=(3407 12345 202502)
client_generation_seed=202502

alphas=(0.3) 
split_method="by_labels_split"
s_frac=1.0
n_clients_values=(10)

backbone="base_patch14_dinov2" #"small_patch14_dinov2 "small_patch16_dino" "small_patch16_augreg"

val_frac=0.1
knn_metric="euclidean" #"cosine"
knn_weights="gaussian_kernel" #"inverse_distances"
global_sampling="random" 
n_neighbors=3 
device="cuda:0"

classifiers=("linear") # #"knn" "mlp"
local_epochs=100 # if classifier = "linear"/"mlp"
classifier_optimizer="adam" 

anonymizers=("centroids")
k_same_values=(10)
clustering_values=("unsupervised")  
return_centroids="False" #"False" for kSame, "True" for sharing only centroids

dp="dp" #"non_dp"
centroids_dp_noise="gaussian"
max_feat_norm=1.5
epsilons=(1.0) 
delta=1e-4 


for n_clients in "${n_clients_values[@]}"; do
    for alpha in "${alphas[@]}"; do
        for classifier in "${classifiers[@]}"; do
            for anonymizer in "${anonymizers[@]}"; do
                for clustering in "${clustering_values[@]}"; do
                    for k_same in "${k_same_values[@]}"; do
                        for epsilon in "${epsilons[@]}"; do
                            # Get the current date and time
                            current_time=$(date +"%Y_%m_%d_%H_%M")

                            # DP:
                            # base_chkpts_dir="chkpts/organsmnist/$backbone/s_frac=${s_frac}/${n_clients}_clients/${split_method}/alpha=${alpha}/gen_seed=${client_generation_seed}/val_frac=${val_frac}/${classifier}_${anonymizer}/knn_metric=${knn_metric}/${knn_weights}/n_neighbors=${n_neighbors}/classifier_optimizer=${classifier_optimizer}/${local_epochs}_local_epochs/k_same=${k_same}/${clustering}/return_centroids=${return_centroids}/${dp}/${centroids_dp_noise}-mfn=${max_feat_norm}-eps=${epsilon}-delta=${delta}/global_sampling=${global_sampling}/${current_time}"
                            # base_results_dir="results/organsmnist/$backbone/s_frac=${s_frac}/${n_clients}_clients/${split_method}/alpha=${alpha}/gen_seed=${client_generation_seed}/val_frac=${val_frac}/${classifier}_${anonymizer}/knn_metric=${knn_metric}/${knn_weights}/n_neighbors=${n_neighbors}/classifier_optimizer=${classifier_optimizer}/${local_epochs}_local_epochs/k_same=${k_same}/${clustering}/return_centroids=${return_centroids}/${dp}/${centroids_dp_noise}-mfn=${max_feat_norm}-eps=${epsilon}-delta=${delta}/global_sampling=${global_sampling}/${current_time}"
                            
                            # Non-DP:
                            base_chkpts_dir="chkpts/organsmnist/$backbone/s_frac=${s_frac}/${n_clients}_clients/${split_method}/alpha=${alpha}/gen_seed=${client_generation_seed}/val_frac=${val_frac}/${classifier}_${anonymizer}/knn_metric=${knn_metric}/${knn_weights}/n_neighbors=${n_neighbors}/classifier_optimizer=${classifier_optimizer}/${local_epochs}_local_epochs/k_same=${k_same}/${clustering}/return_centroids=${return_centroids}/${dp}/${current_time}"
                            base_results_dir="results/organsmnist/$backbone/s_frac=${s_frac}/${n_clients}_clients/${split_method}/alpha=${alpha}/gen_seed=${client_generation_seed}/val_frac=${val_frac}/${classifier}_${anonymizer}/knn_metric=${knn_metric}/${knn_weights}/n_neighbors=${n_neighbors}/classifier_optimizer=${classifier_optimizer}/${local_epochs}_local_epochs/k_same=${k_same}/${clustering}/return_centroids=${return_centroids}/${dp}/${current_time}"

                            results_all_seeds_file="${base_results_dir}/results_all_seeds.txt"
                            echo "Experiment Results Log - Generated on ${current_time}" > "$results_all_seeds_file"

                            for SEED in "${SEEDS[@]}"; do
                                echo "=> Generate data with seed $client_generation_seed"
                                
                                rm -rf data/organsmnist/all_clients_data
                                
                                python data/organsmnist/generate_data.py \
                                    --n_clients $n_clients \
                                    --split_method $split_method \
                                    --n_components -1 \
                                    --alpha $alpha \
                                    --s_frac $s_frac \
                                    --test_clients_frac 0.0 \
                                    --test_data_frac 0.2 \
                                    --seed $client_generation_seed

                                current_seed_time=$(date +"%Y_%m_%d_%H_%M")
                                chkpts_path="${base_chkpts_dir}/${current_seed_time}"  
                                results_dir="${base_results_dir}/seed=${SEED}-chkpt=${current_seed_time}"
                                
                                python eval_ours.py \
                                    organsmnist \
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
                                    --k_same $k_same \
                                    --clustering $clustering \
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
                                    --return_centroids \

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
                                echo "Best F1 Score (Macro) in Grid-Search evaluation for seed $SEED: $BEST_F1_MACRO" 
                                echo "Best F1 Score (Weighted) in Grid-Search evaluation for seed $SEED: $BEST_F1_WEIGHTED" 
                                echo "Optimal weight for seed $SEED: $BEST_WEIGHT"
                                echo "Optimal weight for Balanced Accuracy for seed $SEED: $BEST_WEIGHT_BALANCED"
                                echo "Optimal weight for ROC-AUC for seed $SEED: $BEST_WEIGHT_AUC"
                                echo "Optimal weight for F1 Score (Macro) for seed $SEED: $BEST_WEIGHT_F1_MACRO"
                                echo "Optimal weight for F1 Score (Weighted) for seed $SEED: $BEST_WEIGHT_F1_WEIGHTED" 
                                echo "Average test accuracy for seed $SEED: $AVG_TEST_ACC %"
                                echo "Average test balanced accuracy for seed $SEED: $AVG_TEST_BALANCED_ACC %"
                                echo "Average test ROC-AUC for seed $SEED: $AVG_TEST_AUC"
                                echo "Average test F1 Score (Macro) for seed $SEED: $AVG_TEST_F1_MACRO"
                                echo "Average test F1 Score (Weighted) for seed $SEED: $AVG_TEST_F1_WEIGHTED" 

                                # Save both metrics to the global metrics file
                                echo "$BEST_ACC $BEST_WEIGHT $BEST_BALANCED_ACC $BEST_WEIGHT_BALANCED $BEST_AUC $BEST_WEIGHT_AUC $BEST_F1_MACRO $BEST_WEIGHT_F1_MACRO $BEST_F1_WEIGHTED $BEST_WEIGHT_F1_WEIGHTED $AVG_TEST_ACC $AVG_TEST_BALANCED_ACC $AVG_TEST_AUC $AVG_TEST_F1_MACRO $AVG_TEST_F1_WEIGHTED" >> "$results_all_seeds_file" #"${base_results_dir}/results_all_seeds.txt"   # F1

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
    done
done