SEEDS=(3407 12345 202502)
client_generation_seed=202502

alphas=(0.3)
split_method="by_labels_split"
s_frac=1.0
n_clients_values=(10)

backbone="base_patch14_dinov2" #"small_patch14_dinov2 "small_patch16_dino" "small_patch16_augreg"

val_frac=0.1 
client_types=("normal") 
aggregator_types=("centralized_linear") 
algorithms=("normal") # "normal" "fedadam" "fedprox"
learning_rates=(0.001)
num_rounds=(50)

lr_scheduler="multi_step"
model_name="linear" #"mlp" 
device="cuda:1"

for n_clients in "${n_clients_values[@]}"; do
    for client_type in "${client_types[@]}"; do
        for alpha in "${alphas[@]}"; do
            for aggregator_type in "${aggregator_types[@]}"; do
                for algorithm in "${algorithms[@]}"; do
                    for lr in "${learning_rates[@]}"; do
                        for n_rounds in "${num_rounds[@]}"; do
                            # Get the current date and time in the format YYYY_MM_DD_HH_MM
                            current_time=$(date +"%Y_%m_%d_%H_%M")

                            base_logs_dir="logs/organsmnist/$backbone/s_frac=${s_frac}/${n_clients}_clients/client_type=${client_type}/${split_method}/alpha=${alpha}/gen_seed=${client_generation_seed}/val_frac=${val_frac}/fedavg_${model_name}/aggregator=${aggregator_type}/algorithm=${algorithm}/lr=${lr}/${n_rounds}_rounds/${current_time}" 
                            base_chkpts_dir="chkpts/organsmnist/$backbone/s_frac=${s_frac}/${n_clients}_clients/client_type=${client_type}/${split_method}/alpha=${alpha}/gen_seed=${client_generation_seed}/val_frac=${val_frac}/fedavg_${model_name}/aggregator=${aggregator_type}/algorithm=${algorithm}/lr=${lr}/${n_rounds}_rounds/${current_time}" 
                            base_results_dir="results/organsmnist/$backbone/s_frac=${s_frac}/${n_clients}_clients/client_type=${client_type}/${split_method}/alpha=${alpha}/gen_seed=${client_generation_seed}/val_frac=${val_frac}/fedavg_${model_name}/aggregator=${aggregator_type}/algorithm=${algorithm}/lr=${lr}/${n_rounds}_rounds/${current_time}"
                            
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
                                logs_dir="${base_logs_dir}/${current_seed_time}"  
                                chkpts_dir="${base_chkpts_dir}/${current_seed_time}"  
                                results_dir="${base_results_dir}/seed=${SEED}-chkpt=${current_seed_time}"

                                python eval_fedavg.py \
                                    organsmnist \
                                    --model_name $model_name \
                                    --aggregator_type $aggregator_type \
                                    --client_type $client_type \
                                    --algorithm $algorithm \
                                    --bz 128 \
                                    --backbone $backbone \
                                    --val_frac $val_frac \
                                    --n_rounds $n_rounds \
                                    --lr $lr \
                                    --lr_scheduler $lr_scheduler \
                                    --log_freq 10 \
                                    --eval_freq 10 \
                                    --device $device \
                                    --optimizer sgd \
                                    --logs_dir $logs_dir \
                                    --chkpts_dir $chkpts_dir \
                                    --results_dir $results_dir \
                                    --seed $SEED  \
                                    --verbose 1

                                echo "----------------------------------------------------"

                            done

                            # Aggregate results after running all three seeds
                            echo "=> Aggregating results across all seeds:"

                            python3 utils_baseline.py \
                                --results_dir "${base_results_dir}" \
                                --seeds ${SEEDS[@]} >> "$results_all_seeds_file"

                            echo "----------------------------------------------------"

                        done
                    done
                done
            done
        done
    done
done







