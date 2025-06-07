#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --partition=accelerated
#SBATCH --job-name=max_orbit
#SBATCH --nodes=1
#SBATCH --output=log/Orbit_GC%j.output
#SBATCH --error=error/Orbit_GC%j.error
#SBATCH --account=hk-project-pai00001   # specify the project group
#SBATCH --gres=gpu:1
#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-automorphism/orbit-gnn/

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cshao676@gmail.com

# Request GPU resources

source /hkfs/home/project/hk-project-test-p0021478/cc7738/anaconda3/etc/profile.d/conda.sh

conda activate base

 
# <<< conda initialize <<<
module purge
module load devel/cmake/3.18   
module load devel/cuda/11.8   
module load compiler/gnu/12
conda activate EAsF
cd /hkfs/work/workspace/scratch/cc7738-automorphism/orbit-gnn/
 
echo ">>> .bashrc executed: Environment and modules are set up. <<<"

#!/bin/bash

echo "Start time: $(date)"

# Array of orbit values
orbits_list=(2 3 4 5 6 7 8 9)

# Counter for grouping
counter=0
batch_size=3

for orbits in "${orbits_list[@]}"; do
    # Launch in background
    python main_alchemy.py \
        --model_max_orbit "$orbits" \
        --model 'max_orbit_gcn' \
        --n_epochs 1000 \
        --loss_log_interval 10 \
        --train_eval_interval 10 \
        --test_eval_interval 10 \
        --runs 10 &

    ((counter++))

    # Wait for batch to finish
    if (( counter % batch_size == 0 )); then
        wait
    fi
done

# Wait for any remaining processes
wait

echo "All jobs done at: $(date)"


# TODO 'gcn' 'gat' 'unique_id_gcn' 'rni_gcn' 'orbit_indiv_gcn' 