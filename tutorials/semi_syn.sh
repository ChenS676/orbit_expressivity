#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=accelerated
#SBATCH --job-name=o_gat_356
# Array of orbit values

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

orbits_list=(2)
# <<< conda initialize <<<
module purge
module load devel/cmake/3.18   
module load devel/cuda/11.8   
module load compiler/gnu/12
conda activate EAsF
cd /hkfs/work/workspace/scratch/cc7738-automorphism/archiv/orbit-gnn
echo ">>> .bashrc executed: Environment and modules are set up. <<<"

#!/bin/bash

echo "Start time: $(date)"

# Counter for grouping
counter=0
batch_size=3

for orbits in "${orbits_list[@]}"; do
    # Launch in background
    echo "Running with model_max_orbit: python main.py --max_orbit_alchemy $orbits"
    python main.py --max_orbit_alchemy $orbits
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