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

echo "Start time: $(date)"
for orbits in 7 6 5; do
    for models in 'gcn' 'gat' 'unique_id_gcn' 'rni_gcn' 'orbit_indiv_gcn' 'max_orbit_gcn'; do
        python main_alchemy.py --model_max_orbit $orbits --model $models
    done
done