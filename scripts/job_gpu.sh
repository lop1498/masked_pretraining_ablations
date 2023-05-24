#!/bin/bash	
#SBATCH --job-name=cola_mpt
#SBATCH --output=cola_result_-%J.out
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00
#SBATCH --mem=2gb
#SBATCH --gres=gpu
#SBATCH --mail-user=polgarciarecasens@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --export=ALL

conda activate myenv
source ../../venv/bin/activate
python3 ../bert.py 
