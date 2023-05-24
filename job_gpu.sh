#!/bin/sh
#BSUB -J mpt_bert_sst2
#BSUB -q gpua100
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -o ./lsf_output/modeldriven_%J_%I.out
#BSUB -e ./lsf_output/modeldriven_%J_%I.err

source ../venv/bin/activate
python3 ./bert.py 


