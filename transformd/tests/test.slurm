#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o ./%x_%j.out
#SBATCH --time=5:00:00

python /jumbo/burginlab/myra/transformd/transformd/main.py ./test.config