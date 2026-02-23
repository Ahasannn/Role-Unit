#!/bin/bash
#SBATCH --job-name=gen_plots
#SBATCH --output=logs/plot_generation_%j.out
#SBATCH --error=logs/plot_generation_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --partition=hpg-default

# Generate motivation plots from CSV data
# Saves plots to visualization/output_plots/

cd /home/ah872032.ucf/system-aware-mas

# Activate virtual environment
source .venv/bin/activate

# Run plot generation script
python3 visualization/generate_motivation_plots.py

echo ""
echo "✓ Plots saved to: visualization/output_plots/"
echo "  Download them with: scp or rsync"
