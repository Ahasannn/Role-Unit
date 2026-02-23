#!/bin/bash
# Setup Jupyter kernel for HiperGator JupyterHub
# Run this once to register your virtual environment as a Jupyter kernel

cd /home/ah872032.ucf/system-aware-mas

# Activate virtual environment
source .venv/bin/activate

# Install notebook packages if missing
pip install notebook nbformat nbconvert ipykernel

# Register kernel with Jupyter
python -m ipykernel install --user \
    --name=system-aware-mas \
    --display-name="Python (MasRouter)"

echo ""
echo "✓ Jupyter kernel 'Python (MasRouter)' installed successfully!"
echo ""
echo "Next steps:"
echo "1. Go to https://ood.rc.ufl.edu/"
echo "2. Launch Jupyter Notebook from Interactive Apps"
echo "3. Open your notebook: visualization/motivation_plots.ipynb"
echo "4. Select kernel: 'Python (MasRouter)' from Kernel menu"
echo ""
