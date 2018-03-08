# Set environment variables 
source set_env.sh

# Move to Fonduer home directory
cd "$FONDUERHOME"

# Launch jupyter notebook!
echo "Launching Jupyter Notebook..."
jupyter notebook --notebook-dir="$FONDUERHOME"
