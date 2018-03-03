# Set environment variables 
source set_env.sh

# Move to Fonduer home directory
cd "$FONDUERHOME"

# Make sure phantomjs is installed
PHANTOMJS="phantomjs/bin/phantomjs"
if [ ! -f "$PHANTOMJS" ]; then
    read -p "phantomjs not found- install now?  [y/n] " yn
    case $yn in
        [Yy]* ) echo "Installing phantomjs..."; ./install-phantomjs.sh;;
        [Nn]* ) ;;
    esac
fi

# Launch jupyter notebook!
echo "Launching Jupyter Notebook..."
jupyter notebook --notebook-dir="$FONDUERHOME"
