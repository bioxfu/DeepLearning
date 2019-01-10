#### Install [PlantCV](https://plantcv.readthedocs.io/en/latest/installation/)
```
# Clone PlantCV
git clone https://github.com/danforthcenter/plantcv.git

# Enter the PlantCV directory 
cd plantcv

# Create an Anaconda environment named "plantcv" and automatically install the dependencies
conda create -y --file requirements.txt -n plantcv python=3.5 jupyter
conda install -y -c menpo opencv3 

# Activate the plantcv environment (you will have to do this each time you start a new session)
source activate plantcv

# Install PlantCV
python setup.py install

# If PlantCV is installed successfully it should import without error
python -c 'import cv2'
python -c 'import plantcv'

# Optionally, you can run automated tests on your system to make sure everything is working correctly
python setup.py test
```

#### Jupyter Notebook
```
# Set Jupyter password (optional)
jupyter notebook password

# Start Jupyter notebook
jupyter notebook 
```

