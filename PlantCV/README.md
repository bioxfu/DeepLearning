#### Install [PlantCV](https://plantcv.readthedocs.io/en/latest/installation/)
```
# Clone PlantCV
git clone https://github.com/danforthcenter/plantcv.git

# Enter the PlantCV directory 
cd plantcv

# Create an Anaconda environment named "plantcv" and automatically install the dependencies
conda create --file requirements.txt -n plantcv -c conda-forge python=3.6 opencv=3

# Activate the plantcv environment (you will have to do this each time you start a new session)
source activate plantcv

# Install PlantCV
python setup.py install

# If PlantCV is installed successfully it should import without error
python -c 'import plantcv'

# Optionally, you can run automated tests on your system to make sure everything is working correctly
python setup.py test
```
