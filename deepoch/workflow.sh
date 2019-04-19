# Step1. Edit config.py file

# Step2. Build HDF5 files
python build_hdf5.py

# Step3. Transfer-learnign
python train.py

curl -X POST -F image=@Tomato__healthy.JPG 'http://localhost:5000/predict'

curl -X POST -F image=@Potato___Early_blight.JPG 'http://localhost:5000/predict'
