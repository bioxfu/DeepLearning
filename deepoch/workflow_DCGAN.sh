git clone https://github.com/carpedm20/DCGAN-tensorflow
cd DCGAN-tensorflow/

#DATASET_ROOT_DIR=../../datasets/
#DATASET_NAME=anime

DATASET_ROOT_DIR=../../datasets/cars
DATASET_NAME=car_ims_96_96

python main.py --dataset $DATASET_NAME --data_dir $DATASET_ROOT_DIR --input_fname_pattern "*.jpg" --input_height 96 --output_height 48 --crop --train --epoch 10

