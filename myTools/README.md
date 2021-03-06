#### Starter Bundle

##### ShallowNet on CIFAR-10 (p206)
```
python bin/shallownet_cifar10.py -o output/shallownet_cifar10.png
```

##### ShallowNet on Animals (p202, p211, p214)
```
python bin/shallownet_animals_train.py -d datasets/animals -o output/shallownet_animals.png -m output/shallownet_animals.hdf5

python bin/shallownet_animals_load.py -d datasets/animals -m output/shallownet_animals.hdf5
```

##### LeNet on MNIST (p220)
```
python bin/lenet_mnist.py -o output/lenet_mnist.png
```

##### Architecture Visualization of LeNet (272)
```
python bin/visualize_architecture.py
```

##### MiniVGGNet on CIFAR-10 with Two Learning Rate Schedulers (p230, p245)
```
python bin/minivggnet_cifar10_lr_time_based_decay.py -o output/minivggnet_cifar10_lr_time_based_decay.png

python bin/minivggnet_cifar10_lr_drop_based_decay.py -o output/minivggnet_cifar10_lr_drop_based_decay.png
```

#### Monitor the Training Process and Checkpoint the Best Model (p255, p267)
```
python bin/monitor_and_checkpoint_best_cifar10.py -o output -w output/monitor_and_checkpoint_best_cifar10.hdf5
```

#### Classifying Images with Pre-trained ImageNet CNN (p285)
```
python bin/imagenet_pretrained.py --image datasets/example_images/example_01.jpg --model vgg16
```

#### Smile Detection (p307)
``` 
mkdir -p datasets/SMILEsmileD_balanced/SMILEs/negatives/negatives7
mkdir -p datasets/SMILEsmileD_balanced/SMILEs/positives/positives7

cat datasets/SMILEsmileD/SMILEs/negatives/smiles_01_neg.idx|head -3690 |xargs -I {} cp datasets/SMILEsmileD/SMILEs/negatives/{} datasets/SMILEsmileD_balanced/SMILEs/negatives/negatives7/

cp datasets/SMILEsmileD/SMILEs/positives/positives7/* datasets/SMILEsmileD_balanced/SMILEs/positives/positives7/

python bin/SMILEsmileD_lenet_train.py --dataset datasets/SMILEsmileD_balanced --model output/SMILEsmileD_lenet.hdf5 -o output/SMILEsmileD_lenet.png

python bin/SMILEsmileD_minivggnet_train.py --dataset datasets/SMILEsmileD_balanced --model output/SMILEsmileD_minivggnet.hdf5 -o output/SMILEsmileD_minivggnet.png

python bin/SMILEsmileD_detect_smile.py --cascade bin/CV/haarcascade_frontalface_default.xml --model output/SMILEsmileD_lenet.hdf5 --video datasets/myface.mov

python bin/SMILEsmileD_detect_smile.py --cascade bin/CV/haarcascade_frontalface_default.xml --model output/SMILEsmileD_minivggnet.hdf5 --video datasets/myface.mov
```


#### Practitioner Bundle
#### Data Augmentation
```
python bin/augmentation_demo.py -i /home/xfu/Git/DeepLearning/myTools/datasets/animals/cats/cats_00001.jpg -o output/image_aug

# download flower data
# wget http://download.tensorflow.org/example_images/flower_photos.tgz

python bin/minivggnet_flowers_data_aug.py -d datasets/flower_photos -o output/minivggnet_flowers_data_aug.png
```

#### Feature Extraction
```
python bin/extract_features_with_VGG16.py -d datasets/animals -o output/animals_VGG16_features.hdf5

python bin/logistic_regression_on_extracted_features.py --db output/animals_VGG16_features.hdf5 --model output/animals_logreg.cpickle
```

#### Rank-1 and Rank-5 Accuracies
```
python bin/rank_accuracy.py --db output/animals_VGG16_features.hdf5 --model output/animals_logreg.cpickle
```

#### Fine-tuning
```
#inspect model's layers
python bin/inspect_model_layers.py
python bin/inspect_model_layers.py --include-top -1

# nework surgery and fine-tuning
python bin/finetune_with_VGG16.py -d datasets/flower_photos -m output/finetune_with_VGG16_flowers_model.hdf5
```

#### Ensembl of CNNs
```
# train
python bin/ensembl_minivggnet_train.py -o output/ensembl -m output/ensembl 

# test
python bin/ensembl_minivggnet_evaluate.py -m output/ensembl 
```

#### dogs_vs_cats
```
cd project/dogs_vs_cats
# build HDF5 files
python img2hdf5.py

# train AlexNet
python train_alexnet.py

# evaluate AlexNet
python evaluate_alexnet.py

# extract features with ResNet
python extract_features_with_ResNet.py --dataset ../../datasets/kaggle_dogs_vs_cats/train/ --output ../../datasets/kaggle_dogs_vs_cats/hdf5/features_ResNet.hdf5

# traing a logistic regression classifier
python logistic_regression_on_extracted_features.py --db ../../datasets/kaggle_dogs_vs_cats/hdf5/features_ResNet.hdf5 --model output/dogs_vs_cats.pickle
```

#### MiniGoogLeNet on CIFAR-10
```
python bin/minigooglenet_cifar10.py --output output --model output/minigooglenet_cifar10.hdf5
```

#### Tiny ImageNet
```
wget -c http://cs231n.stanford.edu/tiny-imagenet-200.zip

cd project/deepergooglenet
# build HDF5 files
python img2hdf5.py

# train
python train_deepergooglenet.py --checkpoints output/checkpoints

# test
python evaluate_deepergooglenet.py 

```

#### Train ResNet on CIFAR-10
```
# with ctrl+c method
python bin/resnet_cifar10.py --checkpoints output/checkpoints -r 1e-1

python bin/resnet_cifar10.py --checkpoints output/checkpoints --model output/checkpoints/lr_0.1/050-0.6917.hdf5 --start-epoch 50 -r 1e-2

python bin/resnet_cifar10.py --checkpoints output/checkpoints --model output/checkpoints/lr_0.01/019-0.6052.hdf5 --start-epoch 69 -r 1e-3

# with learning rate decay
python bin/resnet_cifar10_lr_decay.py --output output --model output/resnet_cifar10.hdf5 

```