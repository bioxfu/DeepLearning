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
