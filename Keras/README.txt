#### Install CUDA and cuDNN
see [NVIDIA/README](../NVIDIA/README.md)

#### Install Bazel build tool
```
wget https://github.com/bazelbuild/bazel/releases/download/0.17.2/bazel_0.17.2-linux-x86_64.deb
sudo dpkg -i bazel_0.17.2-linux-x86_64.deb
```

#### Download and build TensorFlow
```
pip install keras_applications keras_preprocessing
wget https://github.com/tensorflow/tensorflow/archive/v1.12.0-rc0.tar.gz -O tensorflow_v1.12.0-rc0.tar.gz
tar zxf tensorflow_v1.12.0-rc0.tar.gz
cd tensorflow-1.12.0-rc0/
./configure 
bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
```

#### Install TensorFlow and Keras
```
bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/tensorflow/1.12/bin/
virtualenv ~/Python/keras
source ~/Python/keras/bin/activate
pip install ~/tensorflow/1.12/bin/tensorflow-1.12.0rc0-cp27-cp27mu-linux_x86_64.whl
pip install h5py graphviz pydot keras
```

