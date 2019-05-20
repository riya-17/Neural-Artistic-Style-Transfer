# Neural Artistic Style Transfer using Tensorflow 
Neural Artistic style transfer is transferring of style from the style image to the content of the content image. It uses two images as its input i.e. Style Image and Content Image, with the help of Convolutional Neural Network we are able to extract the style of the style image and content of the content image and generate an image which is a blend of both.

## [What is a Convolutional Neural Network or CNN?](https://medium.com/technologymadeeasy/the-best-explanation-of-convolutional-neural-networks-on-the-internet-fbb8b1ad5df8)
CNNs are like Neural Network made up of learnable weights and biases. It works on multi-channeled images. Convolution Layer is the main Building block of CNN and Each layer consists of a number of filters. It also consists of a pooling layer which reduces the spatial size of the representation i.e. computations. CNN helps in detecting objects in images.

## Setup
* Follow [Tensorflow Installation](https://www.tensorflow.org/install/) for installing Tensorflow in your OS.<br><br>
* ``` pip install Numpy ``` <br><br>
* ``` pip install Scipy ``` <br><br>

Install the dependencies listed in `requirements.txt`

 ` pip install -r requirements.txt `

If you already have numpy at the latest version installed globaly on your machine you need to either downgrade it to a version compatible with `Tensorflow` or (preferred) you can create a virtual environment

### Using a virtual environment

Virtualenv can be installed with pip

` pip install virtualenv `

Then create a virtualenv on this repo's root directory
```
mkdir .env
virtualenv .env
```

Start it
` source .env/bin/activate `

Install the dependencies
` pip install -r requirements.txt `

Exit the virtualenv with
` deactivate `

## Outputs

### Style Image :

![style](https://user-images.githubusercontent.com/25060937/43782404-e9905f7a-9a7c-11e8-9520-877fefba3db5.jpg)

### Content Image:

![content](https://user-images.githubusercontent.com/25060937/43782443-03226b0e-9a7d-11e8-9f69-dd6478c02716.jpg)

### Processing:

![Processing](https://user-images.githubusercontent.com/25060937/43799647-c4e1a1be-9aab-11e8-960e-400c42afec6e.png)

### Output Image :

![Output](https://user-images.githubusercontent.com/25060937/43782850-e5636a90-9a7d-11e8-8c77-d6210f93fd31.png)

## Pre-trained Imagenet model using Very-Deep Convolutional Neural Network
you can find the model on the following site with name [imagenet-vgg-verydeep-19](http://www.vlfeat.org/matconvnet/pretrained/) 

## Resources
* [Paper: A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576v2.pdf)
* [What is CNN](https://medium.com/data-science-group-iitr/artistic-style-transfer-with-convolutional-neural-network-7ce2476039fd)
* [Artistic Neural Style](http://www.subsubroutine.com/sub-subroutine/2016/11/12/painting-like-van-gogh-with-convolutional-neural-networks)
* [Basics of Neural Style Transfer](https://www.coursera.org/lecture/convolutional-neural-networks/what-is-neural-style-transfer-SA5H8)
* [Artistic Neural Style](https://harishnarayanan.org/writing/artistic-style-transfer/)
* [Github Repo](https://github.com/anishathalye/neural-style)
* [pretrained Networks](http://www.vlfeat.org/matconvnet/pretrained/)
* [Another way of Thinking](https://medium.com/artists-and-machine-intelligence/neural-artistic-style-transfer-a-comprehensive-look-f54d8649c199)
* [VGG in tensorflow](http://www.cs.toronto.edu/~frossard/post/vgg16/#files)
* [Good to read Example](http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style)
* [Neural Style Transfer using Jupyter Notebook](https://github.com/dsgiitr/Neural-Style-Transfer/blob/master/Keras_styletransfer.ipynb)
* [Artistic Style Transfer with Convolutional Neural Network](https://medium.com/data-science-group-iitr/artistic-style-transfer-with-convolutional-neural-network-7ce2476039fd)
* [Course for Understanding Deep Learning in deep](http://course.fast.ai/lessons/lesson1.html)
* [Convolutional Neural Network with TensorFlow implementation](https://medium.com/data-science-group-iitr/building-a-convolutional-neural-network-in-python-with-tensorflow-d251c3ca8117)
* [Fast Style Transfer](https://github.com/ShafeenTejani/fast-style-transfer)
* [Style Transfer in Real Time](https://shafeentejani.github.io/2017-01-03/fast-style-transfer/)
* [Prisma](https://harishnarayanan.org/writing/artistic-style-transfer/)
