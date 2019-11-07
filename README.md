# CSE575-SML Project 

This project is a visual recommeder system, that intends to take an image from the user and recommend 5 other images of the similar sort to the user.

The application of this is primarily in e commerce systems where a user needs to be recommended similar products 

Here we are using pytorch to make use of the available pretrained models like REsNet and alexNet. We are using the AlexNet pretrained model gor our tasks of classification. The model has been trained on ImageNet dataset which consists of over 14 million images. 

## Our key Concept

* take the input image from the user 
* classify the image using the model
* Now we create a dataset of about 10 images(initially, we plan to have	about 100 -200 images for the purpose of easy training on local PC's).
* each of the image in dataset is then classified using the model
* then we calculate the difference between the input image vector and this image vector
* then we square the values
* finally we sort the list of vectors in ascending order and obtain
* then display the top five picks

## Getting Started

The basic concept here is to take an image and transform it using the transforations available in the tensorflow, Each image is converted into a tensor (vector), a tensor in pytorch is same as that of an array, there can be n dimensional array. Our image is tramsformed into 256X256 tensor(array).This tensor is then passed to the model to evaluate and the out putput obtained is 1000 element long tensor, ecavh value representing the affinity of the current image towards 1000 available image classes. 

before getting started, we need to install pytorch and numpy 
### Prerequisites
```
we also need to be running python 3 
```
```
pip3 install numpy 
```
```
pip3 install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```
