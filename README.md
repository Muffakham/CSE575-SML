# CSE575-SML Project 

This project is a visual recommeder system, that intends to take an image from the user and recommend 5 other images of the similar sort to the user.

The application of this is primarily in e commerce systems where a user needs to be recommended similar products 

Here we are using pytorch to make use of the available pretrained models like REsNet and alexNet. We are using the AlexNet pretrained model gor our tasks of classification. The model has been trained on ImageNet dataset which consists of over 14 million images. 

Our Key concept is to 
	- take the input image from the user 
	- classify the image using the model
	- Now we create a dataset of about 10 images(initially, we plan to have	about 100 -200 images for the purpose of easy training on local PC's
	- each of the image in dataset is then classified using the model
	- then we calculate the difference between the input image vector and this image vector
	- then we square the values
	- finally we sort the list of vectors in ascending order and obtain
	- then display the top five picks

