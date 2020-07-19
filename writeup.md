# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./visual/Train_Dataset.png "Train Data Set"
[image2]: ./visual/Valid_Dataset.png "Validation Data Set"
[image3]: ./visual/Test_Dataset.png "Test Data Set"
[image4]: ./visual/signs_gray_color.jpg "Colored and gray signs"
[image5]: ./visual/jittered.jpg "Jittered Data sets"
[image6]: ./visual/test1.jpg "German Sign 1"
[image7]: ./visual/test2.jpg "German Sign 2"
[image8]: ./visual/test3.jpg "German Sign 3"
[image9]: ./visual/test4.jpg "German Sign 4"
[image10]: ./visual/test5.jpg "German Sign 5"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3 pixels. RGB color space.
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.
#### Statistics about the data sets
Following figures show the amount of different classes (traffic signs) and how many samples each class contains.

![alt text][image1] ![alt text][image2] 
![alt text][image3]  				 
 
As one can see, not all classes are represented eaqually. It may happen, that classes will a small number of examples are trained less good.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it decreases the amount of input channels respectivly data without loosing too much information. The recognition of traffic signs doesn't depend on the color information. Each sign is recognisable by only its shape and imprinted graphics..

Here are random examples of a traffic sign images before and after grayscaling.

![alt text][image4]

Additionally, I normalized the image data because data should have mean zero and equal variance for better performance.

#### Generating jittered data
As suggested in LeCun's paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks" one way to increase validation accuracy of a CNN is to generate more training data. As shown above, some sign classes are underrepresented. By adding jittered images (Applying warp function and perspective transformation) the training set is increased significantly from 34799 images to 89860 images. The validation set is increased from 4410 images to 22465 images.

![alt text][image5]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray scale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 1x1x400    |
| Fully connected		| input 400, output 200       				    |
| RELU			  	    |           									|
| Droput                | 50 %                                          |
| Fully connected		| input 200, output 84       				    |
| RELU				    |           									|
| Droput                | 50 %                                          |
| Fully connected		| input 84, output 43       				    |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the ADAM Optimizer. Batchs size was set to 150 and 20 epochs. The learning rate was set to 0.0009.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training set accuracy of 99.9%
* Validation set accuracy of 99.6% 
* Test set accuracy of 94%

An iterative approach was chosen:
At first I implemented the LeNet architecture which was shown in the LeNet Lab. I used the RBG images as data sets for training. With this approach the model realized ~85% validation accuracy. 

To improve it, RGB to Gray function and the normalisation step were added. With this improvements the accuracy increased to about 92%. I also adjusted the drop out hyperparameter and tried the softmax function instead of Relu function for the activation step. This didn't have a big impact on the result.
Adding one more convolutional layer to the LeNet architecture and raising the epochs to 20 was increasing the validation accuracy up to 99%.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Following 5 german traffic signs were choosen to test real data:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

The second image (Priority Road) may yield a bad prediction since it is very dark foreground with low contrast and a bright background. This could lead to misclassification. The third image (No entry) is quite blur which could yield to wrong predictions. The 4th image (Right-of-way at next intersection) could be difficult to classify because there is half of another sign shown under the actual traffic sign.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h        		| No Passing  									| 
| Priority Road  		| Priority Road  										|
| No entry				| No entry											|
| Right-of-way at next intersection| Right-of-way at next intersection				 				|
| Stop      			| Stop      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This fits to the results of the Validation set (99.6%) and Test set accuracy (99.9%).

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

The top five soft max probabilities of all 5 images are 100 %, 0% ,0% ,0% ,0%. The model is very sure about it's predictions.
