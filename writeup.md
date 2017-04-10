# Traffic Sign Recognition

## Writeup Template

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/original.png "Original"
[image3]: ./examples/grayscale.png "Grayscale"
[image4]: ./examples/random_noise.jpg "Random Noise"
[image5]: ./examples/test1.jpg "Traffic Sign 1"
[image6]: ./examples/test2.jpg "Traffic Sign 2"
[image7]: ./examples/test3.jpg "Traffic Sign 3"
[image8]: ./examples/test4.jpg "Traffic Sign 4"
[image9]: ./examples/test5.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/mkoehnke/CarND-TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how all images classes and the number of images.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because looking at the 43 traffic sign images, color doesn't provide critical information and can be ignored. An additional benefit is that the neural network is potentially faster and focuses on features that are more important.

Here is an example of a traffic sign images before and after grayscaling.

![alt text][image2]

![alt text][image3]

As a last step, I normalized the image data because this helps to scale down the disparity within the data. I choose to normalize to [0.1 0.9] because it's commonly used in neural networks. 

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I didn't have to split the training data because there was already a validation set provided with `valid.p`.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the fifth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         			| 32x32x1 Grayscale image | 
| Convolution     		| 1x1 stride, valid padding, outputs 28x28x6 |
| RELU						||
| Max pooling	      		| 2x2 stride, valid padding,  outputs 14x14x6 |
| Convolution	    		| 1x1 stride, valid padding, outputs 10x10x16 |
| RELU						||
| Max pooling				| 2x2 stride, valid padding,  outputs 5x5x16 |
| Flatten					||
| Fully connected		| Input = 400. Output = 200. |
| RELU						||
| Dropout					| Probability: 0.75 | 
| Fully connected		| Input = 200. Output = 100. |
| RELU						||
| Dropout					| Probability: 0.75 |
| Fully connected		| Input = 100. Output = 43. | 
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the sixth cell of the ipython notebook. 

As a starting point, I used the basic LeNet architecture (provided by Udacity) with the default parameter settings. To improve the validation accuracy to at least 93% I added a dropout regularization with a probability of 75%. I also increased the batch size from 128 to 145 and set the learning rate from 0.001 to 0.004. I left the number of epochs untouched.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the sixth/seventh cell of the Ipython notebook.

My final model results were:

* validation set accuracy of 93%
* test set accuracy of 91%

As mentioned above, I used the provided LeNet architecture as a starting point. As this is an image classification problem, I think LeNet is a good choice because it uses convolutional layers which reduce computation (compared to full-connected layers). 

Tweaking the hyper-parameters and adding a dropout regularization resulted in an accuracy of over 93%. In addition, all test images were classified correctly (see below), which can be interpreted as evidence that the model is working well.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

##### Right-of-way at the next intersection
- might be difficult to classify because it's pretty similar to the e.g. "General Caution" sign
- another issue might be the resolution of the images

##### General caution
- similarity to other signs
- sky is overcasted, which might result in a better classification

##### Yield
- sign has a pretty unique shape and no pictogram which should result in a correct classification

##### Stop Sign
- text in combination with a low resolution + not having a plain background might result in wrong classification

##### Ahead only
- might be difficult to classify because signs like "Turn right ahead", "Turn left ahead", "Go straight or right", "Go straight or left", "Keep right" and "Keep left" are quite similar

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection      		| Right-of-way at the next intersection   									| 
| General caution    			| General caution									|
| Yield					| Yield											|
| Stop Sign	      		| Stop Sign					 				|
| Ahead only				| Ahead only      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model predicts the sign **Right-of-way at the next intersection** correctly. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.41         			| Right-of-way at the next intersection 									| 
| 0.19     				| Beware of ice/snow										|
| 0.09					| Pedestrians											|
| 0.05	      			| Double curve				 				|
| 0.002				    | General caution     							|


For the second image, the model predicts the sign **General caution** correctly. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.46         			| General caution 									| 
| 0.18     				| Traffic signals										|
| 0.12				| Pedestrians											|
| -0.02	      			| Right-of-way at the next intersection				 				|
| -0.11				    | Roundabout mandatory      							|

For the third image, the model predicts the sign **Yield** correctly. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.38         			| Yield 									| 
| 0.16     				| Keep left										|
| 0.13					| No vehicles											|
| 0.05	      			| Road work				 				|
| -0.12				    | Priority road      							|

For the fourth image, the model predicts the sign **Stop** correctly. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.06         			| Stop 									| 
| 0.05     				| Speed limit (30km/h)										|
| 0.03					| Speed limit (50km/h)											|
| 0.02	      			| Keep right				 				|
| 0.007				    | Yield    							|

For the fifth image, the model predicts the sign **Ahead only** correctly. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.28         			| Ahead only 									| 
| 0.03     				| Keep left										|
| -0.03					| Turn right ahead											|
| -0.03	      			| Bicycles crossing				 				|
| -0.04				    | Road work     							|