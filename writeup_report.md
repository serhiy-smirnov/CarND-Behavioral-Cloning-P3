# **Behavioral Cloning** 


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model.h5.png "Model Visualization"
[image2]: ./images/source_image_center.jpg "Center camera image"
[image3]: ./images/source_image_left.jpg "Left camera image"
[image4]: ./images/source_image_right.jpg "Right camera image"
[image5]: ./images/source_image_original.jpg "Original image"
[image6]: ./images/source_image_flipped.jpg "Flipped image"
[image7]: ./images/source_image_cropped.jpg "Cropped image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create, train and save the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolutional neural network 
* writeup_report.md summarizing the results
* video.mp4 (a video recording of my vehicle driving autonomously around the track for one full lap)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolutional neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try some of the well known and proven powerful networks that took part in the ImageNet challenge and adapt to the specifics of this project.

But first of all I created a very basic network consisting of a single fully connected layer just to verify the whole processing pipeline. It can be found in the function 'create_model_simple()'.

Then I have implemented a network based on the LeNet architecture with a difference in the  output layer to produce a single value for the steering angle instead of a classification into 10 digits. It can be found in the function 'create_model_lenet()'.
This network consists of two convolutional layers each followed by subsampling layer and then three fully connnected layers.

As a next step I've implemented more powerful network based on Nvidia architecture, which consists of 5 convolutional layers and 4 fully connected layers.
https://developer.nvidia.com/blog/deep-learning-self-driving-cars/

The model consists of a convolutional neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py function 'create_model_nvidia()') and includes RELU layers to introduce nonlinearity.

For the data pre-processing step I used data normalization and mean centering using Keras Lambda layer as a first layer in the model.

#### 3. Training data & Training Process

I captured a small dataset using provided simulator to test the processing pipeline. But capturing of a bigger dataset with smooth steering was not possible due to high latency of my connection to the workspace.
Therefore I decided to use the pre-recorded dataset provided in the workspace.
It consists of 8036 images for each camera (center/left/right), 24108 images in total.

Example of an image from the center camera:

![alt text][image2]

Example of an image from the left camera:

![alt text][image3]

Example of an image from the right camera:

![alt text][image4]

Since I was not able to record recovery scenarious, I used left and right camera images to simulate recovery from 'more to the left' and 'more to the right' situations with steering cofficient of 0.2. I tried varying this coefficient but getting it into the bigger values resulted in too coarse steering actions, while reducing it below 0.2 didn't produce enough effect to get the car on track in some tricky curves, like the one after the bridge and the turn without lane markings on the right.

Without a possibility to record additional training data, it was especially important to augment the data set to produce more training data and to make it more balanced. To achieve this I flipped the images horizontally and accordingly inverted the steering angles, which effectively doubled the size of the dataset for training to 48216 samples.

Example of an original and corresponding flipped image

![alt text][image5]
![alt text][image6]


As an additional processing step I cropped the images to let the network focus only on important parts of the scene. I removed 70 pixels from the top containing sky, trees etc and 25 pixels from the bottom containing hood of the car and no information about lane markings or lane borders.

Example of a cropped image

![alt text][image7]

I implemented cropping as a Keras Cropping2D layer, put before convolutional layers.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.
I used mean square error for the loss function (since it's a regression network) and adam optimizer.
The ideal number of epochs with my model based on the Nvidia architecture was 3 to 5. After this point the mean squared error was not improving but was rather getting even bigger.

#### 2. Attempts to reduce overfitting in the model

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set (which was 80% of the entire data set) but a high mean squared error on the validation set (which was the remaining 20% of the entire data set). This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding dropout layer after the first fullly connected layer with initial rate parameter set to 0.25, meaning that 25% of inputs were dropped.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.
The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle touched the road boundary or fell off the track. To improve the driving behavior in these cases, I experimented with the position of the dropout layer and it's drop rate parameter as well as the angle correction parameter for the left and right camera images. The resulting parameters that allowed the vehicle to stay on track were 0,2 for the correction angle and pretty aggressive dropout rate of 0.75. I believe that such a dropout rate was necessary to compensate for the small dataset.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road and touching the lane markings/road boundaries.

#### 2. Final Model Architecture

The final model architecture (model.py 'create_model_nvidia()') consists of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 320x160x3 RGB image   							| 
| Cropping         		| Crops 70 px from the top and 20 px from the bottom of the image, outputs 65x320x3 | 
| Lambda         		| Input data normalization and mean centering | 
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 31x158x24, with relu activation 	|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 14x77x36, with relu activation 	|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 5x37x48, with relu activation 	|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 3x35x64, with relu activation   |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 1x33x64, with relu activation   |
| Flatten	      	    | outputs 2112                				    |
| Fully connected		| outputs 100  									|
| Dropout				| with rate of 0.75 for training        |
| Fully connected		| outputs 50  									|
| Fully connected		| outputs 10  				        |
| Fully connected		| outputs 1  				        |
|						|												|

Here is a visualization of the architecture (created with https://netron.app/)

![alt text][image1]
