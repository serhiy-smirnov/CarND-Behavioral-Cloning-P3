import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout

# Configuration parameters for the model training

# specify if the model summary must be shown
show_model_summary = False
# specify if the chart of training and validation loss must be shown after model training is done
show_chart = True
# specify whether dataset must be augmented by flipping images horizontally
flip_images = True
# number of epochs to train the model for
epoch_number = 5
# steering shift for the side cameras
correction = 0.2

# location of the driving data
data_path_prefix = '/opt/carnd_p3/data/'
#data_path_prefix = 'data/'



# dataset of images and steering measurements
images = []
steering_measurements = []


# load an image and add it into the dataset along with the corresponding steering measurement
def load_measurements(file_name, steering_measurement):
    # substitute the file name to the actual local path and read in the image
    image_bgr = cv2.imread(data_path_prefix + 'IMG/' + file_name)
    # Convert from BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) 
    # add the image and steering measurement to the dataset
    images.append(image_rgb)
    steering_measurements.append(steering_measurement)

# read in CSV file with the list of images and steering measurements
lines = []
with open(data_path_prefix + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    lines_count = 0
    for line in reader:
        if lines_count != 0:
            lines.append(line)
        lines_count += 1


# fill in the dataset with images and corresponding steering measurements
for line in lines:

    #original steering measurement
    steering_measurement = float(line[3])

    # add center image and original steering measurement to the dataset
    load_measurements(line[0].split('/')[-1], steering_measurement)

    # use images from left and right cameras and create adjusted steering measurements for the side camera images
    
    # add left image and corrected steering measurement to the dataset
    load_measurements(line[1].split('/')[-1], steering_measurement + correction)

    # add right image and corrected steering measurement to the dataset
    load_measurements(line[2].split('/')[-1], steering_measurement - correction)


# augment data by flipping image horizontally and negating steering measurement, if requested
if flip_images:
    images_count = len(images)
    print("Number of images before flipping:", images_count)
    for i in range(images_count):
        images.append(cv2.flip(images[i], 1))
        new_measurement = steering_measurements[i]*-1.0
        steering_measurements.append(steering_measurements[i]*-1.0)

    print("Number of images after flipping:", len(images))


x_train = np.array(images)
y_train = np.array(steering_measurements)


# build network model (three options)

# create very simple testing network
def create_model_simple():
    
    print('Creating simple model')
    
    # create sequential model
    model = Sequential()
    
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(1))
    
    return model

# create model with LeNet architecture
def create_model_lenet():
    
    print('Creating model using LeNet architecture')
    
    # create sequential model
    model = Sequential()
    
    # preprocess the data - normalize and center mean
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    # Crop the pictures to keep only the road ahead - no hood, no sky and trees
    model.add(Cropping2D(cropping=((70,25),(0,0))))


    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dropout(rate=0.75))
    model.add(Dense(84))
    model.add(Dense(1))
    
    return model

# create model with NVidia architecture
def create_model_nvidia():
    
    print('Creating model using NVidia architecture')
    
    # create sequential model
    model = Sequential()
    
    # crop the pictures to keep only the road ahead - no hood, no sky and trees
    model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))
    
    # preprocess the data - normalize and center mean
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    # create convolutional layers
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    
    # create fully connected layer
    model.add(Dense(100))
    
    # use dropoup for regulatization
    model.add(Dropout(rate=0.75))
    
    # create more fully connected layers
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    
    return model

# choose which model to use

#model = create_model_simple()
#model = create_model_lenet()
model = create_model_nvidia()

# use mean square error for the loss function (since it's a regression network) and adam optimizer
model.compile(loss='mse', optimizer='adam')

# train the network
history_object = model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=epoch_number)

# show model structure, if requested
if(show_model_summary):
    model.summary()

# save results for later use with drive.py
model.save('model.h5')

# show chart with the training and validation loss, if requested
if show_chart:
    import matplotlib.pyplot as plt

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
