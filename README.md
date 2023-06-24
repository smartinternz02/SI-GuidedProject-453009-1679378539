**Deep Learning Techniques for Breast Cancer Risk Prediction**

Breast cancer is one of the most prevalent and potentially life-threatening diseases affecting women worldwide. Early detection and accurate diagnosis play a crucial role in improving survival rates and treatment outcomes. Deep learning techniques have shown promising results in various medical applications, including breast cancer prediction. However, there is a need to optimize these techniques to enhance their performance and reliability. 

This project aims to develop an optimized deep learning model for predicting breast cancer, leveraging advanced techniques to improve accuracy and reduce false positives/negatives. The existing methods for breast cancer prediction rely on conventional machine learning algorithms and may have limitations in terms of accuracy and efficiency.

Deep learning, particularly convolutional neural networks (CNNs), has demonstrated remarkable capabilities in image analysis tasks, making it a potential solution for breast cancer prediction. However, optimizing the deep learning models is essential to overcome challenges such as feature extraction, data imbalance, and interpretability. 

This project also focuses on developing an optimized deep learning technique that addresses these challenges and improves breast cancer prediction accuracy.

  I.We will be utilizing a data set consisting of 2,50,000 mount slide images of breast cancer specimens
  II.Clean and Sanitize the dataset .
  III.Preprocess the data to perform data partitioning and handle missing values.
  IV.Create training and testing sets.
  V.Build a classifier and fit the data to the model.
  VI.Check the accuracy of the model and measure its effectiveness.




**1.Import libraries and load data file**

Upload the dataset in colab using google drive import ( Dataset can be reimported from gdrive easily if the environment gets refreshed)

Import required libraries in python:

**2.Search for breast cancer images and assign it for processing**

breast_img = glob.glob('/tmp/breast_cancer_cnn/breastcancerdataset/IDC_regular_ps50_idx5/**/*.png', recursive = True)

The glob module, which is short for global, is a function that's used to search for files that match a specific file pattern or name.

**3.Segregating images based on ‘Negative’ and ‘Positive’ scenario**

N_IDC = []
P_IDC = []

for img in breast_img:
    if img[-5] == '0' :
        N_IDC.append(img)
    
    elif img[-5] == '1' :
        P_IDC.append(img)

**4.Data visualization using MatPlotLib**

from keras.preprocessing import image
plt.figure(figsize = (15, 15))
some_non = np.random.randint(0, len(N_IDC), 18)
some_can = np.random.randint(0, len(P_IDC), 18)
s = 0
for num in some_non:
        img = tf.keras.utils.load_img((N_IDC[num]), target_size=(100, 100))
        #img = image.load_img((N_IDC[num]), target_size=(150,150))
        img = tf.keras.utils.img_to_array(img)
        plt.subplot(6, 6, 2*s+1)
        plt.axis('off')
        plt.title('no cancer')
        plt.imshow(img.astype('uint8'))
        s += 1
s = 1
for num in some_can:
        img = tf.keras.utils.load_img((P_IDC[num]), target_size=(100, 100))
        img = tf.keras.utils.img_to_array(img)
        plt.subplot(6, 6, 2*s)
        plt.axis('off')        
        plt.title('IDC (+)')
        plt.imshow(img.astype('uint8'))
        s += 1

The "plt.figure(figsize=(15, 15))" command is used to set the size of a plot in Matplotlib, a popular Python library used for data visualization. This command is a helpful tool to customize the size of your plots and improve the visual representation of your data.

The “np.random.randint(0, len(N_IDC), 18)” is a command that generates a numpy array of 18 random integer numbers within the range of 0 and the length of the N_IDC list.
random is a submodule of NumPy that contains functions for generating random numbers.
randint is a function that generates random integers within a given range.

The” tf.keras.utils.load_img((N_IDC[num]), target_size=(100, 100))” command loads an image file at the specified path and resizes it to a fixed size, which can then be used as input to a machine learning model..

The plt.subplot() is a very useful function for creating and arranging multiple subplots within a single figure.

The plt.axis('off') is a function from the matplotlib.pyplot module that turns off the axis lines and labels in a plot.

The plt.title() function is used to add a title to a plot and it accepts a string argument that specifies the text to be displayed as the title.

The plt.imshow() function is used to display 2D arrays and images, and it accepts a numpy array or a PIL image instance as an argument. img is a numpy array that represents an image, and the .astype('uint8') method is used to convert the data type of the array to 8-bit unsigned integer format, which is a common format for image data. 

**5.Process image using OpenCV**

non_img_arr = []
can_img_arr = []

for img in N_IDC:
        
    n_img = cv2.imread(img, cv2.IMREAD_COLOR)
    n_img_size = cv2.resize(n_img, (50, 50), interpolation = cv2.INTER_LINEAR)
    non_img_arr.append([n_img_size, 0])
    
for img in P_IDC:
    c_img = cv2.imread(img, cv2.IMREAD_COLOR)
    c_img_size = cv2.resize(c_img, (50, 50), interpolation = cv2.INTER_LINEAR)
    can_img_arr.append([c_img_size, 1])

cv2.imread: This is a function from the OpenCV library that is used to read an image from a file.

cv2.IMREAD_COLOR: This is a flag that tells OpenCV to read the image in color mode (as opposed to grayscale or alpha mode).

The cv2.resize is a method used in OpenCV for resizing images. It allows you to adjust the size of an image to a specified width and height. The method can also be used to scale images up or down, and it supports multiple interpolation methods for resizing the image. 

The "append()" method is a built-in Python function that is used to add a new element to the end of a list.


**6.Slicing dataset images**

can_img_arr2=can_img_arr[0:50000]
non_img_arr2=non_img_arr[0:50000]

can_img_arr2 will contain the first 50,000 elements of can_img_arr. This slicing operation can be useful when working with large datasets, as it enables you to work with smaller subsets of data at a time.

**7.Shuffle dataset images**

X = []
y = []

breast_img_arr = np.concatenate((non_img_arr2, can_img_arr2))
random.shuffle(breast_img_arr)

for feature, label in breast_img_arr:
    X.append(feature)
    y.append(label)
    
X = np.array(X)
y = np.array(y)

The np.concatenate() function is used to join two or more arrays along a specified axis. It is concatenating the non_img_arr2 and can_img_arr2 arrays along the first (default) axis, which is the vertical axis.

The random.shuffle() is a function takes a sequence (such as a list or tuple) and shuffles its elements randomly.

**8.Visualizing the data using bar and pie chart**

import os
def describeData(a,b):
    print('Total number of images: {}'.format(len(a)))
    print('Number of IDC(-) Images: {}'.format(np.sum(b==0)))
    print('Number of IDC(+) Images: {}'.format(np.sum(b==1)))
    print('Image shape (Width, Height, Channels): {}'.format(a[0].shape))
describeData(X,y)
benign_count = np.sum(y==0)
malignant_count =  np.sum(y==1)
counts = [benign_count, malignant_count]
labels = ['Doesn\'t have Breast disease', 'Has Breast disease']
plt.pie(counts, labels=labels, autopct='%1.1f%%')
# plt.xlabel('Class')
# plt.ylabel('Count')
plt.title('Breast Cancer Image Dataset')
plt.show()
counts = [benign_count, malignant_count]
labels = ['Benign', 'Malignant']

plt.bar(labels, counts)
plt.title('Breast Cancer Image Dataset')
plt.xlabel('Class')
plt.ylabel('Count')

for i, v in enumerate(counts):
    plt.text(i, v+1000, str(v), ha='center')

plt.show()

Output:

Total number of images: 100000
Number of IDC(-) Images: 84716
Number of IDC(+) Images: 15284
Image shape (Width, Height, Channels): (50, 50, 3)

**9.Train data**

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)

from tensorflow.keras.utils import to_categorical
Y_train = to_categorical(Y_train, num_classes = 2)
Y_test = to_categorical(Y_test, num_classes = 2)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

Output: 
Training Data Shape: (70000, 50, 50, 3)
Testing Data Shape: (30000, 50, 50, 3)

from tensorflow.keras.utils import to_categorical - To import the to_categorical function from the keras.utils module in TensorFlow. This function is commonly used to convert class vectors (integers) to binary class matrix.

The "to_categorical()" is a function takes an array or a list of integers as input, and returns a matrix where each row corresponds to one of the input integers and each column corresponds to a possible value of the categorical variable.

**10.Reduce training and testing dataset for faster processing**

X_train = X_train[0:50000] 
Y_train = Y_train[0:50000] 
X_test = X_test[0:30000] 
Y_test = Y_test[0:30000] 

Above code is reducing the size of the datasets X_train, Y_train, X_test, and Y_test by keeping only the specified number of elements at the beginning of each dataset. The purpose of this code is to create smaller training and testing sets for faster processing or to work with a limited amount of data for experimentation or debugging purposes.

**11.Importing optimizers, metrics, callbacks, and functions**

from tensorflow.keras.optimizers import Adam, SGD
from keras.metrics import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import itertools


from tensorflow.keras.optimizers import Adam, SGD: This line imports the Adam and SGD optimizers from the tensorflow and are commonly used in training deep learning models to update the model's parameters during the learning process.

from keras.metrics import binary_crossentropy: This line imports the binary_crossentropy metric from the keras.metrics module and it is a common loss function used in binary classification problems.

from tensorflow.keras.callbacks import EarlyStopping: This line imports the EarlyStopping callback from the tensorflow that stops the training process early if certain criteria, such as the validation loss not improving for a specified number of epochs, are met.

from sklearn.metrics import confusion_matrix: This line imports the confusion_matrix function from the sklearn, which is used to compute the confusion matrix, which is a useful tool for evaluating the performance of a classification model.

import itertools: This line imports the itertools module which provides various functions for efficient iteration and combination of iterables.


**12.Classify images into different layers and output classes**

early_stop=EarlyStopping(monitor='val_loss',patience=5)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(50, 50, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.3))
model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(2, activation='softmax'))

The code defines a CNN model with multiple convolutional, pooling, batch normalization, dropout, and dense layers and it is designed for image classification tasks with input images of size 50x50 pixels and 3 color channels. 

The model architecture aims to extract features from the input images and classify them into one of the two output classes using softmax activation. 
The early_stop callback is also defined to monitor the validation loss and stop training if the loss doesn't improve for a certain number of epochs.

**13.Compile the defined model **

model.compile(Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

Compiles the defined model with a specified optimizer, loss function, and evaluation metrics.The model is prepared for training with the specified optimizer, loss function, and evaluation metric(s). This step is necessary before starting the actual training process.

**14.Initiate training process**

history = model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 10, batch_size = 35)

The model.fit() function starts the training process with the specified settings. The model will iterate over the training data for the specified number of epochs, optimizing the model's weights using the defined loss function and optimizer. The validation data will be used to evaluate the model's performance during training. The training progress and metrics will be stored in the history variable.

epochs = 10: This argument determines the number of times the entire training dataset will be iterated over during the training process. Here, it is set to 10, meaning the model will be trained for  ten epoch.

**15.Save the trained model**

model.save("breastcancer.h5")

This line saves the model to a file named "breastcancer.h5" in the current directory. The file format used is Hierarchical Data Format 5 (HDF5), which is a commonly used file format for storing large amounts of numerical data. By saving the model, we can later load it and use it for inference or further training without having to retrain the model from scratch.


**16.Import module for loading trained model**

from keras.models import load_model
from keras.preprocessing import image
import numpy as np

The provided code imports the necessary modules for loading a pre-trained model, working with images, and manipulating arrays

**17.Load the saved model from file**

model= load_model("/tmp/breast_cancer_cnn/breastcancer.h5")

This line loads the pre-trained model stored in the file "breastcancer.h5" located at the specified file path and assigns it to the variable model. It is used to load a saved model from a file.

**18.Load an image (class1) to test the model**

img = tf.keras.utils.load_img("/tmp/breast_cancer_cnn/breastcancerdataset/IDC_regular_ps50_idx5/8863/1/8863_idx5_x1001_y801_class1.png", target_size=(50, 50))

**19.Use the model to make prediction**

x = tf.keras.utils.img_to_array(img)
x = np.expand_dims(x,axis = 0)
pred = model.predict(x)

The provided code converts the loaded image into a NumPy array, expands its dimensions, and uses the pre-trained model to make a prediction and now we will have the prediction for the input image stored in the pred variable, which we can further process or analyze as needed.

**20.Display result (class 1) for positive scenario**

np.round(pred[0][1],0)

Output : 1

**21.Load an image (class0) to test the model**

img = tf.keras.utils.load_img("/tmp/breast_cancer_cnn/breastcancerdataset/IDC_regular_ps50_idx5/10253/0/10253_idx5_x1001_y1001_class0.png", target_size=(50, 50))

**22.Use the model to make prediction**

x = tf.keras.utils.img_to_array(img)
x = np.expand_dims(x,axis = 0)
pred = model.predict(x)

**23.Display result (class 1) for negative scenario**

np.round(pred[0][1],0)

Output: 0
