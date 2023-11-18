# Powdery Mildew Detection in Cherry Leaves

### Deployed app: [Here](https://mildew-detection-in-cherry01-b61ac18ef016.herokuapp.com/)

## Dataset Content

The dataset contains 4208 featured photos of single cherry leaves against a neutral background. The images are taken from the client's crop fields and show leaves that are either healthy or infested by powdery mildew a biotrophic fungus. This disease affects many plant species but the client is particularly concerned about their cherry plantation crop since bitter cherries are their flagship product. The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves).

## Business Requirements

The client requested to develop a Machine Learning based system to detect whether a certain cherry tree presents powdery mildew so it can be treated with a fungicide.
The requested system should be capable of detecting infection on a tree by using a tree leaf image,  to determine if it needs attention.
The system was requested by the Farmy & Food company to automate the detection process conducted manually thus far. The company has thousands of cherry trees, located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.

In sumary:

1. The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one infected by powdery mildew.
2. The client is interested in predicting if a cherry tree is healthy or contains powdery mildew.

## Hypothesis and validation

**Hypothesis**

As the season progresses and infection is spread by wind, leaves may become distorted, curling upward. Severe infections may cause leaves to pucker and twist. Newly developed leaves on new shoots become progressively smaller, are often pale and may be distorted.
Cherry leaves affected by powdery mildew have clear marks, typically the first symptom is a light-green, circular lesion on either leaf surface, then a subtle white cotton-like growth develops in the infected area.
This visual differences can be translated in ML terms, by preparing images before being fed to the model we can optimize feature extraction and training.

When we are dealing with an Image dataset, it's important to normalize the images in the dataset before training a Neural Network on it. This is required because of the following two core reasons:

- It helps the trained Neural Network give consistent results for new test images.
- Helps in Transfer Learning
To normalize an image, one will need the mean and standard deviation of the entire dataset.

To calculate the **mean** and **standard deviation**, the mathematical formula takes into consideration four dimensions of an image (B, C, H, W) where:

- B is batch size that is number of images
- C is the number of channels in the image which will be 3 for RGB images.
- H is the height of each image
- W is the width of each image
Mean and std is calculated separately for each channel. The challenge is that we cannot load the entire dataset into memory to calculate these paramters. We can load a small set of images (batches) one by one and this can make the computation of mean and std non-trivial.

**Image Montage**

An Image Montage shows the differences between a healthy leaf and an infected one.

![montage_healthy]()
![montage_infected]()

Difference between average and variability images shows that affected leaves present more white stripes in the center.

![average variability between samples]()

While image difference between average infected and average infected leaves shows no intuitive difference.

![average variability between samples]()

**3. Conclusion**

The model was able to detect such differences and learn how to differentiate and generalize in order to make accurate predictions.

---

### Model Compilation

The model is a Convolutional Neural Network designed for image classification.

**Input Layer:**

- Convolutional layer with 32 filters, each of size (3, 3).
- Activation function: ReLU (Rectified Linear Unit).
- MaxPooling layer with a pool size of (2, 2).
- This layer is responsible for extracting features from the input image.

**Convolutional Layers:**

- Two additional sets of convolutional layers, each followed by a MaxPooling layer.
- The second convolutional layer has 32 filters, and the third has 64 filters.
- Activation function: ReLU.
Each set helps in capturing more complex patterns and features from the input.

**Flatten Layer:**

- Flattens the output from the previous layers into a one-dimensional vector.
- Prepares the data for the fully connected layers.

**Fully Connected Layer:**

- Dense layer with 64 neurons (units).
- Activation function: ReLU.
- Introduces non-linearity to the model and performs high-level reasoning on the features extracted by the convolutional layers.
- Dropout layer with a dropout rate of 0.2, which helps prevent overfitting by randomly setting a fraction of input units to zero during training

**Output Layer:**

- Dense layer with 2 neurons, representing the classes (e.g., Healthy and Infected).
- Activation function: Softmax, used for multi-class classification.
- Produces probability distributions over the possible classes.

**Compilation:**

- Categorical crossentropy loss function, suitable for multi-class classification problems.
- Adam optimizer, which adapts the learning rates during training.
- Evaluation metric: Accuracy.

## Trial and error

the process that lead to the current hyperparameters settings and model architecture is trial and error, changing one parameter at a time.

