# Image Captioning With Pytorch

![image](https://github.com/user-attachments/assets/c5275789-0bbe-4896-804b-26807d5e6870)

Image captioning using CNN feature extractor and a RNN decoder.

# Overview

This project is the implementation of the "Show and Tell: A Neural Image Caption Generator" paper which is available by the link:
https://arxiv.org/abs/1411.4555

The model is trained with the Flickr8k dataset which consists of ~40k image-caption pairs.

Here is an example of a image-caption from the Flickr8k dataset:
![image](https://github.com/user-attachments/assets/e0c3e1b7-5445-470d-9be0-8d940242c4e4)

a blue and grey race car driving on a dirt track

# Model

The model uses a pretrained ResNet50 encoder and a LSTM decoder for caption generation.
The CNN extracts the features from the image and passes them as the first input to the LSTM decoder.

![image](https://github.com/user-attachments/assets/f8b9e90a-d9a5-409b-a17e-f1c3e2e316b4)

# Result

After training for 10000 steps the model is able to generalize on the data very well and can provide meaningful captions.

Here is an example of the caption from the model. (This is a random dog image found on the Internet)
![image](https://github.com/user-attachments/assets/d1f2dc0b-b2c1-4cac-8a0a-1c5b8a31dd2b)

However the model, due its architecture, can not describe very complex scenes properly and usually mentions only small parts of the image. (Which is still awesome)

# How to use

In order to train the model you can configure the hyperparameters in the train.py and then run the following command in the folder with the file:

````
python train.py
````

Once you have a trained model you can deploy it using streamlit. Make sure you check the model that is being loaded in the app.py file.

````
python run.py
````


