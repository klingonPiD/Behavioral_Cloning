# Project 3: Behavioral Cloning
### Ajay Paidi

# Objective
The objective of this project is to implement and train a generalized model to predict steering angles based on images collected from cameras mounted in front of a virtual car. The training data needs to be generated using a simulator provided by Udacity where one gets to drive the virtual car around a training track and record images in 'training mode'. The model is then evaluated in 'autonomous mode' where the model predicts the steering angles that guide the car along the track. An alternate track has been provided as well to test how well the model generalizes to other tracks.

# File structure
- **train_steering.ipynb**: Jupyter notebook that is the main driver file. Performs data visualization and does the training.
- **model.py**: Script that contains the implementation of the model and training.
- **get_data.py**: Script that contains the implementation of the generator that is called upon during training. Also contains all the pre-processing methods.
- **augment_data.py**: Script that contains the data augmentation logic that is called upon by the generator.
- **drive.py**: Script that drives the car (slightly modified version of the original provided by Udacity)
- **model.json**: json serialized version of the model architecture
- **model.h5**: Model weights of the best performing model
- **model.jpg**: Model diagram
- **Augmentation_Calculation.pdf**: Pdf document that describes the calculation behind some of values used for augmentation.

# Approach
It was tricky to approach this project because I was not sure whether to start by generating my own data or start by fitting a model to the Udacity data. I decided on the latter approach. The reason being if I could develop a decent model using Udacity data, I can always supplement it or replace it with my own data. My approach involved the following steps

1. Perform data visualization (this is further documented in the _train_steering.ipynb_ file).
2. Perform data pre-processing to balance the data (this is further documented in the _train_steering.ipynb_ file).
3. Write a generator function that could be used to train a keras model (more on this below).
4. Use the comma.ai model (see references) on the udacity data : I had very little success doing this. Since this model uses a lot of parameters, my training times were high which limited my ability to experiment different approaches.
5. Use the NVIDIA model: I had moderate success using the NVIDIA model on the Udacity data. Since this model uses a lot fewer parameters, I was able to quickly iterate upon different approaches to improve the model.
6. Perform data augmentation and repeat training using the NVIDIA model.
7. Generate own data and train on the refined/modified NVIDIA model.
My final model was trained upon data that was completely generate by me. I used the Udacity data as a validation set for evaluation.   

# Description of the model
My model architecture was based on the well established NVIDIA end to end driving model architecture https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/. 
The model is a convolutional neural network with 1 input normalization layer, 5 convolutional layers, and 3 fully connected layers. The first 3 convolutional layers use a width and height stride of 2 (thereby downsampling the image) while the last 2 use a stride of 1.

![Model](./Model.jpg?raw=true)

The main differences between the NVIDIA model architecture and my architecture are 

1. I use input images of size 64 x 64. This significantly reduces training time. This also reduces the number of training params to about 150,000 (compared to 250,000 in the NVIDIA model).
2. I use convolutional kernels of size 3x3 instead of 5x5.
3. I use HSV inputs instead of YCbCr inputs.
4. I make extensive use of dropouts. I use a dropout layer with a keep_prob of 0.5 after every layer. The idea is to prevent overfitting and help the model generalize well to other tracks.
5. I use the ELU (Exponential Linear Unit) activation function (not sure what NVIDIA used in their model). I did not see any significant difference between the use of RELU or ELU. I stuck with ELU because in theory ELU supposedly provides faster convergence by keeping mean activations close to zero (unlike RELU where mean activation is always positive). In addition the model by comma.ai also makes extensive use of ELU https://github.com/commaai/research/blob/master/train_steering_model.py.
6. I use the 'he_normal' weights initialization for all the layers (again I am not sure what NVIDIA uses in their model). The 'he_normal' initialization scheme initializes the weights from a random normal distribution and scales down the initialized values based on the size of the input layer. The main advantage is that by keeping the initialized values small, it helps the optimizer achieve faster convergence.

# Generator
One important aspect of this project was writing a generator function that could read, process and feed data in small batches to the model during training. This is important because the other option is to keep all the image data in memory (all 60000 images which is approximately 10gb of run time memory). Obviously this is not a practical or scaleable approach. My generator function takes in a pandas dataframe (which contains the filenames) and a batch size. It then loops indefinitely generating batches of training data. The generator shuffles the input data at the end of every cycle (i.e after one full run through all the data samples).I also hooked up the data augmentation pipeline to the generator. This makes the generator super powerful in the sense that it not only reads and processes data in small batches on the fly but also augments the data on the fly as well. This helps avoid having to generate and store all the augmented data before training.

				def aug_data_generator(df, batch_size):
					df_shuffle = shuffle(df)
					df_len = len(df)
					batch_idx = 0
					while True:
						X, y = [], []
						for num_item in range(batch_idx, batch_idx + batch_size):
							im, st_angle = apply_random_augmentation(df_shuffle.iloc[num_item], False)
							X.append(im)
							y.append(st_angle)
						X_train = np.array(X).astype(np.float32)
						y_train = np.array(y).astype(np.float64)
						batch_idx += batch_size
						if (batch_idx + batch_size) > df_len:
							batch_idx = 0
							df_shuffle = shuffle(df)
						yield X_train, y_train
		
A validation generator ('valid_data_generator' in get_data.py) was written along similar lines (without the augmentation) to perform model evaluation in small batches.

# Results
The model was trained over 12 epochs (with 25600 samples per epoch). The weights were saved for each epoch and independently tested on both the tracks. Although epoch 7 had the lowest validation loss, the weights corresponding to epoch 9 seemed to produce better results on track 1 and track2. It was quite astonishing to see the weights generalize fairly well onto track 2. The car does get stuck on some sharp bends on track 2. I believe that with more training and data augmentation, track 2 can be surmounted as well.

[![Track 1](https://img.youtube.com/vi/POhfBo7z414/0.jpg)](https://youtu.be/POhfBo7z414)

[![Track 2](https://img.youtube.com/vi/W_Kz2pVlh2U/0.jpg)](https://youtu.be/W_Kz2pVlh2U)


# References
- NVIDIA end to end paper. https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
- Comma.ai sample code. https://github.com/commaai/research
- Blog post by Vivek Yadav. https://chatbotslife.com/learning-human-driving-behavior-using-nvidias-neural-network-model-and-image-augmentation-80399360efee#.kqhedxptk
- Discussions on slack forums and multiple blog posts by super enthusiastic Udacity students!