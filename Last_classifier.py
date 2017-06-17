"""
Created on Mon May 15 12:43:56 2017

@author: WangZ Bianca
"""

from keras import applications
from resnet50 import ResNet50
from keras.layers import Flatten, Dense, Input,Dropout,BatchNormalization
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras import backend as K
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model
from keras import callbacks, metrics

# Visualization imports
import numpy as np
from matplotlib import pyplot as plt
from vis.utils import utils
from vis.visualization import visualize_activation, get_num_filters
from keras.callbacks import TensorBoard
from keras import callbacks
from sklearn.metrics import classification_report, confusion_matrix



# build the ResNet50 network
model = ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(200,200,3)))
print('Model loaded.')

# ------------------Vizualize features--------------------------------------------------

#layer_name = 'conv1'
#layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

# Visualize all filters in this layer.
#filters = np.arange(get_num_filters(model.layers[layer_idx]))
#print(len(filters))
#filters=filters[1:20]
# Generate input image for each filter. Here `text` field is used to overlay `filter_value` on top of the image.
#vis_images = []
#for idx in filters:
    #img = visualize_activation(model, layer_idx, filter_indices=idx)
    #img = utils.draw_text(img, str(idx))
    #vis_images.append(img)

# Generate stitched image palette with 8 cols.
#stitched = utils.stitch_images(vis_images, cols=8)
#plt.axis('off')
#plt.imshow(stitched)
#plt.title(layer_name)
#plt.show()

#-------------------------------------------------------


#'rain or not/train'
train_data_dir ='DATA/train'
validation_data_dir = 'DATA/validation'
test_data_dir='DATA/test'
nb_train_samples = 6740

# Dimensions of our images.
img_width, img_height = 200, 200
nb_validation_samples = 1188
nb_test_samples = 1984

# parameters
epochs = 6
batch_size = 20
class_weight = {0 : 4.,1: 1.}


# Build top layer
x = model.output
#print(model.summary())
x = Flatten(input_shape=model.output_shape[1:], name='flatten')(x)
x = BatchNormalization(axis=1, name='batch_norm')(x)
x = Dense(256, activation='relu',name='fc1')(x)
x = (Dropout(0.5))(x)
x = Dense(2, activation='softmax', name='predictions')(x)

# Add top layer
final_model = Model(inputs=model.input, outputs=x)

#==============================================================================
# # note that it is necessary to start with a fully-trained
# # classifier, including the top classifier,
# # in order to successfully do fine-tuning
#top_model_weights_path = 'fc_model.h5'
#top_model.load_weights(top_model_weights_path)
# # add the model on top of the convolutional base
#model.add(top_model)
#==============================================================================

# set the first 105 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
print(len(final_model.layers))
for layer in final_model.layers[:173]:
    layer.trainable = False
print(final_model.summary())

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
final_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy','categorical_accuracy'])
print('model compiled')

#==============================================================================
# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# train generator and validation
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    classes = ['0','1'])

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

# Create callbacks objects
filepath = "./Best_model_0_1/best.{epoch:02d}-{val_acc:.2f}.hdf5"
tboard=callbacks.TensorBoard(log_dir='./Graph_0_1', histogram_freq=1,write_graph=True, write_images=False, embeddings_freq=1,embeddings_layer_names='predictions')
#earlyStop=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
best_model=callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

# Fine-tune the model
final_model.fit_generator(
    train_generator,
    steps_per_epoch= nb_train_samples // batch_size,
    epochs= epochs,
    validation_data= validation_generator,
    validation_steps= nb_validation_samples // batch_size,
    callbacks=[tboard,best_model],
    pickle_safe = True,
    workers=1)
    #class_weight=class_weight)


#======================Testing============================
# score = final_model.evaluate_generator(test_generator, nb_test_samples//batch_size, workers=1,pickle_safe = True)
# scores = final_model.predict_generator(test_generator, nb_test_samples//batch_size, workers=1,pickle_safe = True)
# 
# 
# correct = 0
# for i, n in enumerate(test_generator.filenames):
#     if i < len(scores):
#     	if n.startswith("0") and scores[i][0] >= 0.5:
#         	correct += 1
#     	if n.startswith("1") and scores[i][1] > 0.5:
#         	correct += 1
# 
# print("Correct:", correct, " Total: ", len(test_generator.filenames))
# print("Loss: ", score[0], "Accuracy: ", score[1])
# 
# print(score)
# print("------------------------------")
# print(scores)
#==============================================================================
