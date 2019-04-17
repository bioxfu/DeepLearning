from cnn import FCHeadNet
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.applications import VGG16, VGG19, InceptionV3, Xception, ResNet50
from keras.layers import Input
from keras.models import Model, load_model
import keras.backend as K
from callback import TrainingMonitor
from keras.callbacks import ModelCheckpoint
import os

def finetune_shallow(modelName, trainGen, valGen, batch_size, output_path, saved_model, learning_rate, epcho):
    
    if modelName == 'VGG16':
        baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    elif modelName == 'VGG19':
        baseModel = VGG19(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    elif modelName == 'ResNet50':
        baseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    elif modelName == 'InceptionV3':
        baseModel = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(299, 299, 3)))
        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 249 layers and unfreeze the rest:
    elif modelName == 'Xception':
        baseModel = Xception(weights="imagenet", include_top=False, input_tensor=Input(shape=(299, 299, 3)))

    # initialilze the new head of the network, a set of FC layers followed by a softmax classifier
    headModel = FCHeadNet.build(modelName, baseModel, trainGen.classes, 256)

    # place the head FC model on top of the base model -- this will become the actual model we will train
    model = Model(inputs=baseModel.input, output=headModel)

    # loop over all layers in the base model and freeze them so they will not be updated during the training process
    for layer in baseModel.layers:
        layer.trainable = False

    # compile our model
    print("[INFO]: Compiling model....")
    optimizer = RMSprop(lr=learning_rate)
    #optimizer = SGD(lr=learning_rate, decay=learning_rate/batch_size, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # construct the set of callbacks
    figure_path = os.path.sep.join([output_path, '{}_{}.png'.format(modelName, os.getpid())])
    json_path = os.path.sep.join([output_path, "{}_{}.json".format(modelName, os.getpid())])
    monitor = TrainingMonitor(figure_path, json_path)
    checkpoint = ModelCheckpoint(saved_model, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
    callbacks = [monitor, checkpoint]
    
    # train the head of the network for a few epoches (all other laysers are frozen) -- this will allow the new FC layers to 
    # start to become initialize with actual "learned" values versus pure random 
    print("[INFO]: Training head....")
    H = model.fit_generator(trainGen.generator(), 
        validation_data=valGen.generator(), 
        steps_per_epoch=trainGen.numImages//batch_size,
        validation_steps=valGen.numImages//batch_size, 
        callbacks=callbacks, epochs=epcho, verbose=1)

    
def finetune_deep(modelName, trainGen, valGen, batch_size, output_path, saved_model, learning_rate, epcho):
    if modelName == 'VGG16':
        unfreeze_layer = 15
    elif modelName == 'VGG19':
        unfreeze_layer = 15
    elif modelName == 'ResNet50':
        unfreeze_layer = 129
    elif modelName == 'InceptionV3':
        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 249 layers and unfreeze the rest:
        unfreeze_layer = 249
    elif modelName == 'Xception':
        unfreeze_layer = 249

    print("[INFO]: Loadign model....")
    model = load_model(saved_model)

    # construct the set of callbacks
    figure_path = os.path.sep.join([output_path, '{}_{}.png'.format(modelName, os.getpid())])
    json_path = os.path.sep.join([output_path, "{}_{}.json".format(modelName, os.getpid())])
    monitor = TrainingMonitor(figure_path, json_path)
    checkpoint = ModelCheckpoint(saved_model, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
    callbacks = [monitor, checkpoint]

    # unfreeze all the layers
    for layer in model.layers:
        layer.trainable = True

    optimizer = SGD(lr=learning_rate, decay=learning_rate/batch_size, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    print("[INFO]: Fine-tuning model....")
    H = model.fit_generator(trainGen.generator(), 
        validation_data=valGen.generator(), 
        steps_per_epoch=trainGen.numImages//batch_size, 
        validation_steps=valGen.numImages//batch_size, 
        callbacks=callbacks, epochs=epcho, verbose=1)
