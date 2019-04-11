from cnn import FCHeadNet
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.applications import VGG16, VGG19, InceptionV3, Xception, ResNet50
from keras.layers import Input
from keras.models import Model
from callback import TrainingMonitor
from keras.callbacks import ModelCheckpoint
import os

def finetune(model, trainGen, valGen, batch_size, output_path, saved_model):
    
    if model == 'VGG16':
        baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    elif model == 'VGG19':
        baseModel = VGG19(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    elif model == 'ResNet50':
        baseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    elif model == 'InceptionV3':
        baseModel = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(299, 299, 3)))
    elif model == 'Xception':
        baseModel = Xception(weights="imagenet", include_top=False, input_tensor=Input(shape=(299, 299, 3)))

    # initialilze the new head of the network, a set of FC layers followed by a softmax classifier
    headModel = FCHeadNet.build(model, baseModel, trainGen.classes, 256)

    # place the head FC model on top of the base model -- this will become the actual model we will train
    model = Model(inputs=baseModel.input, output=headModel)

    # loop over all layers in the base model and freeze them so they will not be updated during the training process
    for layer in baseModel.layers:
        layer.trainable = False

    # compile our model
    print("[INFO]: Compiling model....")
    optimizer = RMSprop(lr=1e-3)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # construct the set of callbacks
    figure_path = os.path.sep.join([output_path, '{}_{}.png'.format(model, os.getpid())])
    json_path = os.path.sep.join([output_path, "{}_{}.json".format(model, os.getpid())])
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
        callbacks=callbacks, epochs=20, verbose=1)
    
    # now that the head FC layers have been trained/initialized, lets
    # unfreeze the final set of CONV layers and make them trainable
    # if classification accuracy continues to improve withou overfitting
    # you may want to consider unfreezing more layers in the body
    #for layer in baseModel.layers[15:]:
    for layer in baseModel.layers:
        layer.trainable = True

    # for the changes to the model to take affect we need to recompile
    # the model, this time using SGD with a very small learning rate
    print("[INFO]: Re-compiling model....")
    optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    monitor = TrainingMonitor(fig_path, json_path, start_at=21)
    callbacks = [monitor, checkpoint]

    # train the model again, this time fine-tuning both the final set
    # of CONV layers along with our set of FC layers
    print("[INFO]: Fine-tuning model....")
    H = model.fit_generator(trainGen.generator(), 
        validation_data=valGen.generator(), 
        steps_per_epoch=trainGen.numImages//batch_size, 
        validation_steps=valGen.numImages//batch_size, 
        callbacks=callbacks, epochs=20, verbose=1)
