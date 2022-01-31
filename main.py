import tensorflow as tf
from tensorflow import keras
import numpy as np
import SEED
import SaveRes
from tensorflow.keras import backend as kerasbe

CHOSEN_SEED = SEED.seed()

tf.random.set_seed(CHOSEN_SEED)

# Filename in which to save some important results
RESULT_FILE_NAME = "result_file.txt"

# List of directory names
directory_names = [
    "TallBuilding", "Suburb", "Street", "Store", "OpenCountry", "Office",
    "Mountain", "LivingRoom", "Kitchen", "InsideCity", "Industrial", "Highway",
    "Forest", "Coast", "Bedroom"
]

trainall = False
loadall = False
# Set to True a value to train a Network specific to a part of the exercise
to_train = dict(base=False,
                augment=False,
                batchno=False,
                finetune=False,
                incrconv=False,
                vgg=False,
                dropout=False,
                regs_1 = True,
                extravgg = False,
                regslad = True,
                basead = False,
                batchad = False,
                extravggad = False)
to_load = dict(base=True,
               augment=True,
               batchno=True,
               finetune=True,
               incrconv=True,
               vgg=True,
               dropout=True,
               regs_1 = False,
               extravgg = True,
               regslad = False,
               basead = True,
               batchad = True,
               extravggad = True)
if (trainall):
    for key in to_train.keys():
        to_train[key] = True
if (loadall):
    for key in to_load.keys():
        to_load[key] = True
for key in to_train.keys():
    if (not (to_train[key] != to_load[key])):
        print("Error occured when processing key " + str(key))
        raise (NameError("Contradicting instructions: train or load? "))

# PIck value for momentum = 0.9 following deeplearningbook.pdf
# and momentum.pdf (article with Hinton as coauthor)

# START LOADING THE DATASET

img_height = 64
img_width = 64
input_image_size = (img_height, img_width)
size_validation = 0.15
# Only anisotropic scaling: this value must be set to False
preserve_aspect_ratio = False
# In Keras, the batch size is defined when building the dataset
batch_size = 32
# The subset argument is to be set according to the dataset that will
# be returned (set to "training" to get the training set, and to
# "validation" to get the validation dataset)
training_dataset = keras.utils.image_dataset_from_directory(
    directory="data/train",
    color_mode="grayscale",
    batch_size=batch_size,
    seed=CHOSEN_SEED,
    image_size=input_image_size,
    validation_split=0.15,
    subset="training",
    interpolation="bilinear",
    crop_to_aspect_ratio=preserve_aspect_ratio)

validation_dataset = keras.utils.image_dataset_from_directory(
    directory="data/train",
    color_mode="grayscale",
    batch_size=batch_size,
    seed=CHOSEN_SEED,
    image_size=input_image_size,
    validation_split=0.15,
    subset="validation",
    interpolation="bilinear",
    crop_to_aspect_ratio=preserve_aspect_ratio)
# Build the architecture specified in the directions
# Prepare the layers with the specified intializers
## Default bias initializer is already "zeros"

# The input layer will also normalize the input
kernel_weights_initializer = keras.initializers.RandomNormal(mean=0.0,
                                                             stddev=0.01)
conv_1 = keras.layers.Conv2D(8, (3, 3),
                             strides=(1, 1),
                             kernel_initializer=kernel_weights_initializer,
                             padding="same")
relu_1 = keras.layers.ReLU()
pool_1 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
conv_2 = keras.layers.Conv2D(16, (3, 3),
                             strides=(1, 1),
                             kernel_initializer=kernel_weights_initializer,
                             padding="same")
relu_2 = keras.layers.ReLU()
pool_2 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
conv_3 = keras.layers.Conv2D(32, (3, 3),
                             strides=(1, 1),
                             kernel_initializer=kernel_weights_initializer,
                             padding="same")
relu_3 = keras.layers.ReLU()
fully_connected = keras.layers.Dense(
    15, kernel_initializer=kernel_weights_initializer)
softmax_layer = keras.layers.Softmax()

partone_model = keras.Sequential([
    keras.layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 1)),
    conv_1,
    relu_1,
    pool_1,
    conv_2,
    relu_2,
    pool_2,
    conv_3,
    relu_3,
    # Flatten before fully connected
    keras.layers.Flatten(),
    fully_connected,
    softmax_layer
])

###keras.utils.plot_model(partone_model,
#                       to_file="partonemodel.png",
#                       show_shapes=True,
#                       show_layer_names=False,
#                       dpi=300,
#                       show_layer_activations=True)

#### About Sparse
# In what follows, SparseCategoricalAccuracy is used instead of
# CategoricalAccuracy because the labels are given as integers (see
# the Keras documentation. SparseCategoricalAccuracy may be slightly
# more efficient)

# See report for reference for this specific momentum rate
# In Keras, default momentum is 0.0, so it needs to be specified
momentum_constant = 0.9
learning_rate_constant = 0.001
partone_model.compile(optimizer=keras.optimizers.SGD(
    learning_rate=learning_rate_constant, momentum=momentum_constant),
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[keras.metrics.SparseCategoricalAccuracy()])

# Since the dataset already takes care of batching,
# we don't pass a `batch_size` argument.

### EARLY STOPPING CRITERION

# Custom defined stopping criteria
import StoppingCriteria

# Changed: patience was 1000
stop_criterion = keras.callbacks.EarlyStopping(
    monitor="val_sparse_categorical_accuracy",
    patience=400,
    mode="max",
    restore_best_weights=True)
stop_criterion_second = keras.callbacks.EarlyStopping(
    monitor="val_sparse_categorical_accuracy",
    patience=500,
    mode="max",
    restore_best_weights=True)

plateu_change = keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                  patience=50,
                                                  factor=0.5,
                                                  min_lr=1e-5,
                                                  verbose=1)

val_stop_criterion = keras.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=20,
                                                   mode="min",
                                                   restore_best_weights=True)

add_stop_criterion = StoppingCriteria.EarlyStoppingByAccuracy(value=0.9,
                                                              patience=5)
add_stop_criterion_2 = StoppingCriteria.EarlyStoppingByAccuracy(value=0.9,
                                                                patience=5)

stop_after_treshold = StoppingCriteria.EarlyStoppingAfterTreshold(trsh=0.30,
                                                                  patience=30)

if (to_train["base"]):
    history = partone_model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=10000,
        callbacks=[add_stop_criterion, stop_after_treshold])
    partone_model.save("trained_base_model")
    # Plot the loss and the accuracy
    SaveRes.history_plot(history, ["loss", "val_loss"], ["Epoch", ""],
                         "loss_base.png", ["k", "g"])
    SaveRes.history_plot(
        history,
        ["sparse_categorical_accuracy", "val_sparse_categorical_accuracy"],
        ["Epoch", ""], "accuracy_base.png", ["b", "k"])
    SaveRes.history_save(history,"base_case_log")
if (to_load["base"]):
    partone_model = keras.models.load_model('trained_base_model')
# Now test
test_dataset = keras.utils.image_dataset_from_directory(
    directory="data/test",
    color_mode="grayscale",
    batch_size=batch_size,
    seed=CHOSEN_SEED,
    image_size=input_image_size,
    interpolation="bilinear",
    crop_to_aspect_ratio=preserve_aspect_ratio)

# test the model
test_history = partone_model.evaluate(test_dataset, return_dict=False)

# Save the important metrics to file
SaveRes.savetofile("Loss, accuracy for the base case: ", test_history)

# Now compute the confusion matrix (the file Confusion has routines that automatically save to a file)
import Confusion

# Get the true labels:
true_lab = Confusion.true_labels(test_dataset)

# Now get the predicted labels
y_pred = np.argmax(partone_model.predict(test_dataset), axis=1)

# Now, compute the confusion matrix and save it to file.
Confusion.save_confusion_matrix(y_pred, true_lab, test_dataset.class_names,
                                "firstconfusion.png")

#######################################################################
########### BEGIN PART TWO ############################################
#######################################################################

import Part2

# Clear the object from the first part
del partone_model

augmented_dataset = training_dataset.map(Part2.aug_lr_flip)
# Now create a training dataset by concatening both of them
complete_dataset = training_dataset.concatenate(augmented_dataset)
# Let's shuffle
complete_dataset = complete_dataset.shuffle(3000, seed=CHOSEN_SEED)
# Train the Network again, with the augmented dataset

# First, get the base model
model_second = Part2.get_untrained_base_model()

# Now, compile it
model_second.compile(optimizer=keras.optimizers.SGD(
    learning_rate=learning_rate_constant, momentum=momentum_constant),
                     loss=keras.losses.SparseCategoricalCrossentropy(),
                     metrics=[keras.metrics.SparseCategoricalAccuracy()])
add_stop_criterion_3 = StoppingCriteria.EarlyStoppingByAccuracy(value=0.9,
                                                              patience=5)
#add_stop_criterion_3 = keras.callbacks.EarlyStopping(patience = 50);
# Now train, with the augmented dataset
#Try with a reduced learning rate
augment_reduce = keras.callbacks.ReduceLROnPlateau(patience = 300, factor = 0.1,cooldown=20)
stop_val = keras.callbacks.EarlyStopping(patience = 400)
if (to_train["augment"]):
    history_augmented = model_second.fit(
        complete_dataset,
        validation_data=validation_dataset,
        epochs=2000,
        callbacks=[ stop_val, augment_reduce]) #,augment_reduce]) #stop_Criterion #was add_stop_criterion_3, changed stop_val
    SaveRes.history_save(history_augmented,"augment_case_log.csv")
    
    # Save once tha training stops
    model_second.save("trained_with_augmented")
if (to_load["augment"]):
    model_second = keras.models.load_model("trained_with_augmented")

# Evaluate on the test set and save the result to file
SaveRes.eval_and_save(test_dataset, model_second,
                      "After augmenting all the images with lr flip:  ")

#### BATCH NORMALIZATION ################
batchno_model = Part2.get_model_batchno()

# With BatchNormalization, it is possible to increase the learning rate (see Ioffe-Szegedy).
batchno_model.compile(optimizer=keras.optimizers.SGD(
    learning_rate=0.01, momentum=momentum_constant),
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[keras.metrics.SparseCategoricalAccuracy()])

batchno_stop_criterion = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=30, restore_best_weights=True)
add_stop_criterion = StoppingCriteria.EarlyStoppingByAccuracy(value=0.9,
                                                              patience=5)
# Train
if (to_train["batchno"]):
    history_batchno = batchno_model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=500,
        callbacks=[batchno_stop_criterion])
    SaveRes.history_save(history_batchno,"batchno_case_log")
    batchno_model.save("batchno")

if (to_load["batchno"]):
    batchno_model = keras.models.load_model("batchno")
# Evaluate on the test set and save the result to a file
SaveRes.eval_and_save(
    test_dataset, batchno_model,
    "With Batch Normalization: (training options changed): ")

#SaveRes.history_plot(history_batchno,["loss","val_loss"],["Epoch",""],"testfile.png", ["k","k--"])
######### Different Size of Input filters ##########
incr_model = Part2.get_increasing_convmodel()
# Same learning options
incr_model.compile(optimizer=keras.optimizers.SGD(
    learning_rate=learning_rate_constant, momentum=momentum_constant),
                   loss=keras.losses.SparseCategoricalCrossentropy(),
                   metrics=[keras.metrics.SparseCategoricalAccuracy()])
# Train
convnet_stop_criterion = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=50,
    restore_best_weights=True)
if (to_train["incrconv"]):
    history_incrconv = incr_model.fit(training_dataset,
                                      validation_data=validation_dataset,
                                      epochs=500,
                                      callbacks=[convnet_stop_criterion])
    incr_model.save("incr_conv_model")
    SaveRes.history_save(history_incrconv,"incrconv_case_log")
if (to_load["incrconv"]):
    incr_model = keras.models.load_model("incr_conv_model")
# Evaluate and save
SaveRes.eval_and_save(
    test_dataset, incr_model,
    "With increasing size of conv. filters: (same training options): ")

#### DROPOUT LAYER
dropout_model = Part2.get_dropout_model([0.2, 0.5, 0.5])

dropout_model.compile(optimizer=keras.optimizers.SGD(
    learning_rate=learning_rate_constant, momentum=momentum_constant),
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[keras.metrics.SparseCategoricalAccuracy()])

dropout_earlystopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=50,
    mode="min",
    restore_best_weights=True)
dropout_lr_change = keras.callbacks.ReduceLROnPlateau()


if (to_train["dropout"]):
    history_drop = dropout_model.fit(training_dataset,
                                     validation_data=validation_dataset,
                                     epochs=500,
                                     callbacks=[dropout_earlystopping])
    dropout_model.save("dropout_model")
    SaveRes.history_save(history_drop,"dropout_case_log")
if (to_load["dropout"]):
    dropout_model = keras.models.load_model("dropout_model")

SaveRes.eval_and_save(test_dataset, dropout_model, "With dropout:  ")

###### Playing with parameters ###########################

# Let's regularize by performing L2 regularization on the convultional
# layers, still with dropout, following Srivastava et al.
l2regmodel = Part2.model_l2_reg(0.002, [0.0, 0.0, 0.0])
dropout_earlystopping_2 = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=30,
    mode="min",
    restore_best_weights=True)
dropoutcb = [dropout_earlystopping_2] #dropout_lr_change]
l2regmodel.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001,
                                                  momentum=momentum_constant),
                   loss=keras.losses.SparseCategoricalCrossentropy(),
                   metrics=[keras.metrics.SparseCategoricalAccuracy()])
reducelrcb = keras.callbacks.ReduceLROnPlateau(patience = 100)
dropout_earlystopping_3 = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=150,
    mode="min",
    restore_best_weights=True)
if (to_train["regs_1"]):
    l2hist = l2regmodel.fit(training_dataset,
                            validation_data=validation_dataset,
                            epochs=1000)#, #callbacks = [dropout_earlystopping_3, reducelrcb])#,
                            #callbacks=dropoutcb)
    l2regmodel.save("l2cost_and_dropout")
    SaveRes.history_save(l2hist,"l2reg_case_log")
if (to_load["regs_1"]):
    l2regmodel = keras.models.load_model("l2cost_and_dropout")
SaveRes.eval_and_save(test_dataset, l2regmodel,
                      "Dropout and Weight regularization:    ")
                      
#### DROPOUT, WEIGHT REG, ADAM OPTIMIZER #########
l2regmodelad = Part2.model_l2_reg(0.002, [0.2, 0.5, 0.5])
l2regmodelad.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01,
                                                  momentum=momentum_constant),
                   loss=keras.losses.SparseCategoricalCrossentropy(),
                   metrics=[keras.metrics.SparseCategoricalAccuracy()])
dropout_earlystopping_3 = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=50,
    mode="min",
    restore_best_weights=True)
if (to_train["regslad"]):
    l2hist = l2regmodelad.fit(training_dataset,
                            validation_data=validation_dataset,
                            epochs=1000, callbacks = [dropout_earlystopping_3])
    l2regmodelad.save("l2cost_and_dropout_reg")
    SaveRes.history_save(l2hist,"l2regadam_case_log")
if (to_load["regslad"]):
    l2regmodel = keras.models.load_model("l2cost_and_dropout")
SaveRes.eval_and_save(test_dataset, l2regmodel,
                      "Dropout and Weight regularization, adam optimizer:    ")
                      
# Base + Adam optimizer

if (to_train["basead"]):
    base_mod = Part2.get_untrained_base_model()
    base_mod.compile(optimizer = keras.optimizers.Adam(),loss=keras.losses.SparseCategoricalCrossentropy(),
                   metrics=[keras.metrics.SparseCategoricalAccuracy()])
    earlystop = StoppingCriteria.EarlyStoppingByAccuracy(patience = 5)
    history_base_ad = base_mod.fit(training_dataset,validation_data = validation_dataset, epochs = 1000, callbacks = [earlystop])
    base_mod.save("base_model_adam")
if (to_load["basead"]):
    base_mod = keras.models.load_model("base_model_adam")
SaveRes.eval_and_save(test_dataset, base_mod, "Base with Adam optimizer:   ")


# Batch, adam optimizer
if (to_train["batchad"]):
    batchad = Part2.get_model_batchno()
    batchad.compile(optimizer = keras.optimizers.Adam(),loss=keras.losses.SparseCategoricalCrossentropy(),
                   metrics=[keras.metrics.SparseCategoricalAccuracy()])
    hist_batchad = batchad.fit(training_dataset,validation_data = validation_dataset, epochs = 1000, callbacks = [batchno_stop_criterion])
    batchad.save("batchno_ad")
if (to_load["batchad"]):
    batchad = keras.models.load_model("batchno_ad")
SaveRes.eval_and_save(test_dataset, batchad, "Batchno with Adam Optimizer:       ")

##############################################################################
###########BEGIN PART 3 #####################################################
#############################################################################

import Part3
# We'll try with the ResNet50 model. This model accepts only color images, so we need to do a couple of things first: first convert all the images to rgb.
rgb_train_ds = training_dataset.map(Part3.f_gray_2rgb_prep)
rgb_val_ds = validation_dataset.map(Part3.f_gray_2rgb_prep)
rgb_test_ds = validation_dataset.map(Part3.f_gray_2rgb_prep)

# Each Keras application requires very specific preprocessing, so
# we'll do that as well (for example, resnet requires inputs to be in
# a BGR rather than RGB format. The function in Part3.py take care of
# this
resnet_train_ds = rgb_train_ds.map(Part3.f_resnet50_pre)
resnet_val_ds = rgb_val_ds.map(Part3.f_resnet50_pre)
resnet_test_ds = rgb_test_ds.map(Part3.f_resnet50_pre)

# Now load the model
resnet_base = keras.applications.resnet50.ResNet50(weights="imagenet",
                                                   include_top=False,
                                                   pooling = "max",
                                                   input_shape=(64, 64, 3))

# We only want to train the final layer that we are about to add
resnet_base.trainable = False

# Add the last parameters
fine_tune_model = keras.Sequential([
    resnet_base,
    keras.layers.Flatten(),
    #keras.layers.Dense(1000),
    keras.layers.Dense(15),
    keras.layers.Softmax()
])

# Compile and then train
stop_criterion_b = keras.callbacks.EarlyStopping(
    monitor="val_sparse_categorical_accuracy",
    mode="max",
    # min_delta=0.001,
    patience=50,
    restore_best_weights=True)
fine_tune_model.compile(optimizer=keras.optimizers.SGD(learning_rate = 1e-5,momentum = 0.9),
                        loss=keras.losses.SparseCategoricalCrossentropy(),
                        metrics=[keras.metrics.SparseCategoricalAccuracy()])

# cringe_criterion = StoppingCriteria.EarlyStoppingByAccuracy(value=0.9999,
#        patience=200)
# Will multiply the learning rate by factor if no improvement is detected for patience times:
# Notice that the patience is smaller than the one for the early stoppign criterion, otherwise there is no point here!
reduce_lr_finetune = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                       factor=0.2,
                                                       patience=15,
                                                       cooldown = 5,
                                                       min_lr=1e-8)
val_stop_criterion_ft = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=50,
    #min_delta=0.0001,
    mode="min",
    restore_best_weights=True)

if (to_train["finetune"]):
    history_fine_tune = fine_tune_model.fit(
        resnet_train_ds,
        validation_data=resnet_val_ds,
        epochs=10000,
        callbacks=[reduce_lr_finetune, val_stop_criterion_ft])
    fine_tune_model.save("resnet_finetuned")
    SaveRes.history_save(history_fine_tune,"resnetft_case_log")
    print("Finished RESNET50")
if (to_load["finetune"]):
    fine_tune_model = keras.models.load_model("resnet_finetuned")
# Evaluate and save
SaveRes.eval_and_save(resnet_test_ds, fine_tune_model,
                      "Fine-tuning ResNet50:  ")

####################### VGG19 #######
# Trying with vgg19, load it with no final densely connected layers
# and with max pooling on the pooling layer before the final fully
# connected layers
base_vgg19 = keras.applications.vgg19.VGG19(weights="imagenet",
                                            include_top=False,
                                            input_shape=(64, 64, 3),
                                            pooling="avg")

# Set it to untrainable
base_vgg19.trainable = False
# Build the final model: to get slightly more parameters, add an
# additional fully connected layer before the final classification
fine_tune_vgg19 = keras.Sequential([
    base_vgg19,
    keras.layers.Dense(500),
    keras.layers.Dense(15),
    keras.layers.Softmax()
])
# Plot the model:
# keras.utils.plot_model(fine_tune_vgg19,
#                       to_file="vgg19model.png",
#                       show_shapes=True,
#                       show_layer_names=False,
#                       dpi=300,
#                       show_layer_activations=True)

# Prepare the datasets
vgg19_train_ds = rgb_train_ds.map(Part3.f_vgg19_pre)
vgg19_val_ds = rgb_val_ds.map(Part3.f_vgg19_pre)
vgg19_test_ds = rgb_test_ds.map(Part3.f_vgg19_pre)
### Compile and train:
fine_tune_vgg19.compile(  #optimizer=keras.optimizers.SGD(learning_rate = 1e-3),
    optimizer=keras.optimizers.SGD(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()])

#history_fine_tune_vgg19_1 = fine_tune_vgg19.fit(vgg19_train_ds,
#                                                validation_data=vgg19_val_ds,
#                                                epochs=50)
#SaveRes.history_save(history_fine_tune_vgg19_1, "fine_tune_vgg19_2.csv")
# Change to lower learning rate and continue
#kerasbe.set_value(fine_tune_vgg19.optimizer.learning_rate, 1e-3)

if (to_train["vgg"]):
    history_fine_tune_vgg19_2 = fine_tune_vgg19.fit(
        vgg19_train_ds,
        validation_data=vgg19_val_ds,
        epochs=1000,
        callbacks=[val_stop_criterion_ft, reduce_lr_finetune])
    SaveRes.history_save(history_fine_tune_vgg19_2, "fine_tune_vgg19_2.csv")
    fine_tune_vgg19.save("fine_tune_vgg19")
    print("FINISHED VGG19")
if (to_load["vgg"]):
    fine_tune_vgg19 = keras.models.load_model("fine_tune_vgg19")
    SaveRes.eval_and_save(vgg19_test_ds, fine_tune_vgg19,
                          "FineTuning vgg19:   ")
 

## Once again with a different model, this time we only add a single
## densely connected outputlayer with 15 output neurons
reduce_lr_finetune_2 = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                       factor=0.2,
                                                       patience=5,
                                                       cooldown = 1,
                                                       min_lr=1e-9)
val_stop_criterion_ft_2 = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    min_delta=0,
    mode="min",
    restore_best_weights=True)
if (to_train["extravgg"]):
    fine_tune_vgg19_noadd = keras.Sequential([base_vgg19,
                                          keras.layers.Flatten(),  keras.layers.Dense(15),
                                              keras.layers.Softmax()])
    fine_tune_vgg19_noadd.compile(optimizer=keras.optimizers.SGD(learning_rate = 1e-5, momentum = 0.9),#SGD(learning_rate=1e-4,momentum=0.9),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()])    
    history_ft_vgg19_noadd = fine_tune_vgg19_noadd.fit(vgg19_train_ds, validation_data = vgg19_val_ds, epochs = 1000, callbacks = [val_stop_criterion_ft_2,reduce_lr_finetune_2])#[val_stop_criterion_ft_2, reduce_lr_finetune_2]) 
    SaveRes.history_save(history_ft_vgg19_noadd,"ftvgg19_case_log_actual.csv")
    fine_tune_vgg19_noadd.save("fine_tune_extra_vgg19")

if  (to_load["extravgg"]):
    fine_tune_vgg19_noadd = keras.models.load_model("fine_tune_extra_vgg19")

SaveRes.eval_and_save(vgg19_test_ds, fine_tune_vgg19_noadd, "Fine tuning vgg19, no intermediate Dense layer: ")
if (to_train["extravggad"]):
    fine_tune_vgg19_noadd = keras.Sequential([base_vgg19,
                                          keras.layers.Flatten(),  keras.layers.Dense(15),
                                              keras.layers.Softmax()])
    fine_tune_vgg19_noadd.compile(optimizer=keras.optimizers.Adam(),#SGD(learning_rate=1e-4,momentum=0.9),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()])    
    history_ft_vgg19_noadd = fine_tune_vgg19_noadd.fit(vgg19_train_ds, validation_data = vgg19_val_ds, epochs = 1000, callbacks = [val_stop_criterion_ft_2])#,reduce_lr_finetune_2])#[val_stop_criterion_ft_2, reduce_lr_finetune_2]) 
    SaveRes.history_save(history_ft_vgg19_noadd,"ftvgg19adan_case_log_actual.csv")
    fine_tune_vgg19_noadd.save("fine_tune_extra_vgg19_adam")

if  (to_load["extravggad"]):
    fine_tune_vgg19_noadd = keras.models.load_model("fine_tune_extra_vgg19_adam")
SaveRes.eval_and_save(vgg19_test_ds, fine_tune_vgg19_noadd, "Fine tuning vgg19, adam: ")