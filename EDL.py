#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import keras
from keras.datasets import mnist, cifar10, cifar100
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import cv2
import tensorflow as tf
#import GPy
#import gpflow, gpflux
import time
from tensorflow.keras.applications import VGG16,ResNet50
from keras import regularizers

import numpy as np

import sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import sklearn.gaussian_process as gp
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import matplotlib.pyplot as plt
# import official.nlp.modeling.layers as nlp_layers
# from official.nlp.modeling.layers import SpectralNormalization
import gp_layer
from sklearn.metrics import roc_auc_score
#%matplotlib inline
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#Load training data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

num_classes = 10
y_train_one_hot = keras.utils.to_categorical(y_train, num_classes)
y_test_one_hot = keras.utils.to_categorical(y_test, num_classes)

print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# kernel = gpflow.kernels.SquaredExponential()

# inducing_variable = gpflow.inducing_variables.InducingPoints(
#          np.linspace(0, 1, 128*100).reshape(-1, 128)
# )

# mean = gpflow.mean_functions.Zero()

# invlink = gpflow.likelihoods.RobustMax(10)
# likelihood = gpflow.likelihoods.MultiClass(10, invlink=invlink)

# likelihood_container = gpflux.layers.TrackableLayer()

# likelihood_container.likelihood = likelihood

# loss = gpflux.losses.LikelihoodLoss(likelihood)


gp_layer = gp_layer.RandomFeatureGaussianProcess(units=10,
                                               num_inducing=2048,
                                               normalize_input=True,
                                               scale_random_features=False,
                                               gp_cov_momentum=-1, 
                                               return_gp_cov=True)

def feature_extractor(inputs):

    feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')(inputs)
    return feature_extractor

def classifier(inputs):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dropout(0.3)(x)
    # x = tf.keras.layers.Dense(256, activation="relu")(x)
    # x = tf.keras.layers.Dense(128, activation="relu")(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    #x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
    #x = tf.keras.layers.SpectralNormalization(tf.keras.layers.Dense(512, activation='relu'))(x)
    x = (tf.keras.layers.Dense(256, activation='relu'))(x)
    x = (tf.keras.layers.Dense(128, activation='relu'))(x)
    x = (tf.keras.layers.Dense(10, activation='linear'))(x)
    # outputs = gpflux.layers.GPLayer(mean_function=mean,
    #                           kernel=kernel, 
    #                           inducing_variable=inducing_variable,
    #                           num_data=X_train.shape[0],
    #                           num_latent_gps=10)(x)
    #outputs, sd = gp_layer(x)

    return x


def final_model(inputs):

    resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)

    resnet_feature_extractor = feature_extractor(resize)
    classification_output = classifier(resnet_feature_extractor)


    return classification_output


# lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#   0.001,
#   decay_steps=20*50,
#   decay_rate=1,
#   staircase=False)


# def get_optimizer():
#   return tf.keras.optimizers.Adam(lr_schedule)


def define_compile_model():
    inputs = tf.keras.layers.Input(shape=(32,32,3))
  
    classification_output = final_model(inputs) 
    model = tf.keras.Model(inputs=inputs, outputs = classification_output)

    # model.compile(optimizer=get_optimizer(), 
    #                 loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #                 metrics = ['accuracy'])
    return model

# inputs = tf.keras.Input(shape=(28, 28, 1))

# x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
# x = tf.keras.layers.MaxPooling2D((1, 1))(x)
# x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
# x = tf.keras.layers.MaxPooling2D((2, 2))(x)
# x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
# x = tf.keras.layers.MaxPooling2D((2, 2))(x)
# x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
# x = tf.keras.layers.MaxPooling2D((2, 2))(x)
# x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
# x = tf.keras.layers.MaxPooling2D((2, 2))(x)
# x = tf.keras.layers.Flatten()(x)
# #x = tf.keras.layers.Dropout(0.5)(x)
# x = tf.keras.layers.Dense(256, activation='linear')(x)
# #x = tf.keras.layers.Dense(128, activation='linear')(x)
# #l = tf.keras.layers.Dense(10, activation='linear')(x)
# gp_output, gp_std= gp_layer(x)

# model = tf.keras.Model(inputs=inputs, outputs=gp_output)

model = define_compile_model()

model.summary()

# t = tf.expand_dims(X_train[0], axis=0)

# model(t)[0]


# from tensorflow.keras.callbacks import ReduceLROnPlateau

# lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#   0.001,
#   decay_steps=20*50,
#   decay_rate=1,
#   staircase=False)

# def get_optimizer():
#   return tf.keras.optimizers.Adam(lr_schedule)


# #Compiling the model
# model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer = get_optimizer(), metrics=['accuracy'])
# # early_stop = EarlyStopping(monitor='val_loss',patience=5)
# # checkpoint = ModelCheckpoint("./Best_model/",save_best_only=True,)
# rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)


# # # # Train the model
# model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test), callbacks=[rlrp])

# predictions = np.argmax(model.predict(X_test), axis=1)

# print(classification_report(y_test, predictions))  

# print(model(X_train[0].reshape(1,32,32,3)))

#t = X_train[0].reshape(1,32,32,3)

#model.predict(t)


def relu_evidence(logits):
    return tf.nn.relu(logits)

def exp_evidence(logits):
    return tf.exp(tf.clip_by_value(logits, -10, 10))
    

def softplus_evidence(logits):
    return tf.nn.softplus(((logits + 1)**2) / 2)

# # # def log_marginal_likelihood_gp_layer(model, X_train, y_train):
# # #     """Compute the log marginal likelihood for a GP layer within the model."""
# # #     gp_layer = model.layers[-1] 


# # #     kernel = gp_layer.kernel
# # #     inducing_points = gp_layer.inducing_variable.Z.numpy()
# # #     mean = gp_layer.mean_function


# # #     y_train_subset = y_train[:inducing_points.shape[0]].astype(np.float64)  # Ensure float64 dtype


# # #     K = kernel.K(inducing_points)  
# # #     K += np.eye(inducing_points.shape[0]) * 1e-6  


# # #     L = tf.linalg.cholesky(K)


# # #     alpha = tf.linalg.cholesky_solve(L, y_train_subset)


# # #     log_likelihood = -0.5 * tf.reduce_sum(tf.matmul(tf.transpose(y_train_subset), alpha)) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L))) - 0.5 * inducing_points.shape[0] * np.log(2 * np.pi)

# # #     return tf.squeeze(log_likelihood)



def kl_divergence(alpha):
    # KL divergence for Dirichlet distribution
    beta = tf.ones_like(alpha)
    S_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)
    S_beta = tf.reduce_sum(beta, axis=1, keepdims=True)
    
    lnB = tf.math.lgamma(S_alpha) - tf.reduce_sum(tf.math.lgamma(alpha), axis=1, keepdims=True)
    lnB_uni = tf.reduce_sum(tf.math.lgamma(beta), axis=1, keepdims=True) - tf.math.lgamma(S_beta)
    
    dg0 = tf.math.digamma(S_alpha)
    dg1 = tf.math.digamma(alpha)
    
    kl = tf.reduce_sum((alpha - beta) * (dg1 - dg0), axis=1, keepdims=True) + lnB + lnB_uni
    return kl



def loglikelihood_loss(y, alpha):
    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    S = tf.cast(S, tf.float32)  
    y = tf.cast(y, tf.float32) 
    alpha = tf.cast(alpha, tf.float32)  
    loglikelihood_err = tf.reduce_sum(tf.square(y - (alpha / S)), axis=1, keepdims=True)
    loglikelihood_var = tf.reduce_sum(alpha * (S - alpha) / (S * S * (S + 1)), axis=1, keepdims=True)
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes=10, annealing_step=10):
    loglikelihood = loglikelihood_loss(y, alpha)

    annealing_coef = tf.minimum(
        tf.constant(1.0, dtype=tf.float32),
        tf.cast(epoch_num / annealing_step, dtype=tf.float32),
    )
    
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha)
    
    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    vacuity = num_classes / tf.stop_gradient(S)
    vacuity = tf.identity(vacuity, name="vacuity")
    

    # gp_layer = model.layers[-1] 

    # ker = gp_layer.kernel
    # ind = gp_layer.inducing_variable
    
    # K = ker.K(inducing_variable.Z)  # Kernel matrix at inducing points
    # reg = tf.sqrt(tf.reduce_sum(tf.square(K))).numpy()*0.001
    #reg = log_marginal_likelihood_gp_layer(model, X_train, y_train_one_hot)
    #reg = tf.cast(reg, dtype=tf.float32)

    return loglikelihood + kl_div, vacuity


# # # def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
# # #     y = tf.convert_to_tensor(y, dtype=tf.float32)
# # #     alpha = tf.convert_to_tensor(alpha, dtype=tf.float32)
# # #     S = tf.reduce_sum(alpha, axis=1, keepdims=True)

# # #     A = tf.reduce_sum(y * (func(S) - func(alpha)), axis=1, keepdims=True)

# # #     annealing_coef = tf.minimum(
# # #         tf.constant(1.0, dtype=tf.float32),
# # #         tf.constant(epoch_num / annealing_step, dtype=tf.float32),
# # #     )

# # #     kl_alpha = (alpha - 1) * (1 - y) + 1
# # #     kl_div = annealing_coef * kl_divergence(kl_alpha)
    
# # #     S = tf.reduce_sum(alpha, axis=1, keepdims=True)
# # #     with tf.GradientTape() as tape:
# # #         vacuity = num_classes / tf.stop_gradient(S)
    
# # #     return A +  kl_div, vacuity


def compute_metrics(logits, Y, epoch, global_step, annealing_step, lmb=0.0005):
    logits = tf.cast(logits, tf.float32)
    evidence = exp_evidence(logits)
    alpha = evidence + 1
    alpha = tf.cast(alpha, tf.float32)
    Y_onehot = tf.one_hot(Y, depth=10)
    K = 10

    if len(alpha.shape) == 1:  
        u = K / tf.reduce_sum(alpha) 
    else:
        u = K / tf.reduce_sum(alpha, axis=1, keepdims=True) 

    #u = K / tf.reduce_sum(alpha, axis=1, keepdims=True)  # uncertainty
    prob = alpha / tf.reduce_sum(alpha, axis=1, keepdims=True)

    mse_loss_val, vacuity = mse_loss(Y_onehot, alpha, epoch, num_classes, annealing_step)
    loss = tf.reduce_mean(mse_loss_val)

    output_correct = logits * Y_onehot
    #print(vacuity * output_correct)
    
    loss -= (tf.reduce_sum(vacuity * output_correct) / tf.cast(tf.shape(output_correct)[0], tf.float32))
    #print(loss)
    # loss, vacuity = mse_loss(Y_onehot, alpha, epoch)
    # l2 = model.l2_loss_last_layers()
    # loss = tf.reduce_mean(loss) + lmb * l2
    return loss, u, prob


x_train = np.array(X_train)
y_train = np.array(y_train)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer)
num_epochs = 15
batch_size = 32

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.shuffle(buffer_size=len(X_test)).batch(batch_size)

# # # def get_multiple_samples(model, inputs, num_samples=5):
# # #     samples = [model(inputs, training=True) for _ in range(num_samples)]
# # #     mean_output = tf.reduce_mean(samples, axis=0)
# # #     return mean_output

for epoch in range(num_epochs):
    total_loss = 0.0
    correct = 0
    total = 0
    
    
    # indices = np.random.permutation(len(x_train))
    # x_train_shuffled = x_train[indices]
    # y_train_shuffled = y_train[indices]
    
    for inputs, labels in train_dataset:
        labels = tf.squeeze(labels)
        # inputs = x_train_shuffled[i:i+batch_size]
        # labels = y_train_shuffled[i:i+batch_size]
        
        # inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        # labels = tf.convert_to_tensor(labels, dtype=tf.int32)

        with tf.GradientTape() as tape:

            outputs = model(inputs, training=True)
            #outputs = outputs[0]
            #outputs = get_multiple_samples(model, inputs, num_samples=5)
            #print(outputs)
            #gradient_penalty = calc_gradient_penalty(X_train, outputs)


            loss, _, _ = compute_metrics(outputs, labels, epoch, global_step=epoch, annealing_step=10)
            

            #print(loss)

        gradients = tape.gradient(loss, model.trainable_variables)

        # gradients_l2 = [tf.norm(grad) for grad in gradients]

        # gradients_l2 = [0.000001*(grad_norm - 1)**2 for grad_norm in gradients_l2]

        # # Penalize the loss with the L2 norm of gradients
        # penalty_weight = 0.001  # Adjust this weight as needed
        # penalty = tf.reduce_sum([tf.square(grad) for grad in gradients_l2])
        # loss += penalty_weight * penalty

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        

        total_loss += loss.numpy()
        
        predicted = tf.argmax(outputs, axis=1)
        predicted = tf.cast(predicted, tf.int32)
        total += labels.shape[0]
        #labels = tf.squeeze(labels)
        #print(predicted)
        #print(labels)
        
        correct += tf.reduce_sum(tf.cast(predicted == tf.cast(labels, tf.int32), tf.float32)).numpy()
        
    #print(correct)
    #print(len(x_train))
    avg_loss = total_loss / (len(x_train) // batch_size)
    accuracy = 100 * correct / len(x_train)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    if avg_loss < 0.05:
        print(f'Stopping training. Loss ({avg_loss:.4f}) is below threshold ({0.05}).')
        break

predictions = np.argmax(model.predict(X_test), axis=1)

print(classification_report(y_test, predictions))

# # # #model.save('test_sngp.keras')

def test(model, test_dataset):
    correct = 0
    total = 0
    all_predictions = []
    all_uncertainties = []

    for inputs, labels in test_dataset:
        labels = tf.squeeze(labels)
        outputs = model(inputs, training=False)
        #outputs[0]
        predicted = tf.argmax(outputs, axis=1)
        predicted = tf.cast(predicted, tf.int32)

        _, u, _ = compute_metrics(outputs, labels, epoch=0, global_step=0, annealing_step=10)  # Calculate loss and uncertainty

        all_predictions.append(predicted.numpy())
        all_uncertainties.append(u.numpy())

        total += labels.shape[0]
        correct += tf.reduce_sum(tf.cast(predicted == tf.cast(labels, tf.int32), tf.float32)).numpy()

    accuracy = 100 * correct / total
    all_predictions = np.concatenate(all_predictions)
    all_uncertainties = np.concatenate(all_uncertainties)

    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Shape of predictions array: {all_predictions.shape}')
    print(f'Shape of uncertainties array: {all_uncertainties.shape}')

    np.save('predictions.npy', all_predictions)
    np.save('uncertainties.npy', all_uncertainties)

    return accuracy, all_predictions, all_uncertainties


# def add_gaussian_noise_to_image(image, noise_stddev=0.3):
#     noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=noise_stddev)
#     corrupted_image = tf.clip_by_value(image + noise, 0.0, 1.0)  # Clip values to [0, 1]
#     return corrupted_image

# # Corrupt the test dataset images with Gaussian noise
# corrupted_test_dataset = test_dataset.map(lambda x, y: (add_gaussian_noise_to_image(x), y))

# X, y = corrupted_test_dataset

# predictions = np.argmax(model.predict(X), axis=1)

# print(classification_report(y, predictions))


# _,u,_ = compute_metrics(predictions, y_test, 1, global_step=1, annealing_step=10)
test_accuracy, predictions_1, uncertainties = test(model, test_dataset)

TC_indices = []  # True Certainty (TC)
TU_indices = []  # True Uncertainty (TU)
FU_indices = []  # False Uncertainty (FU)
FC_indices = []  # False Certainty (FC)


for i in range(len(predictions)):
    #p = y_pred_mc_dropout[i]
    
    if (predictions[i] == y_test[i]):
        
        if uncertainties[i] < 0.3:
            # True certainty (TU): Correct and certain
            TC_indices.append(i)
        else:
            # False certainty (FU): Correct and uncertain
            FU_indices.append(i)
    else:
        # Certain prediction
        if uncertainties[i] < 0.3:
            # True Unertainty (TC): Incorrect and certain
            FC_indices.append(i)
        else:
            # False Uncertainty (FC): Incorrect and uncertain
            TU_indices.append(i)


print('USen:',len(TU_indices) / (len(TU_indices) + len(FC_indices)))

print('USpe:', len(TC_indices) / (len(TC_indices) + len(FU_indices)))

print('UPre:', len(TU_indices) / (len(TU_indices) + len(FU_indices)))
      
print('UAcc:', (len(TU_indices) + len(TC_indices)) / (len(TU_indices) + len(TC_indices) + len(FU_indices) + len(FC_indices)))



# def combine_images_with_padding(img_index_1, img_index_2, padding_type="top_bottom"):
#   """
#   Combines two CIFAR-10 images with padding and normalization.

#   Args:
#       img_index_1: Index of the first image in the dataset.
#       img_index_2: Index of the second image in the dataset.
#       padding_type: Type of padding to use ("top_bottom" or "left_right").

#   Returns:
#       A combined image tensor.
#   """

# def combine_images_with_padding(img_index_1, img_index_2, padding_type):
    
#     (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    
#     img_1 = tf.convert_to_tensor(test_images[img_index_1], dtype=tf.float32) / 255.0
#     img_2 = tf.convert_to_tensor(test_images[img_index_2], dtype=tf.float32) / 255.0

    
#     if padding_type == "top_bottom":
#         padding_amount = (img_2.shape[0] - img_1.shape[0]) // 2
#         top_bottom_padding = tf.zeros((padding_amount, img_1.shape[1], 3))
#         padded_img_1 = tf.concat([top_bottom_padding, img_1, top_bottom_padding], axis=0)
#         padded_img_2 = img_2
#     elif padding_type == "left_right":
#         padding_amount = (img_2.shape[1] - img_1.shape[1]) // 2
#         left_right_padding = tf.zeros((img_1.shape[0], padding_amount, 3))
#         padded_img_1 = tf.concat([left_right_padding, img_1, left_right_padding], axis=1)
#         padded_img_2 = img_2
#     else:
#         raise ValueError("Invalid padding type. Choose 'top_bottom' or 'left_right'.")

    
#     combined_img = tf.concat([padded_img_1, padded_img_2], axis=0)

    
#     combined_img_resized = tf.image.resize(combined_img, [32, 32])

#     return combined_img_resized



# img_index_1 = 50
# img_index_2 = 100
# padding_type = "top_bottom" 

# combined_img = combine_images_with_padding(img_index_1, img_index_2, padding_type)

# combined_img = np.expand_dims(combined_img, axis=0)

# image1_index = 10
# image2_index = 21


# combined_img = np.zeros((32, 32))
# combined_img[:, :-6] += x_train[image1_index][:, 6:]
# combined_img[:, 14:] += x_train[image2_index][:, 5:19]
# combined_img /= combined_img.max()

# combined_img = combined_img.reshape(1, 32, 32, 3)


(train_images, _), (_, _) = mnist.load_data()


mnist_image = train_images[np.random.randint(0, train_images.shape[0])]


rescaled_image = cv2.resize(mnist_image, (32, 32))


rgb_image = cv2.cvtColor(rescaled_image, cv2.COLOR_GRAY2RGB)

rgb_image = np.expand_dims(rgb_image, axis=0)


# pred_unc = model(combined_img)
pred = model(X_test[0].reshape(1, 32, 32, 3))
#var = pred.variance().numpy()

pred_rgb = model(rgb_image)
#var_rgb = pred_rgb.variance().numpy()
# l_unc, u_unc, p_unc = compute_metrics(pred_unc, y_test[50], 0, global_step=0, annealing_step=10)
l, u, p = compute_metrics(pred, y_test[0], 0, global_step=0, annealing_step=10)
l_rgb, u_rgb, p_rgb = compute_metrics(pred_rgb, y_test[0], 0, global_step=0, annealing_step=10)

# print('u_unc:',u_unc)
# print('p_unc:',p_unc)
# print('preds:', pred_unc)

print('u:', u)
print('p:', p)
print('pred:', pred)
#print('sd:', var)

print('u_rgb:', u_rgb)
print('p_rgb:', p_rgb)
print('preds:', pred_rgb)
#print('sd_rgb:', var_rgb)

#----------------------------------------------------------------------------------------------------
#Variance based EDL

def uncertainty(alpha, reduce=True):
    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    p = alpha / S
    variance = p - tf.square(p)
    EU = (alpha / S) * (1 - alpha / S) / (S + 1)
    AU = variance - EU
    if reduce:
        AU = tf.reduce_sum(AU) / alpha.shape[0]
        EU = tf.reduce_sum(EU) / alpha.shape[0]
    return AU, EU

pred_var = model(rgb_image)
pred_var = exp_evidence(pred_var)

unc_ale, unc_eps = uncertainty(pred_var)
print('u_ale:', unc_ale)
print('p_eps:', unc_eps)

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

#-----------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------
#Different Variance based unc

# def total_uncertainty_variance(probs):
#     if isinstance(probs, tf.Tensor):
#         mean = tf.reduce_mean(probs, axis=2)
#         t_u = tf.reduce_sum(mean * (1 - mean), axis=1)
#     else:
#         probs = tf.convert_to_tensor(probs, dtype=tf.float32)
#         mean = tf.reduce_mean(probs, axis=2)
#         t_u = tf.reduce_sum(mean * (1 - mean), axis=1)
#     return t_u

# def aleatoric_uncertainty_variance(probs):
#     if isinstance(probs, tf.Tensor):
#         a_u = tf.reduce_mean(tf.reduce_sum(probs * (1 - probs), axis=1), axis=1)
#     else:
#         probs = tf.convert_to_tensor(probs, dtype=tf.float32)
#         a_u = tf.reduce_mean(tf.reduce_sum(probs * (1 - probs), axis=1), axis=1)
#     return a_u

# def epistemic_uncertainty_variance(probs):
#     if isinstance(probs, tf.Tensor):
#         mean = tf.reduce_mean(probs, axis=2, keepdims=True)
#         e_u = tf.reduce_mean(tf.reduce_sum(probs * (probs - mean), axis=1), axis=1)
#     else:
#         probs = tf.convert_to_tensor(probs, dtype=tf.float32)
#         mean = tf.reduce_mean(probs, axis=2, keepdims=True)
#         e_u = tf.reduce_mean(tf.reduce_sum(probs * (probs - mean), axis=1), axis=1)
#     return e_u

# eu = epistemic_uncertainty_variance(pred_rgb)
# au = aleatoric_uncertainty_variance(pred_rgb)

# print('eu:', eu)
# print('au:', au)


#------------------------------------------------------------------------------------------------------

def softmax(vector):
    e = np.exp(vector)
    return e / e.sum()

def expected_calibration_error(samples, true_labels, M=5):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    #samples = softmax(samples)

    # get max probability per sample i
    confidences = np.max(samples, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(samples, axis=1)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece

ece = expected_calibration_error(y_pred_probs, y_test)
print("Expected Calibration Error:", ece)

# xtest = X_test[0]

# xtest = tf.convert_to_tensor([xtest])

# # Define the FGSM attack function
# def fgsm_attack(image, label, epsilon):
#     with tf.GradientTape() as tape:
#         tape.watch(image)
#         prediction = model(image)
#         prediction = exp_evidence(prediction) + 1
#         loss,_ = mse_loss(label, prediction, epoch_num=1, num_classes=10, annealing_step=10)
#         #loss = tf.keras.losses.sparse_categorical_crossentropy(label, prediction)
#     gradient = tape.gradient(loss, image)
#     signed_grad = tf.sign(gradient)
#     adversarial_image = image + epsilon * signed_grad
#     adversarial_image = tf.clip_by_value(adversarial_image, -1, 1)
#     return adversarial_image

# # Create the adversarial image
# epsilon = 0.5
# label = tf.convert_to_tensor([y_test[0]], dtype=tf.int64)
# adversarial_image = fgsm_attack(xtest, label, epsilon)


# # Get the model predictions for both images
# original_pred = model(xtest)
# adversarial_pred = model(adversarial_image)

# l1, u1, p1 = compute_metrics(adversarial_pred, y_test[0], 0, global_step=0, annealing_step=10)

# print('u_rgb:', u1)
# print('p_rgb:', p1)
#print('preds:', pred_rgb)

# # # def plot_reliability_diagram(confidences, true_labels, M=5):
# # #   """Plots the reliability diagram for the given data."""
# # #   bin_boundaries = np.linspace(0, 1, M + 1)
# # #   bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

# # #   # Get binned accuracy (average accuracy for each confidence bin)
# # #   binned_accuracy = np.zeros(M)
# # #   for i, bin_lower in enumerate(bin_boundaries[:-1]):
# # #     bin_upper = bin_boundaries[i + 1]
# # #     in_bin = np.logical_and(confidences >= bin_lower, confidences < bin_upper)
# # #     if in_bin.sum() > 0:
# # #       binned_accuracy[i] = true_labels[in_bin].mean()

# # #   # Perfect calibration line (y = x)
# # #   perfect_calibration = np.linspace(0, 1, M)

# # #   plt.plot(bin_centers, binned_accuracy, 'o', label='Binned Accuracy')
# # #   plt.plot(perfect_calibration, perfect_calibration, '-', label='Perfect Calibration')
# # #   plt.xlabel('Predicted Probability')
# # #   plt.ylabel('Observed Accuracy')
# # #   plt.title('Reliability Diagram')
# # #   plt.legend()
# # #   plt.grid(True)
# # #   plt.show()


# # #plot_reliability_diagram(y_pred_probs, y_test)



# # # def fgsm_attack(image, epsilon, data_grad):
# # #     # Collect the element-wise sign of the data gradient
# # #     sign_data_grad = tf.sign(data_grad)
# # #     # Create the perturbed image by adjusting each pixel of the input image
# # #     perturbed_image = image + epsilon * sign_data_grad
# # #     # Adding clipping to maintain [0,1] range
# # #     perturbed_image = tf.clip_by_value(perturbed_image, 0, 1)
# # #     # Return the perturbed image
# # #     return perturbed_image

# # # # Restores the tensors to their original scale
# # # def denorm(batch, mean=[0.1307], std=[0.3081]):
# # #     mean = tf.convert_to_tensor(mean)
# # #     std = tf.convert_to_tensor(std)

# # #     return batch * std + mean


# # # def test(model, test_dataset, epsilon):

# # #     # Accuracy counter
# # #     correct = 0
# # #     adv_examples = []

# # #     # Loop over all examples in test set
# # #     for data, target in test_dataset:

# # #         # Send the data and label to the device
# # #         data, target = data.numpy(), target.numpy()

# # #         # Set requires_grad attribute of tensor. Important for Attack
# # #         data = tf.convert_to_tensor(data, dtype=tf.float32)
# # #         with tf.GradientTape() as tape:
# # #             tape.watch(data)
# # #             # Forward pass the data through the model
# # #             output = model(data)
# # #             init_pred = tf.argmax(output, axis=1, output_type=tf.int32)

# # #             # If the initial prediction is wrong, don't bother attacking, just move on
# # #             if not np.array_equal(init_pred.numpy(), target):
# # #                 continue

# # #             # Calculate the loss
# # #             loss, _, _ = compute_metrics(outputs, target, epoch=1, global_step=0, annealing_step=10)

# # #         # Calculate gradients of model in backward pass
# # #         data_grad = tape.gradient(loss, data)

# # #         # Call FGSM Attack
# # #         perturbed_data = fgsm_attack(data, epsilon, data_grad)

# # #         # Re-classify the perturbed image
# # #         output = model(perturbed_data)

# # #         # Check for success
# # #         final_pred = tf.argmax(output, axis=1, output_type=tf.int32)
# # #         if np.array_equal(final_pred.numpy(), target):
# # #             correct += 1
# # #             # Special case for saving 0 epsilon examples
# # #             if epsilon == 0 and len(adv_examples) < 5:
# # #                 adv_examples.append((init_pred.numpy()[0], final_pred.numpy()[0], perturbed_data.numpy()))
# # #         else:
# # #             # Save some adv examples for visualization later
# # #             if len(adv_examples) < 5:
# # #                 adv_examples.append((init_pred.numpy()[0], final_pred.numpy()[0], perturbed_data.numpy()))

# # #     # Calculate final accuracy for this epsilon
# # #     final_acc = correct / float(len(test_dataset))
# # #     print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_dataset)} = {final_acc}")

# # #     # Return the accuracy and adversarial examples
# # #     return final_acc, adv_examples


# # # accuracies = []
# # # examples = []
# # # epsilons = [0,0.05, 0.1, 0.15,0.2,0.25,0.3]

# # # # Run test for each epsilon
# # # for eps in epsilons:
# # #     acc, ex = test(model, test_dataset, eps)
# # #     accuracies.append(acc)
# # #     examples.append(ex)


# # # import matplotlib.pyplot as plt

# # # # Plot accuracy vs epsilon
# # # plt.figure(figsize=(5,5))
# # # plt.plot(epsilons, accuracies, "*-")
# # # plt.yticks(np.arange(0, 1.1, step=0.1))
# # # plt.xticks(np.arange(0, .35, step=0.05))
# # # plt.title("Accuracy vs Epsilon")
# # # plt.xlabel("Epsilon")
# # # plt.ylabel("Accuracy")
# # # plt.grid(True)
# # # plt.show()

# # # # Save the plot as a PNG file
# # # plt.savefig('accuracy_vs_epsilon.png')

