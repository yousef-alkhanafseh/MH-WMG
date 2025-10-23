#!/usr/bin/env python
# coding: utf-8

# ### @Author: Yousef Alkhanafseh
# ### @E-mail: alkhanafseh15@gmail.com

# # Libraries

# In[7]:


import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import tensorflow as tf
import tf_keras
import tensorflow_hub as hub


# # Variables

# In[8]:


SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# Option A: simple, script-relative
image_dir = SCRIPT_DIR / "data" / "images" / "images"
image_dir

# In[ ]:


fm_path = "https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5"
image_dir = os.path.join(os.getcwd(), "data", "images", "images")

image_shape = (224, 224)

mainclass_dict = {
                    'A1':0, 'A2':1, 'A3':2,
                    'A4':3, 'A5':4, 'A6':5
                 }

subclass_dict = { 
                    'off_off_on_on':0, 'off_on_off_on':1, 'off_on_on_off':2,
                    'off_on_on_on':3, 'on_off_off_on':4, 'on_off_on_off':5,
                    'on_off_on_on':6, 'on_on_off_off':7, 'on_on_off_on':8,
                    'on_on_on_off':9, 'off_off_off_off':10, 'on_on_on_on':11
                }

bins = {
        0:  (0,  0.3),   
        1:  (0.3,  3.0),  
        2:  (3.0,  5.6),  
        3:  (5.6,  8.2),  
        4:  (8.2, 13.0),   
        5:  (13.0, 17.0),  
        6:  (17.0, 20.5),  
        7:  (20.5, 24.5),   
        8:  (24.5, 46.2),   
        9:  (46.2, 68.2),  
        10:  (68.2, 90.2),   
        11: (90.2, 107.9), 
        12: (107.9, 110.0)
    }

# # Image data read

# In[ ]:


def return_img_data(X_mainLst, y_mainLst, y_subLst, y_distlst):

    print("Image reading process is started !!")

    X_out_mainLst = []
    y_out_mainLst = []
    y_out_subLst = []
    y_out_distlst = []
    X_path_lst = []
    error_lst = []
    
    for img_path, ymain, ysub, ydist in zip(X_mainLst, y_mainLst, y_subLst, y_distlst):
        try:
            one_image = cv2.imread(str(img_path))
            if one_image is None:
                print(f"Could not read image file: {img_path}")
                error_lst.append(img_path)
            one_image = cv2.cvtColor(one_image, cv2.COLOR_BGR2RGB)

            one_image = cv2.resize(one_image, image_shape)
            
            X_out_mainLst.append(one_image)
            X_path_lst.append(img_path)
            y_out_mainLst.append(ymain)
            y_out_subLst.append(ysub)
            y_out_distlst.append(ydist)

        except Exception as e:
            print(img_path, e)

    return error_lst, X_path_lst, np.array(X_out_mainLst), np.array(y_out_mainLst), np.array(y_out_subLst), np.array(y_out_distlst)

# In[ ]:


def file_path_finder(image_dir, fprefix):
    file_paths = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith(fprefix):
                file_paths.append(os.path.join(root, file))
    return file_paths

# In[ ]:


file_img_paths = file_path_finder(image_dir, '.png')

y_main = []
y_sub = []
y_dist = []  

for imgpath in file_img_paths:
    main_key = imgpath.rsplit("/", 2)[-2]
    filename = imgpath.rsplit("/", 1)[-1].replace(".png", "")
    first_underscore_split = filename.split("_", 1)
    distance_str = first_underscore_split[0]
    distance_val = str(distance_str)
    remainder = first_underscore_split[1]
    sub_key    = remainder.replace("_plot", "")


    y_main.append(mainclass_dict[main_key])
    y_sub.append(subclass_dict[sub_key])

    if sub_key == "off_off_off_off":
        y_dist.append(0)
    else:
        y_dist.append(distance_val)
        
y_main = np.array(y_main)
y_sub  = np.array(y_sub)
y_dist = np.array(y_dist)


X_train_val, X_test_temp, \
y_main_train_val, y_main_test, \
y_sub_train_val,  y_sub_test, \
y_dist_train_val, y_dist_test = train_test_split(
    file_img_paths,     
    y_main, y_sub, y_dist,    
    test_size=0.15,      # 15 % held out for final testing
    random_state=42,
    stratify=y_dist      # <-- preserves distance-bin distribution
)

# ── 2) train + val ───────────────────
X_train_temp, X_val_temp, \
y_main_train, y_main_val, \
y_sub_train,  y_sub_val, \
y_dist_train, y_dist_val = train_test_split(
    X_train_val,
    y_main_train_val, y_sub_train_val, y_dist_train_val,
    test_size=0.1765,     # 0.1765 × 0.85 ≈ 0.15  → overall 15 % validation
    random_state=42,
    stratify=y_dist_train_val   # <-- keep the same balance here too
)

print(len(y_main_train), len(y_main_val), len(y_main_test))
print(len(y_sub_train),  len(y_sub_val),  len(y_sub_test))
print(len(y_dist_train), len(y_dist_val), len(y_dist_test))

# In[ ]:


error_train_lst, X_train_temp, X_train_img, y_main_train, y_sub_train, y_dist_train = return_img_data(X_train_temp, y_main_train, y_sub_train, y_dist_train)
error_val_lst, X_val_temp, X_val_img, y_main_val, y_sub_val, y_dist_val = return_img_data(X_val_temp, y_main_val, y_sub_val, y_dist_val)
error_test_lst, X_test_temp, X_test_img, y_main_test, y_sub_test, y_dist_test = return_img_data(X_test_temp, y_main_test, y_sub_test, y_dist_test)

# # Fault distance bins balancing

# In[ ]:


def balanced_bins(datalist):
    new_list = []
    for i in datalist.tolist():
        for binindex, binrange in bins.items():
            if  float(binrange[0]) <= float(i) < float(binrange[1]):
                new_list.append(binindex)
    return np.array(new_list)

def assign_bin(x):
    for idx, (lo, hi) in bins.items():
        if lo <= float(x) <= hi:
            return idx

def assign_binrange(x):
    for idx, (lo, hi) in bins.items():
        if lo <= float(x) <= hi:
            return str(lo) + " - " + str(hi) + " km"

# In[ ]:


y_dist_train_f = balanced_bins(y_dist_train)
y_dist_val_f = balanced_bins(y_dist_val)
y_dist_test_f = balanced_bins(y_dist_test)

# In[ ]:


bins_temp_df = pd.DataFrame(y_dist_train.tolist())
bins_temp_df = bins_temp_df.rename({0:"distance"}, axis=1)
bins_temp_df["bin"] = bins_temp_df["distance"].apply(assign_bin)
bins_temp_df["distance_range"] = bins_temp_df["distance"].apply(assign_binrange)
bins_temp_df = bins_temp_df.groupby(["distance_range", "bin"], as_index=False).count()
temp_dist_df = bins_temp_df.rename({"distance":"count", "bin":"class_id"}, axis=1)

temp_dist_df.head()

# # Model training

# ## GLA layer class

# In[ ]:


class GLALayer(tf_keras.layers.Layer):
    """
    Gated Linear Attention (single head, O(N·d) time, O(d) memory)
    • inputs  : (batch, seq_len, dim)
    • outputs : (batch, seq_len, dim)    -- same shape
    """
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        # linear projections for q, k, v, and the gate
        self.to_q = tf_keras.layers.Dense(dim, use_bias=False)
        self.to_k = tf_keras.layers.Dense(dim, use_bias=False)
        self.to_v = tf_keras.layers.Dense(dim, use_bias=False)
        self.to_gate = tf_keras.layers.Dense(dim, use_bias=True)

    # ϕ(x) = ELU(x) + 1  (as in the paper)
    @staticmethod
    def phi(x):  
        return tf.nn.elu(x) + 1.0

    def call(self, inputs, training=None):
        q = self.phi(self.to_q(inputs))          # (B,S,D)
        k = self.phi(self.to_k(inputs))          # (B,S,D)
        v =            self.to_v(inputs)         # (B,S,D)

        # -------- linear attention core ------------
        kv = tf.einsum('bsd,bsf->bdf', k, v)      # (B,D,D)
        z  = 1.0 / (tf.einsum('bsd,bd->bs', q,
                              tf.reduce_sum(k, axis=1)) + 1e-6)   # (B,S)
        y  = tf.einsum('bsd,bdf->bsf', q, kv)     # (B,S,D)
        y  = y * tf.expand_dims(z, -1)            # (B,S,D)

        # -------- gating ---------------------------
        gate = tf.sigmoid(self.to_gate(inputs))   # (B,S,D)
        return gate * y

# Custom sharpening layer
class SharpenLayer(tf_keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SharpenLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Define a 3x3 sharpening kernel
        kernel = tf.constant([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]], dtype=tf.float32)
        kernel = tf.reshape(kernel, [3, 3, 1, 1])
        in_channels = tf.shape(inputs)[-1]
        kernel = tf.tile(kernel, [1, 1, in_channels, 1])
        return tf.nn.depthwise_conv2d(inputs, kernel, strides=[1, 1, 1, 1], padding='SAME')

# ## Data augmentation

# In[ ]:


# Custom sharpening layer
class SharpenLayer(tf_keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SharpenLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Define a 3x3 sharpening kernel
        kernel = tf.constant([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]], dtype=tf.float32)
        kernel = tf.reshape(kernel, [3, 3, 1, 1])
        in_channels = tf.shape(inputs)[-1]
        kernel = tf.tile(kernel, [1, 1, in_channels, 1])
        return tf.nn.depthwise_conv2d(inputs, kernel, strides=[1, 1, 1, 1], padding='SAME')

# In[ ]:


def mhwmg_model(n_main=6, n_sub=12, n_dist=12):
    """
    Inference-only MH-WMG model builder.
    Same architecture as your training graph, but without compile/optimizer/callbacks.
    """

    data_augmentation = tf_keras.Sequential(
        [
            tf_keras.layers.RandomContrast(0.3, input_shape=image_shape + (3,)),
            tf_keras.layers.Lambda(lambda x: tf.image.adjust_brightness(x, delta=0.1)),
            SharpenLayer(),
        ],
        name="data_augmentation",
    )

    inputs = tf_keras.Input(shape=image_shape + (3,), name="input_image")
    augmented = data_augmentation(inputs)

    features = hub.KerasLayer(
        fm_path,
        input_shape=image_shape + (3,),
        trainable=True,
        name="mobilenetv3_small_features",
    )(augmented)

    print(features.shape)

    x1 = tf_keras.layers.Dropout(0.3, name="dropout_x1")(features)
    x1 = tf_keras.layers.Dense(
        2 ** 8, activation="relu",
        kernel_regularizer=tf_keras.regularizers.L2(0.2),
        name="dense_x1",
    )(x1)

    x2 = tf_keras.layers.Dropout(0.3, name="dropout_x2")(x1)

    xx = tf_keras.layers.Dense(
        2 ** 9, activation="relu",
        kernel_regularizer=tf_keras.regularizers.L1(0.1),
        name="dense_xx",
    )(features)
    xx = tf_keras.layers.Dropout(0.5, name="dropout_xx")(xx)

    tokens = tf_keras.layers.Reshape((32, 16), name="tokens_reshape")(xx)
    tokens = GLALayer(dim=16, name="gla")(tokens)

    x3 = tf_keras.layers.Flatten(name="flatten_tokens")(tokens)
    x3 = tf_keras.layers.BatchNormalization(name="bn_x3")(x3)
    x3 = tf_keras.layers.Dropout(0.6, name="dropout_x3")(x3)

    main_output = tf_keras.layers.Dense(
        n_main, activation="softmax",
        kernel_regularizer=tf_keras.regularizers.L2(0.1),
        name="falut_area_output",
    )(x1)

    subclass_output = tf_keras.layers.Dense(
        n_sub, activation="softmax",
        kernel_regularizer=tf_keras.regularizers.L2(0.2),
        name="falut_type_output",
    )(x2)

    dist_output = tf_keras.layers.Dense(
        n_dist, activation="softmax",
        kernel_regularizer=tf_keras.regularizers.L1(0.1),
        name="falut_distance_bin_output",
    )(x3)

    return tf_keras.Model(
        inputs=inputs,
        outputs=[main_output, subclass_output, dist_output],
        name="mhwmg"
    )


# In[ ]:


os.makedirs("ckpts", exist_ok=True)

ckpt = tf_keras.callbacks.ModelCheckpoint(
    filepath="ckpts/mhwmg_{epoch:03d}_{val_falut_distance_bin_output_accuracy:.3f}.weights.h5",
    monitor="val_falut_distance_bin_output_accuracy",
    save_best_only=True,
    save_weights_only=True,
    mode="max",
    verbose=1,
)

BATCH_SIZE = 2**4
base_learning_rate = 0.000045
EPOCH = 100

model = mhwmg_model()
model.compile(
    optimizer= tf_keras.optimizers.Adam(base_learning_rate),
    loss={
        'falut_area_output': 'sparse_categorical_crossentropy',
        'falut_type_output': 'sparse_categorical_crossentropy',
        'falut_distance_bin_output': 'sparse_categorical_crossentropy'
    },

    loss_weights={
       'falut_area_output': 1.0,
       'falut_type_output': 3.0,
       'falut_distance_bin_output': 6.0
    },
    metrics={
        'falut_area_output': 'accuracy',
        'falut_type_output': 'accuracy',
        'falut_distance_bin_output': 'accuracy'
    }
)

history = model.fit(
    x=X_train_img,
    y={
       'falut_area_output': y_main_train,
       'falut_type_output': y_sub_train,
       'falut_distance_bin_output': np.array(y_dist_train_f)
    },
    validation_data=(
        X_val_img,
        {
            'falut_area_output': y_main_val,
            'falut_type_output': y_sub_val,
            'falut_distance_bin_output': np.array(y_dist_val_f)
        }
    ),
    epochs=EPOCH,
    batch_size=BATCH_SIZE,
    callbacks=[ckpt])


# # Model evaluation

# In[ ]:


pred_main, pred_sub, pred_dist = model.predict(X_test_img)

# In[ ]:



def head_metrics(y_true, y_pred_probs):
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.asarray(y_true)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return acc, prec, rec, f1

results = []
for head_name, y_true, y_pred_probs in [
    ("fault_area", y_main_test, pred_main),
    ("fault_type", y_sub_test, pred_sub),
    ("fault_distance_bin", np.array(y_dist_test_f), pred_dist),
]:
    acc, prec, rec, f1 = head_metrics(y_true, y_pred_probs)
    results.append(
        {"head": head_name,
         "accuracy": acc,
         "precision_macro": prec,
         "recall_macro": rec,
         "f1_macro": f1}
    )

metrics_df = pd.DataFrame(results)
print(metrics_df)

# In[ ]:



