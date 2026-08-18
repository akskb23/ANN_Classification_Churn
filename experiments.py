import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle


# Load the dataset
data= pd.read_csv("Churn_Modelling.csv")
# print(data.head())

# Preprocess the data
# Drop irrelevant features or columns
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
# print(data.head())

# Encode categorical variables
label_encoder_gender = LabelEncoder()
data["Gender"]= label_encoder_gender.fit_transform(data["Gender"])
# print(data.head())

# One-hot encode the "Geography" column
from sklearn.preprocessing import OneHotEncoder
onehot_encoder_geo = OneHotEncoder()
geo_encoder = onehot_encoder_geo.fit_transform(data[["Geography"]])

# print(geo_encodeer)

# print(onehot_encoder_geo.get_feature_names_out(['Geography'])) 

geo_encoded_df = pd.DataFrame(geo_encoder.toarray(),columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
# print(geo_encoded_df)

# Combine one -hot encoded columns with the original dataset
data=pd.concat([data.drop('Geography',axis=1),geo_encoded_df],axis=1)
# print(data.head())

# Save the encoders and scaler for future use
# with open('label_encoder_gender.pkl', 'wb') as file:
#     pickle.dump(label_encoder_gender, file)

# with open('onehot_encoder_geo.pkl', 'wb') as file:
#     pickle.dump(onehot_encoder_geo, file)

# Divide the dataset in to independent and dependent features
X = data.drop("Exited", axis=1)
y = data["Exited"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# print(X_train)
# print(X_train.shape[1])

# with open('scaler.pkl', 'wb') as file:
#     pickle.dump(scaler, file)

# ANN Implementation

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime

# Build the ANN model
model = Sequential([
    Input(shape=(X_train.shape[1],)),  # INPUT LAYER
    Dense(64, activation="relu"),  # HL1 connected with input layer
    Dense(32, activation="relu"),  # HL2
    Dense(1, activation="sigmoid")  # output layer
])

# print(model.summary())

opt= tf.keras.optimizers.Adam(learning_rate=0.01)
loss = tf.keras.losses.BinaryCrossentropy()

# Compile the model
model.compile(optimizer=opt,loss=loss,metrics=["accuracy"])

# Setup the Tensorboard
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callback= TensorBoard(log_dir=log_dir,histogram_freq=1)

# Setup Early Stopping
early_stopping_callback= EarlyStopping(monitor="val_loss",patience=10,restore_best_weights=True)

# Training the model
history= model.fit(
    X_train,y_train,validation_data=(X_test,y_test),epochs=100,
    callbacks=[tensorflow_callback,early_stopping_callback]
)

# model.save("model.h5")

# To Load TensorBoard Extension in Jupyter Notebook
# %load_ext tensorboard 

# %tensorboard --logdir logs/fit


