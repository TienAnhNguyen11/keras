from keras import Sequential
from keras.layers import Dense
from numpy import loadtxt
from sklearn.model_selection import train_test_split

dataset = loadtxt("diabetes.csv", delimiter=",")

X = dataset[:,0: 8]
Y = dataset[:, 8]

X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.2)

model = Sequential()
model.add(Dense(16, input_dim=8, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, Y_train, epochs=100, batch_size=8, validation_data=(X_val, Y_val))

model.save("mymodel.h5")
