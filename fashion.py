import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt



f = keras.datasets.fashion_mnist
(train_images, train_lables), (test_images, test_lables) = f.load_data()



class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0



plt.imshow(train_images[1])



model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])



model.compile(optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy'])



model.fit(train_images, train_lables, epochs = 4)



test_loss, test_acc = model.evaluate(test_images, test_lables, verbose = 1)
print("Test accuracy: ", test_acc)




predictions = model.predict(test_images)
print(class_names[np.argmax(predictions[0])])
plt.imshow(test_images[0])