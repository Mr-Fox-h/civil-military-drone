import numpy as np
import datasets as dt
import matplotlib.pyplot as plt
import tensorflow as tf

### Import dataset
whole_data = dt.load_dataset("imagefolder", data_dir = "./Datasets/", split = "train")

### Train & test split
train_and_test_data = whole_data.train_test_split(test_size = 0.2)
train_images, train_labels = train_and_test_data['train'].select_columns('image'), train_and_test_data['train'].select_columns('label')
test_images, test_labels = train_and_test_data['test'].select_columns('image'), train_and_test_data['test'].select_columns('label')

train_labels = np.array(train_labels['label'])
test_labels = np.array(test_labels['label'])

### Image processing
def transforms(examples):
    examples['img'] = [image.convert("RGB").resize((32, 32)) for image in examples['image']]
    return examples

train_images = train_images.map(transforms, remove_columns = ['image'], batched = True)
test_images = test_images.map(transforms, remove_columns = ['image'], batched = True)

train_tensor = []
test_tensor = []

for index in range(len(train_images)):
    train_tensor.append(np.array(train_images[index]['img']))

for index in range(len(test_images)):
    test_tensor.append(np.array(test_images[index]['img']))

train_tensor = np.array(train_tensor)
test_tensor = np.array(test_tensor)

### Normalizing
train_tensor = train_tensor / 255
test_tensor = test_tensor / 255

### Class names of data
class_name = ['Civil', 'Militray']

### CNN
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', input_shape = (32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'))

model.add(tf.keras.layers.Flatten(input_shape = (32, 32, 3)))
model.add(tf.keras.layers.Dense(32, activation = 'relu'))
model.add(tf.keras.layers.Dense(3, activation = 'sigmoid'))

model.compile(optimizer = 'SGD', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(train_tensor, train_labels, epochs = 10)

#model.save("catvsdog.model")

loss, accuracy = model.evaluate(test_tensor, test_labels)
print(f'Accuracy: {accuracy * 100}% \nLoss: {loss}')

# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot loss
plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()
