import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

image_dir=r'E:\DUK\SEM 3\mini project\violence data'
non_violence_images=os.listdir(image_dir+ '/non_violence')
violence_images=os.listdir(image_dir+ '/violence')
print("--------------------------------------\n")

print('The length of Non Violence image is',len(non_violence_images))
print('The length of Violence image is',len(violence_images))
print("--------------------------------------\n")


dataset=[]
label=[]
img_siz=(128,128)


for i , image_name in tqdm(enumerate(non_violence_images),desc="Non Violence"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/non_violence/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(0)
        
        
for i ,image_name in tqdm(enumerate(violence_images),desc="Violence"):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'/violence/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(1)
        
        
dataset=np.array(dataset)
label = np.array(label)

print("--------------------------------------\n")
print('Dataset Length: ',len(dataset))
print('Label Length: ',len(label))
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Train-Test Split")
x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2,random_state=42)
print("--------------------------------------\n")

print("--------------------------------------\n")
print("Normalaising the Dataset. \n")

# x_train=x_train.astype('float')/255
# x_test=x_test.astype('float')/255 

# Same step above is implemented using tensorflow functions.

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

print("--------------------------------------\n")


model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
print("--------------------------------------\n")
model.summary()
print("--------------------------------------\n")

model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])


print("--------------------------------------\n")
print("Training Started.\n")
history=model.fit(x_train,y_train,epochs=5,batch_size =128,validation_split=0.1)
print("Training Finished.\n")
print("--------------------------------------\n")


# Plot and save accuracy
plt.plot(history.epoch,history.history['accuracy'], label='accuracy')
plt.plot(history.epoch,history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig(r'E:\DUK\SEM 3\mini project\results\violence_sample_accuracy_plot.png')


# Clear the previous plot
plt.clf()

# Plot and save loss
plt.plot(history.epoch,history.history['loss'], label='loss')
plt.plot(history.epoch,history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig(r'E:\DUK\SEM 3\mini project\results\violence_sample_loss_plot.png')


print("--------------------------------------\n")
print("Model Evalutaion Phase.\n")
loss,accuracy=model.evaluate(x_test,y_test)
print(f'Accuracy: {round(accuracy*100,2)}')
print("--------------------------------------\n")
y_pred=model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int)
print('classification Report\n',classification_report(y_test,y_pred))
print("--------------------------------------\n")

model.save(r'E:\DUK\SEM 3\mini project\results\model\cnn_violence2.h5')

print("--------------------------------------\n")
print("Model Prediction.\n")

# model = tf.keras.models.load_model('CNN/tumor_detection/results/model/cnn_tumor.h5')

def make_prediction(img,model):
    # img=cv2.imread(img)
    img=Image.fromarray(img)
    img=img.resize((128,128))
    img=np.array(img)
    input_img = np.expand_dims(img, axis=0)
    res = model.predict(input_img)
    if res:
        print("Violence Detected")
    else:
        print("No Violence")
    return res
        
make_prediction(cv2.imread(r'E:\DUK\SEM 3\mini project\violence data\violence\6.jpg'),model)
print("--------------------------------------\n")
make_prediction(cv2.imread(r'E:\DUK\SEM 3\mini project\violence data\non_violence\5.jpg'),model)
print("--------------------------------------\n")



