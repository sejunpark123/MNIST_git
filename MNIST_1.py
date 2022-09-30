#이미지 학습 코드
from cgi import test
from codecs import EncodedFile
from pyexpat import model
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
mnist = tf.keras.datasets.mnist
import cv2
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
score = model.evaluate(x_test, y_test)

#이미지 불러와 출력
gray = cv2.imread("C:/Users/MSI/AppData/Local/Programs/Python/Python38/Scripts/DataSet//1.png", cv2.IMREAD_GRAYSCALE)
plt.imshow(gray)
plt.show()

#이미지 사이즈 변경
gray = cv2.resize(255-gray, (28,28))
test_num = gray.flatten() / 255.0
test_num = test_num.reshape((-1, 28, 28, 1))

predictions = np.argmax(model.predict(test_num), axis=-1)

#이미지 숫자 테스트
print('The Answer is: ', predictions)