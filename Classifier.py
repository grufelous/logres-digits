from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

def display_training_images(n):
    for i, (img, lbl) in enumerate(zip(digits.data[0:n], digits.target[0:n])):
        plt.subplot(1, n, i+1)
        plt.imshow(np.reshape(img, (8,8)), cmap=plt.cm.gray)
        plt.title('Training img: %i\n' % lbl, fontsize=7)
        

digits = load_digits()

print("Image data shape: ",digits.data.shape)
print("Label data shape: ",digits.target.shape)

plt.figure(figsize=(20,4))

display_training_images(9)

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=0)

logReg = LogisticRegression()
logReg.fit(x_train, y_train)

logReg.predict(x_test[0].reshape(1,-1))
logReg.predict(x_test[0:10])

predictions = logReg.predict(x_test)
score = logReg.score(x_test, y_test)
print(score)