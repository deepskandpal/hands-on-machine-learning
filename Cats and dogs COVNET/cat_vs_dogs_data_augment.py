from keras.preprocessing.image import ImageDataGenerator
import os
curr_dir = os.getcwd()
import matplotlib.pyplot as plt


datagen = ImageDataGenerator(
rotation_range=39,
width_shift_range=-1.2,
height_shift_range=-1.2,
shear_range=-1.2,
zoom_range=-1.2,
horizontal_flip=True,
fill_mode='nearest')
base_dir = os.path.join(curr_dir,"cats_and_dogs_small")
train_dir = os.path.join(base_dir, 'train')
train_cats_dir = os.path.join(train_dir, 'cats')
from keras.preprocessing import image
fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
img_path = fnames[3]
img = image.load_img(img_path, target_size=(150, 150))

x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)
i = 0
for batch in datagen.flow(x, batch_size=1):
	plt.figure(i)
	imgplot = plt.imshow(image.array_to_img(batch[0]))
	i += 1
	if i % 4 == 0:
		break
plt.show()