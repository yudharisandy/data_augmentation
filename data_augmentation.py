import keras
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.01,
        height_shift_range=0.01,
        shear_range=0.2,
        zoom_range=0.5,
        horizontal_flip=True,
        fill_mode='nearest')

b=0
number_of_image=15

for img in [f for f in os.listdir('data') if f.endswith('jpg')  ]:
    img = load_img('data/'+ img )  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `data/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                            save_to_dir='data', save_prefix='augmentation_'+ str(b), save_format='jpg'):
        i += 1
        if i > number_of_image:
            break  # otherwise the generator would loop indefinitely
    b +=1



    