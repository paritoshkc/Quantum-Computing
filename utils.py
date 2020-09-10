import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from resizeimage import resizeimage
from PIL import Image, ImageOps

def get_MNIST_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return x_train



def get_Cat_320_image():
    return Image.open("cat_320.png").convert('LA')

def get_Cat_image():
    return Image.open("cat.png").convert('LA')

def large_image_normalization(images,w,h):
    image=np.array([])
   
    for y in range(h-32,h):
         for x in range(w-32,w):
            image=np.append(image,images.getpixel((x,y))[0])
    genimg = image.reshape((32,32))
    image = image.flatten()
    # change type
    image = image.astype('float64')
    # Normalization(0~pi/2)
    image /= 255.0
    generated_image=image
    # generated_image = np.arcsin(image)
    print(generated_image)
    image=images
    return generated_image



def image_normalization(image,size,show):
    image = resizeimage.resize_cover(image, [size, size])
    w, h = size, size
    image = np.array([[image.getpixel((x,y))[0] for x in range(w)] for y in range(h)])
    
    # display the image
    if show:
        genimg = image.reshape((size,size))
        plt.imshow(genimg, cmap='gray', vmin=0, vmax=255)
        plt.show()
    
    image = image.flatten()
    image = image.astype('float64')
    image /= 255.0
    generated_image = np.arcsin(image)
    # print(generated_image)
    return generated_image

def generate_image(size):
    w, h = size, size
    data = np.zeros((h, w, 3), dtype=np.uint8)
    w_bound=int(w/2)
    h_bound=int(h/2)
    data[0:w_bound, 0:h_bound] = [255, 0, 0] # red patch in upper left
    data[0:w_bound, h_bound+1:h] = [0, 255, 0] # red patch in upper left
    data[w_bound+1:w, 0:h_bound] = [0, 0, 255] # red patch in upper left
    data[w_bound+1:w, h_bound+1:h] = [128, 128, 128] # red patch in upper left

    imgq = Image.fromarray(data, 'RGB')
    imgq.save('my.png')
    return imgq.convert('LA')

def get_image_pixel_value(image,size):
    img_arr= np.array([[image.getpixel((x,y))[0] for x in range(size)] for y in range(size)])
    img_arr = img_arr.flatten()
    return img_arr

def get_count_of_pixel(arr1,arr2):
    same=0
    notsame=0
    for i in range(len(arr1)):
        if arr1[i]==arr2[i]:
            same+=1
        else:
            notsame+=1
    return (same,notsame)

    
