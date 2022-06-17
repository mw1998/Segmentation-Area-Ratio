from PIL import Image
import tensorflow as tf 
import os
from tqdm import tqdm


from unet import Unet

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == '__main__':

    unet = Unet()

    image_ids = open('ImageName.txt','r').read().splitlines()

    if not os.path.exists("Predictions"):
        os.makedirs("Predictions")

    for image_id in tqdm(image_ids):
        image_path = "Image/"+image_id+".png"
        image = Image.open(image_path)
        image = unet.detect_image(image)
        image.save("Predictions/" + image_id + ".png")