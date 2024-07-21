import glob
import io
import pathlib
import random

import nude_detector
from wdv3_jax_worker import ImageTaggerWorker

list_sfw = glob.glob('test_image/SFW/*.jpg')
list_sfw.extend(glob.glob('test_image/SFW/*.png'))

list_nsfw = glob.glob('test_image/NSFW/*.jpg')
list_nsfw.extend(glob.glob('test_image/NSFW/*.png'))

list_images_to_test = list_nsfw + list_sfw
random.shuffle(list_images_to_test)

wdv3_Worker = ImageTaggerWorker()
model = nude_detector.load_model(
    "/home/taruu/PycharmProjects/nude-check-tests/models/nsfw_models/mobilenet_v2_140_224/saved_model.h5")

for image_path in list_images_to_test:
    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()
        img_file = io.BytesIO(image_bytes)
        caption, tag_list, ratings, character, general = wdv3_Worker.get_image_marks(img_file)
        result = nude_detector.classify(model, img_file)
        print()
        print(image_path)
        print(ratings)
        print(result)
