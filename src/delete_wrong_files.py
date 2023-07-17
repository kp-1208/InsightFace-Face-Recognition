import os
import cv2
import time
from imutils import paths
from mtcnn import MTCNN

start = time.time()

detector = MTCNN()

target_resolution = 37632  # Provide your desired resolution here
imagePaths = list(paths.list_images("/home/kp1208/Desktop/InsightFace-Face-Recognition/datasets/train"))
print(imagePaths)

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]
    print(name)
    '''input_path = os.path.join(folder, filename)
    output_path = os.path.join(folder, filename)'''
    try:
        image = cv2.imread(imagePath)
        if image.size != target_resolution:
            os.remove(imagePath)
            print(f"Deleted: {name}")
    except Exception as e:
        print(f"Error processing {name}: {str(e)}")


