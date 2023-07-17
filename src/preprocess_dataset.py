import os
import cv2
import time
from imutils import paths
from mtcnn import MTCNN

start = time.time()

detector = MTCNN()


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
    image = cv2.imread(imagePath)
    faces = detector.detect_faces(image)
    for face in faces:
        x, y, w, h = face['box']
        face_crop = image[y:y+h, x:x+w]
        # Resize the cropped face to 112x112 pixels
        resized_face = cv2.resize(face_crop, (112, 112))
        # Save the cropped and resized face to the output folder
        cv2.imwrite(imagePath, resized_face)

end = time.time()
print("Execution time: {:.2f} seconds".format(end - start))

