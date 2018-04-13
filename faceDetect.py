import os
import cv2
from PIL import Image, ImageDraw


def detectFaces(image_name):
    img = cv2.imread(image_name)
    face_cascade = cv2.CascadeClassifier(
        "/home/hugo/miniconda3/pkgs/opencv-3.1.0-np112py36_1/share/OpenCV/"
        "haarcascades/haarcascade_frontalface_default.xml")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    result = []
    for (x, y, width, height) in faces:
        result.append((x, y, x + width, y + height))
    return result


def saveFaces(image_name):
    faces = detectFaces(image_name)
    if faces:
        if not os.path.exists('detected_face'):
            os.mkdir('detected_face')
        save_dir = 'detected_face'
        count = 0
        for (x1, y1, x2, y2) in faces:
            file_name = os.path.join(save_dir, image_name.split('.')[0] + ".jpg")
            Image.open(image_name).crop((x1, y1, x2, y2)).save(file_name)
            count += 1


def drawFaces(image_name):
    faces = detectFaces(image_name)
    if faces:
        img = Image.open(image_name)
        draw_instance = ImageDraw.Draw(img)
        for (x1, y1, x2, y2) in faces:
            draw_instance.rectangle((x1, y1, x2, y2), outline=(255, 0, 0))
        img.save(os.path.join('detected_face', image_name.split('.')[0] + "_boudingbox.jpg"))


if __name__ == '__main__':
    # drawFaces('')
    saveFaces('delrey.jpg')
