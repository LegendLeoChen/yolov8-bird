"""
作者:LegendLeo
日期:2024年03月27日

功能说明：
这个脚本用于标注图像中的目标位置和关键点，并自动保存标注结果到新文件中。具体功能包括：
1. 读取图像和标注文件，并在图像上框出第一个目标边界框。
2. 用户按键交互式地标注图像。按下数字键可以切换到下一个目标边界框，并在框出该目标后自动保存分类，0——停留，1——飞行。
3. 自动将原图像复制到一个新目录，并在同名的文本文件中保存目标的分类和关键点坐标信息。
"""

import os
import shutil
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk

def read_annotation(file_path, image_width, image_height):          # 读取原始标注文件
    annotations = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = line.split()
            if len(data) < 10:
                continue
            category = int(data[0])
            x, y, w, h = map(float, data[1:5])
            x *= image_width
            y *= image_height
            w *= image_width
            h *= image_height
            keypoints = []
            statuses = []
            for i in range(5, len(data), 3):
                keypoint_x = float(data[i])
                keypoint_y = float(data[i + 1])
                keypoints.append((keypoint_x, keypoint_y))
                statuses.append(int(data[i + 2]))
            annotations.append((category, (x, y, w, h), keypoints, statuses))
    return annotations


def draw_annotation(image, annotation):             # 绘制标注的边界框
    draw = ImageDraw.Draw(image)
    category, bbox, keypoints, statuses = annotation
    x, y, w, h = bbox
    draw.rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], outline='red')
    return image


def on_key_pressed(event):                  # 检测键值，记录姿态分类
    global current_target, annotations, image, photo, canvas, classifications

    try:
        if event.char.isdigit():
            current_target += 1
            classifications.append(event.char)
            if current_target < len(annotations):
                annotated_image = image.copy()
                draw = ImageDraw.Draw(annotated_image)
                draw_annotation(annotated_image, annotations[current_target])
                photo = ImageTk.PhotoImage(annotated_image)
                canvas.itemconfig(image_item, image=photo)
            else:
                save_annotation(image_files[current_image_index], annotations, classifications)
                next_image()
    except AttributeError:
        pass

def next_image():               # 标注下一张图
    global image_files, label_files, current_image_index, image, annotations, current_target, photo, image_item, classifications

    classifications = []
    current_image_index += 1
    if current_image_index < len(image_files):
        img_file = image_files[current_image_index]
        label_file = label_files[current_image_index]

        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, label_file)

        image = Image.open(img_path)
        image_width, image_height = image.size

        annotations = read_annotation(label_path, image_width, image_height)

        current_target = 0

        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        draw_annotation(annotated_image, annotations[current_target])
        photo = ImageTk.PhotoImage(annotated_image)
        canvas.itemconfig(image_item, image=photo)
    else:
        print("All images annotated.")
        root.quit()

def save_annotation(image_file, annotations, classifications):          # 保存新的标注 姿态分类 + 关键点（9*2）
    img_name, _ = os.path.splitext(image_file)
    new_img_path = os.path.join(new_image_dir, image_file)
    shutil.copy(os.path.join(image_dir, image_file), new_img_path)

    txt_name = img_name + ".txt"
    new_txt_path = os.path.join(new_label_dir, txt_name)
    with open(new_txt_path, 'w') as txt_file:
        for i, classification in enumerate(classifications):
            txt_file.write(classification)
            for keypoint, status in zip(annotations[i][2], annotations[i][3]):
                if status == 0:
                    txt_file.write(" 0 0")
                else:
                    txt_file.write(f" {keypoint[0]} {keypoint[1]}")
            txt_file.write("\n")


def main():
    global current_target, annotations, image, photo, canvas, image_item, image_files, label_files, image_dir, label_dir, current_image_index, classifications, new_image_dir, new_label_dir

    image_dir = 'bird/images/train'
    label_dir = 'bird/labels/train'
    new_image_dir = 'bird_pose/images'
    new_label_dir = 'bird_pose/labels'

    if not os.path.exists(new_image_dir):
        os.makedirs(new_image_dir)
    if not os.path.exists(new_label_dir):
        os.makedirs(new_label_dir)

    image_files = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and len(f) >= 1])
    label_files = sorted([f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f)) and len(f) >= 1])

    current_image_index = 0
    classifications = []

    root = tk.Tk()
    root.title("Image Annotation")
    root.bind('<Key>', on_key_pressed)              # 防止按键检测和图像显示阻塞

    canvas = tk.Canvas(root, width=640, height=640)
    canvas.pack()

    img_file = image_files[current_image_index]
    label_file = label_files[current_image_index]

    img_path = os.path.join(image_dir, img_file)
    label_path = os.path.join(label_dir, label_file)

    image = Image.open(img_path)
    image_width, image_height = image.size

    annotations = read_annotation(label_path, image_width, image_height)        # 读取标注

    current_target = 0

    annotated_image = image.copy()
    draw_annotation(annotated_image, annotations[current_target])           # 画标注
    photo = ImageTk.PhotoImage(annotated_image)
    image_item = canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    root.mainloop()

if __name__ == "__main__":
    main()
