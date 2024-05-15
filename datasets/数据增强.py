"""
作者:LegendLeo
日期:2024年03月27日

功能说明：
1、随机读取background文件夹下一张图转换为特定尺寸（可设置的参数）作为背景图；
2、随机读取bird/images/train下的n张（可设置的参数）图片（jpg）；
3、读取bird/labels/train下有对应2中图片同名的txt文件，这些是目标的标签，里面有多行多列用空格隔开的数，每行都代表一个识别的目标，其中前五个分别是标签和边界框四元数（左上角x、y和框的w、h，已经归一化到0-1），后面是多个关键点的三元数（关键点坐标x、y及其状态（0/1/2之一），xy已经归一化到0-1）；
4、根据3得到的边界框将对应图像的目标切割下来（只要目标部分，而不是整张图），再按比例缩小成指定大小（可设置的参数），n张图都要切割，然后贴进背景图的随机位置，保证每个目标互不覆盖，这样生成图片，命名为5位数字（前补0）的jpg格式；
5、根据新图中目标的位置大小，按照原来的格式生成新的txt（与4中图片同名，注意也是归一化0-1之间，每个目标的边界框和关键点都要）；
6、至此生成一张新的图和标注就完成了，可设置按上述方法连续生成x张图。
"""

import os
import random
from PIL import Image, ImageFilter

random.seed(42)

directory_images = 'bird/images/test'
directory_labels = 'bird/labels/test'
directory_background = 'background'
ratio = 0.25             # 目标缩小比例
num_files = 5           # 每张新图需要贴多少个目标
image_size = (640, 640)  # 最终图像的大小、背景图像的大小
output_directory_images = 'bird_enhance/images'
output_directory_labels = 'bird_enhance/labels'
num_of_pictures = 200       # 生成图片数量
padding = True

# 读取原标注进行变换
def process_file(file_path, ratio):
    # 返回反归一化后的标注
    with open(file_path, 'r') as file:
        lines = file.readlines()
        processed_lines = []
        for line in lines:
            data = line.strip().split()
            processed_data = []
            for i, item in enumerate(data):
                if i == 0 or (i >= 7 and (i - 7) % 3 == 0):     # 跳过类别
                    processed_data.append(item)
                else:                                           # 坐标、长宽都进行反归一化 + 缩放
                    processed_data.append(str(float(item) * 640 * ratio))
            processed_lines.append(processed_data)
        return processed_lines

# 裁切图片得到目标区域
def crop_object(image, annotation, ratio, use_padding=True):
    x, y, w, h = map(float, annotation[1:5])
    # 计算目标区域的左上角和右下角坐标
    x1, y1 = x1_t, y1_t = (x - w / 2), (y - h / 2)
    x2, y2 = x2_t, y2_t = (x + w / 2), (y + h / 2)
    padding = [0, 0, 0, 0]
    if use_padding:
        padding = [random.uniform((w + h) / 8, (w + h) / 3) for _ in range(4)]
        padding[0] = padding[0] if x1 - padding[0] > 0 else x1
        padding[1] = padding[1] if y1 - padding[1] > 0 else y1
        padding[2] = padding[2] if x2 + padding[2] < image.width else image.width - x2
        padding[3] = padding[3] if y2 + padding[3] < image.height else image.height - y2
        x1_t, y1_t, x2_t, y2_t = x1 - padding[0], y1 - padding[1], x2 + padding[2], y2 + padding[3]
    # 裁剪目标区域
    cropped_image = image.crop((x1_t, y1_t, x2_t, y2_t))
    return cropped_image, (x1, y1, x2, y2), padding

def main(num):
    files = os.listdir(directory_labels)

    # 随机选读取一个背景图像
    background_image_file = random.choice(os.listdir(directory_background))
    background_image_path = os.path.join(directory_background, background_image_file)
    background_image = Image.open(background_image_path)
    background_image = background_image.resize(image_size)

    # 创建一个空的背景图像，并以背景图像为背景
    output_image = Image.new('RGB', image_size)
    output_image.paste(background_image, (0, 0))

    # 存储已放置目标的区域、更新后的标注
    placed_areas = []
    updated_annotations = []

    for _ in range(num_files):
        random_file = random.choice(files)
        label_file_path = os.path.join(directory_labels, random_file)
        image_file_path = os.path.join(directory_images, random_file.replace('.txt', '.jpg'))
        processed_annotations = process_file(label_file_path, ratio)

        # 打开并缩小图像
        image = Image.open(image_file_path)
        image = image.resize((int(image.size[0] * ratio), int(image.size[1] * ratio)))

        # 裁剪目标区域
        cropped_images = []
        cropped_areas = []
        paddings = []
        for annotation in processed_annotations:
            cropped_image, cropped_area, padding = crop_object(image, annotation, ratio)
            cropped_images.append(cropped_image)
            cropped_areas.append(cropped_area)
            paddings.append(padding)

        # 在背景图像上放置目标
        for cropped_image, cropped_area, padding, annotation in zip(cropped_images, cropped_areas, paddings, processed_annotations):
            intersects = True
            max_attempts = 1000  # 最大尝试次数
            attempts = 0
            while intersects and attempts < max_attempts:               # 保证多个目标不覆盖彼此
                x_offset = random.randint(0, image_size[0] - cropped_image.width)       # 随机放置的左上角
                y_offset = random.randint(0, image_size[1] - cropped_image.height)
                area = (x_offset, y_offset, x_offset + cropped_image.width, y_offset + cropped_image.height)
                intersects = any(_intersects(area, placed_area) for placed_area in placed_areas)
                attempts += 1

                # 如果成功放置，则退出循环
                if not intersects:
                    output_image.paste(cropped_image, (x_offset, y_offset))
                    placed_areas.append(area)

                    # 更新所有标注并归一化
                    new_x = (x_offset + padding[0] + (cropped_image.width - padding[0] - padding[2]) / 2) / image_size[0]
                    new_y = (y_offset + padding[1] + (cropped_image.height - padding[1] - padding[3]) / 2) / image_size[1]
                    new_w = (cropped_image.width - padding[0] - padding[2]) / image_size[0]
                    new_h = (cropped_image.height - padding[1] - padding[3]) / image_size[1]
                    updated_annotation = [annotation[0], str(new_x), str(new_y), str(new_w), str(new_h)]
                    for i in range(5, len(annotation)):
                        if (i - 5) % 3 == 0 or (i - 5) == 0:
                            new_keypoint_x = float(annotation[i]) - float(annotation[1]) + float(annotation[3]) / 2 + x_offset + padding[0]
                            updated_annotation.append(str(new_keypoint_x / image_size[0]))
                        elif (i - 6) % 3 == 0 or (i - 6) == 0:
                            new_keypoint_y = float(annotation[i]) - float(annotation[2]) + float(annotation[4]) / 2 + y_offset + padding[1]
                            updated_annotation.append(str(new_keypoint_y / image_size[1]))
                        else:
                            updated_annotation.append(annotation[i])
                    updated_annotations.append(updated_annotation)
                    break
            else:
                print("Failed to place object. Too many attempts.")

    enhanced_image = output_image.filter(ImageFilter.BLUR)
    # 保存输出图像
    output_image_file = os.path.join(output_directory_images, f"{num:05}.jpg")
    output_image.save(output_image_file)

    # 保存更新后的标注信息
    output_label_file = os.path.join(output_directory_labels, f"{num:05}.txt")
    with open(output_label_file, 'w') as f:
        for annotation in updated_annotations:
            f.write(' '.join(annotation) + '\n')

    print(f"Saved image {output_image_file} and label {output_label_file}")

def _intersects(area1, area2):
    """
    检查两个区域是否相交。
    area1和area2是四元组，分别表示区域的左上角和右下角坐标。
    """
    x1, y1, x2, y2 = area1
    x3, y3, x4, y4 = area2
    return not (x2 <= x3 or x4 <= x1 or y2 <= y3 or y4 <= y1)

if __name__ == "__main__":
    for num in range(2200, 2200 + num_of_pictures):
        main(num)
