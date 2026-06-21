import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

# 设置红绿灯图片文件夹路径
folder_path = r"hld"

# 创建输出目录
output_dir = os.path.join(folder_path, "processed_results")
os.makedirs(output_dir, exist_ok=True)
print(f"处理后的图片将保存在: {output_dir}")


def detect_traffic_light(image_path):
    """
    检测红绿灯的颜色
    :param image_path: 图片文件路径
    :return: 检测结果（红灯、黄灯、绿灯）和处理后的图片
    """
    # 1. 读取图片
    img = cv2.imread(image_path)

    # 如果图片读取失败，返回错误
    if img is None:
        print(f"无法读取图片: {image_path}")
        return "error", None

    # 2. 转换为HSV颜色空间（更易识别颜色）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. 定义颜色范围（红、黄、绿）
    # 红色范围（有两个区间，因为红色在HSV空间中跨越0度）
    lower_red1 = np.array([0, 100, 150])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 150])
    upper_red2 = np.array([180, 255, 255])

    # 黄色范围
    lower_yellow = np.array([22, 120, 140])  # 调整黄色范围，减少与绿色的重叠
    upper_yellow = np.array([28, 255, 255])

    # 绿色范围
    lower_green = np.array([40, 100, 100])  # 调整绿色范围，减少与黄色的重叠
    upper_green = np.array([80, 255, 255])

    # 4. 创建颜色掩膜
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)  # 合并红色掩膜
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # 5. 查找颜色轮廓
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. 存储检测到的灯信息（位置、颜色、面积）
    lights = []

    # 处理红灯
    for contour in contours_red:
        area = cv2.contourArea(contour)
        if area > 20:  # 忽略太小的区域
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])  # 中心点x坐标
                cy = int(M["m01"] / M["m00"])  # 中心点y极坐标
                lights.append((cx, cy, "red", area))

    # 处理黄灯
    for contour in contours_yellow:
        area = cv2.contourArea(contour)
        if area > 20:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                lights.append((cx, cy, "yellow", area))

    # 处理绿灯
    for contour in contours_green:
        area = cv2.contourArea(contour)
        if area > 20:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                lights.append((cx, cy, "green", area))

    # 7. 如果没有检测到灯，返回未知
    if not lights:
        return "unknown", None

    # 8. 确定方向（水平或垂直）
    # 获取所有灯的中心坐标
    xs = [light[0] for light in lights]
    ys = [light[1] for light in lights]

    # 计算X和Y的范围
    x_range = max(xs) - min(xs) if xs else 0
    y_range = max(ys) - min(ys) if ys else 0

    # 方向判断：如果X方向范围大于Y方向，则为水平方向，否则为垂直方向
    direction = "horizontal" if x_range > y_range else "vertical"
    print(f"方向: {direction}")

    # 9. 按方向排序灯的位置
    if direction == "horizontal":
        # 水平方向：从左到右排序
        sorted_lights = sorted(lights, key=lambda x: x[0])
    else:
        # 垂直方向：从上到下排序
        sorted_lights = sorted(lights, key=lambda x: x[1])

    # 10. 找出主灯（面积最大的灯）
    main_light = max(sorted_lights, key=lambda x: x[3])
    print(f"主灯: 位置{main_light[0]},{main_light[1]}, 颜色{main_light[2]}, 面积{main_light[3]}")

    # 11. 找到主灯在排序列表中的位置索引
    try:
        main_index = sorted_lights.index(main_light)
        print(f"主灯索引: {main_index}")
    except:
        return "unknown", None

    # 12. 应用识别规则
    # 规则1: 第一个灯亮且识别为红灯 → 红灯
    if main_index == 0 and main_light[2] == "red":
        result = "red"

    # 规则2: 第二个灯亮（无论识别红还是黄） → 黄灯
    elif main_index == 1:
        if main_light[2] == "red":
            # 检查是否可能是红灯（例如，红色像素明显多于其他颜色）
            red_pixels = cv2.countNonZero(mask_red)
            yellow_pixels = cv2.countNonZero(mask_yellow)
            green_pixels = cv2.countNonZero(mask_green)

            # 如果红色像素远多于其他颜色，可能是红灯
            if red_pixels > yellow_pixels * 2 and red_pixels > green_pixels * 2:
                result = "red"
            else:
                result = "yellow"
        else:
            result = "yellow"

    # 规则3: 第三个灯亮（无论识别绿还是黄） → 绿灯
    elif main_index == 2:
        result = "green"

    # 不符合规则的情况返回检测到的颜色
    else:
        result = main_light[2]

    # 13. 额外检查：如果主灯是绿色但被识别为黄灯，强制校正为绿灯
    if result == "yellow" and main_light[2] == "green":
        print("额外校正: 黄灯结果但主灯是绿色 → 绿灯")
        result = "green"

    # 14. 额外检查：如果主灯是黄色但被识别为绿灯，强制校正为黄灯
    if result == "green" and main_light[2] == "yellow":
        print("额外校正: 绿灯结果但主灯是黄色 → 黄灯")
        result = "yellow"

    print(f"识别结果: {result}")

    # 15. 在图片上添加识别结果文字
    # 使用PIL打开图片（支持中文）
    pil_img = Image.open(image_path)

    # 创建绘图对象
    draw = ImageDraw.Draw(pil_img)

    # 设置文字和颜色
    if result == "red":
        text = "红灯"
        text_color = (255, 0, 0)  # 红色
    elif result == "yellow":
        text = "黄灯"
        text_color = (255, 255, 0)  # 黄色
    elif result == "green":
        text = "绿灯"
        text_color = (0, 255, 0)  # 绿色
    else:
        text = "未知"
        text_color = (255, 255, 255)  # 白色

    # 设置字体（宋体，100磅）
    try:
        # 尝试查找宋体
        font_path = "C:/Windows/Fonts/simsun.ttc"
        if not os.path.exists(font_path):
            # 如果找不到宋体，尝试其他中文字体
            font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体
        font = ImageFont.truetype(font_path, 100)
        print(f"使用字体: {font_path}")
    except:
        # 字体加载失败时使用默认字体
        font = ImageFont.load_default()
        print("使用默认字体")

    # 获取图片和文字的尺寸
    img_width, img_height = pil_img.size

    # 获取文字尺寸
    try:
        # 对于新版Pillow
        left, top, right, bottom = font.getbbox(text)
        text_width = right - left
        text_height = bottom - top
    except:
        try:
            # 对于旧版Pillow
            text_width, text_height = font.getsize(text)
        except:
            # 如果都失败，使用估计值
            text_width = len(text) * 50  # 粗略估计
            text_height = 100

    # 计算文字位置（居中）
    position = (
        (img_width - text_width) // 2,
        (img_height - text_height) // 2
    )

    # 添加黑色描边（使文字更清晰）
    outline_color = (0, 0, 0)  # 黑色
    outline_width = 5  # 加粗描边

    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:  # 不在中心位置绘制
                try:
                    draw.text(
                        (position[0] + dx, position[1] + dy),
                        text,
                        font=font,
                        fill=outline_color
                    )
                except:
                    # 如果添加文字失败，跳过
                    pass

    # 添加主文字
    try:
        draw.text(position, text, font=font, fill=text_color )
    except:
        # 如果添加文字失败，返回原始图片
        pass

    return result, pil_img


def process_all_images():
    """
    处理文件夹中的所有图片
    """
    # 获取所有图片文件（支持常见格式）
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not image_files:
        print("文件夹中没有找到图片文件")
        return

    print(f"找到 {len(image_files)} 张图片需要处理")

    # 处理每张图片
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        print(f"\n处理进度: {i + 1}/{len(image_files)} - {image_file}")

        # 检测红绿灯
        result, result_image = detect_traffic_light(image_path)

        # 保存结果
        if result_image:
            output_path = os.path.join(output_dir, f"processed_{image_file}")
            result_image.save(output_path)
            print(f"  结果: {result} | 保存到: {os.path.basename(output_path)}")
        else:
            print(f"  处理失败")


if __name__ == "__main__":
    # 处理所有图片
    process_all_images()
    print("\n处理完成!")