import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import matplotlib
import os
import math
import random
from PIL import Image, ImageDraw, ImageFont

matplotlib.use('TkAgg')  # 设置TkAgg后端
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 读取模板图像目录(将 标记职业人数.py 移动到与zy同一目录)
template_dir = r'zy'  # 模板图像目录
target_dir = r'./'  # 目标图像目录

def putChineseText(img, text, position, textColor=(0, 255, 0), textSize=20):
    """在OpenCV图像上绘制中文文本"""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("simhei.ttf", textSize, encoding="utf-8")
    except:
        font = ImageFont.load_default()

    draw.text(position, text, textColor, font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def calculate_angles(points):
    """计算四边形四个角的度数"""
    angles = []
    n = len(points)
    for i in range(n):
        p1 = points[i][0]
        p2 = points[(i + 1) % n][0]
        p3 = points[(i + 2) % n][0]

        v1 = p1 - p2
        v2 = p3 - p2

        angle = np.degrees(math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0]))
        angle = angle + 360 if angle < 0 else angle
        angles.append(angle)
    return angles


def calculate_area(points):
    """计算四边形面积"""
    x = points[:, 0, 0]
    y = points[:, 0, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def is_convex(points):
    """检查四边形是否凸的（无交叉）"""
    n = len(points)
    if n != 4:
        return False

    def cross_product(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    pts = [points[i][0] for i in range(4)]
    cross1 = cross_product(pts[0], pts[1], pts[2])
    cross2 = cross_product(pts[1], pts[2], pts[3])
    cross3 = cross_product(pts[2], pts[3], pts[0])
    cross4 = cross_product(pts[3], pts[0], pts[1])

    return (cross1 * cross2 > 0) and (cross2 * cross3 > 0) and (cross3 * cross4 > 0)


def is_valid_match(angles, area):
    """检查匹配是否有效"""
    min_angle = min(angles)
    max_angle = max(angles)
    if min_angle < 30 or max_angle > 120 or area < 10000:
        return False
    return True


def clip_coordinates(points, img_width, img_height):
    """裁剪坐标到图像边界内"""
    clipped = []
    for pt in points:
        x = max(0, min(pt[0][0], img_width - 1))
        y = max(0, min(pt[0][1], img_height - 1))
        clipped.append([[x, y]])
    return np.array(clipped, dtype=np.float32)



# 创建结果保存目录
output_dir = os.path.join(target_dir, 'match_results')
os.makedirs(output_dir, exist_ok=True)

# 获取模板目录下所有JPG文件
template_files = [f for f in os.listdir(template_dir) if f.lower().endswith('.jpg')]
print(f"找到 {len(template_files)} 个模板图像")

# 获取目标目录下所有JPG文件
target_files = [f for f in os.listdir(target_dir) if f.lower().endswith('.jpg')]
print(f"找到 {len(target_files)} 个目标图像")

# 创建SIFT检测器
sift = cv2.SIFT_create()

# 设置FLANN匹配器
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# ====== 新增：预先存储所有模板特征 ======
print("\n正在预处理模板图像...")
template_features = []
for template_file in template_files:
    template_path = os.path.join(template_dir, template_file)
    img = cv2.imread(template_path)
    if img is None:
        print(f"无法读取模板图像: {template_path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)

    if des is not None:
        template_features.append({
            'filename': template_file,
            'keypoints': kp,
            'descriptors': des,
            'size': gray.shape  # (height, width)
        })
        print(f"  已预处理模板: {template_file} (特征点: {len(kp)})")
    else:
        print(f"  无法提取模板特征: {template_file}")

print(f"\n完成模板预处理，有效模板数量: {len(template_features)}")

# 处理每个目标图像
for target_file in target_files:
    target_path = os.path.join(target_dir, target_file)
    print(f"\n处理目标图像: {target_file}")

    # 加载目标图像
    img2 = cv2.imread(target_path)
    if img2 is None:
        print(f"无法读取目标图像: {target_file}")
        continue

    # 转换为灰度图
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 检测目标图像的关键点和描述符
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des2 is None:
        print(f"无法提取目标图像特征: {target_file}")
        continue

    # 存储所有有效匹配信息
    valid_matches = []
    match_details = []

    # 处理每个模板特征
    for template in template_features:
        # print(f"  匹配模板: {template['filename']}")

        # 执行KNN匹配
        matches = flann.knnMatch(template['descriptors'], des2, k=2)

        # 筛选优质匹配点
        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)

        # 计算单应性矩阵
        M, dst = None, None
        if len(good_matches) >= 4:
            src_pts = np.float32([template['keypoints'][m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                h, w = template['size']
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

        # 检查是否找到有效匹配
        if M is None or dst is None:
            continue

        # 裁剪坐标到图像边界内
        img_height, img_width = img2.shape[:2]
        clipped_dst = clip_coordinates(dst, img_width, img_height)

        # 检查标记框有效性
        convex = is_convex(clipped_dst)
        angles = calculate_angles(clipped_dst)
        area = calculate_area(clipped_dst)
        valid = is_valid_match(angles, area) and convex

        if not valid:
            continue

        # 存储匹配信息
        match_info = {
            'template': template['filename'],
            'points': clipped_dst,
            'good_matches': good_matches,
            'angles': angles,
            'area': area,
            'convex': convex
        }
        valid_matches.append(match_info)

        # 添加匹配详细信息
        detail = (f"模板: {template['filename']}\n"
                  f"匹配点数量: {len(good_matches)}\n"
                  f"标记框面积: {area:.2f} 像素\n"
                  f"四个角度数: {angles[0]:.1f}°, {angles[1]:.1f}°, {angles[2]:.1f}°, {angles[3]:.1f}°")
        match_details.append(detail)

    # 创建标记后的目标图像
    img2_marked = img2.copy()

    # 准备不同颜色的数组
    color = (255, 255, 255)

    # 为每个有效匹配绘制标记框和名称
    for i, match_info in enumerate(valid_matches):
        img2_marked = cv2.polylines(img2_marked, [np.int32(match_info['points'])], True, color, 3, cv2.LINE_AA)
        center = np.mean(match_info['points'], axis=0)[0]
        img2_marked = putChineseText(img2_marked, f"模板: {match_info['template']}",
                                     (int(center[0]) - 50, int(center[1]) - 20), color, 80)

    # 转换颜色空间
    img2_marked_rgb = cv2.cvtColor(img2_marked, cv2.COLOR_BGR2RGB)

    # 创建可视化结果
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(img2_marked_rgb)
    ax.set_title(f'目标图像: {target_file} - 找到 {len(valid_matches)} 个模板匹配', fontsize=14)
    ax.axis('off')

    # 添加信息框
    info_text = f"目标图像: {target_file}\n"
    info_text += f"匹配到的模板数量: {len(valid_matches)}\n"

    if valid_matches:
        info_text += f"匹配到的模板名称: {', '.join([m['template'] for m in valid_matches])}\n\n"
        info_text += "\n\n".join(match_details)
    else:
        info_text += "未找到有效匹配\n"
        info_text += f"已尝试匹配 {len(template_features)} 个模板"

    fig.text(0.5, -0.01, info_text, ha='center', va='top', fontsize=10,
             bbox=dict(facecolor='lightyellow', alpha=0.5))

    # 调整布局
    plt.subplots_adjust(bottom=0.3)

    # 保存结果
    result_filename = f"match_result_{os.path.splitext(target_file)[0]}.jpg"
    result_path = os.path.join(output_dir, result_filename)
    plt.tight_layout()
    plt.savefig(result_path, dpi=150, bbox_inches='tight')
    plt.close()


    if valid_matches:
        print(f"目标图像 {target_file} 的结果已保存，共找到 {len(valid_matches)} 个有效匹配")
    else:
        print(f"目标图像 {target_file} 的结果已保存，未找到有效匹配")

print(f"\n所有处理完成，结果保存在: {output_dir}")
