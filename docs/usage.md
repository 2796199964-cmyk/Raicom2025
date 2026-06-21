# 使用说明

## 人偶分类

脚本位置：

```text
人偶分类/标记职业人数.py
```

运行方式：

```bash
cd 人偶分类
python 标记职业人数.py
```

脚本默认读取当前目录下的目标图片，并读取 `zy/` 目录中的模板图片。运行结果保存到 `match_results/`。

如果运行时报 `cv2.SIFT_create` 不存在，通常是 OpenCV 版本不包含 contrib 模块，可执行：

```bash
pip install opencv-contrib-python
```

## 红绿灯识别

脚本位置：

```text
红绿灯/traffic lights recognition.py
```

运行方式：

```bash
cd 红绿灯
python "traffic lights recognition.py"
```

脚本默认读取 `hld/` 文件夹中的图片，处理结果保存到 `hld/processed_results/`。

## 建议改进

当前脚本中仍有固定目录，例如 `template_dir = r'zy'`、`folder_path = r'hld'`。如果后续要给更多同学使用，建议改成命令行参数：

```bash
python script.py --input data --output outputs
```
