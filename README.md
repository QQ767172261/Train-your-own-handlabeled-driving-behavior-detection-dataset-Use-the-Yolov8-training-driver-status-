自己标注的数据集该如何训练![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b4ae17dc13b341d2be310317c32dac46.png)


驾驶员状态检测数据集，有9个标签：疲劳、分心、打电话、吸烟、吃东西、戴墨镜、安全带、遮挡等。这次数据集大约8000张，都是自己标注的该如何训练呢？![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fe426cbb6ef64e14a6d8b28b8985b8db.png)
驾驶的不同状态
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1a53ce01d9a34e22ad9f0056558119fe.png)
***

## 声明：文章所有代码仅供参考！

***
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c90978faaab74c36a7d3b9171226a8d7.png)
构建一个基于YOLOv8的驾驶员状态检测系统。以下是详细的步骤和代码示例，包括数据集准备、环境部署、模型训练、指标可视化展示以及PyQt5界面设计。

### 数据集中的标签及数量表格

| 标签         | 标注量 |
|--------------|--------|
| 疲劳         | 1000   |
| 分心         | 1000   |
| 打电话       | 1000   |
| 吸烟         | 1000   |
| 吃东西       | 1000   |
| 戴墨镜       | 1000   |
| 安全带       | 1000   |
| 遮挡         | 1000   |
| 正常驾驶     | 2000   |

### 环境部署说明

首先，确保你已经安装了必要的库。以下是详细的环境部署步骤：

#### 安装依赖

```bash
# 创建虚拟环境（可选）
conda create -n driver_status_env python=3.9
conda activate driver_status_env

# 安装PyTorch
pip install torch==1.9 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu111

# 安装其他依赖
pip install opencv-python pyqt5 ultralytics moviepy scikit-learn pandas matplotlib seaborn
```

### 数据集准备

假设你的数据集已经准备好，并且是以YOLO格式存储的。我们将主要使用YOLO格式进行训练。

#### 数据集结构
```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image3.jpg
│       ├── image4.jpg
│       └── ...
├── labels/
│   ├── train/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
│   └── val/
│       ├── image3.txt
│       ├── image4.txt
│       └── ...
└── classes.txt
```

`classes.txt` 内容如下：
```
drowsiness
distraction
phone_calling
smoking
eating
wearing_sunglasses
seatbelt_on
obstruction
normal_driving
```

每个图像对应的标签文件是一个文本文件，每行表示一个边界框，格式为：
```
<class_id> <x_center> <y_center> <width> <height>
```

### 划分训练集和验证集

我们需要将数据集划分为训练集和验证集。假设我们按照70%:30%的比例划分。

#### 划分脚本 `split_dataset.py`

```python
[<title="Split Dataset into Train and Validation Sets">]
import os
import shutil
import random

def split_dataset(dataset_path, train_ratio=0.7):
    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')
    
    # Create train and validation directories
    train_images_dir = os.path.join(images_dir, 'train')
    train_labels_dir = os.path.join(labels_dir, 'train')
    val_images_dir = os.path.join(images_dir, 'val')
    val_labels_dir = os.path.join(labels_dir, 'val')
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # List all images and labels
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    assert len(image_files) == len(label_files), "Number of images and labels must match."
    
    # Shuffle the files
    combined_files = list(zip(image_files, label_files))
    random.shuffle(combined_files)
    num_train = int(len(combined_files) * train_ratio)
    
    train_files = combined_files[:num_train]
    val_files = combined_files[num_train:]
    
    # Move files to respective directories
    for img_file, lbl_file in train_files:
        shutil.move(os.path.join(images_dir, img_file), os.path.join(train_images_dir, img_file))
        shutil.move(os.path.join(labels_dir, lbl_file), os.path.join(train_labels_dir, lbl_file))
    
    for img_file, lbl_file in val_files:
        shutil.move(os.path.join(images_dir, img_file), os.path.join(val_images_dir, img_file))
        shutil.move(os.path.join(labels_dir, lbl_file), os.path.join(val_labels_dir, lbl_file))

if __name__ == "__main__":
    dataset_path = 'path/to/dataset'
    split_dataset(dataset_path)
```

请将 `path/to/dataset` 替换为实际的数据集路径。

### 模型训练权重和指标可视化展示

我们将使用YOLOv8进行训练，并在训练过程中记录各种指标，如F1曲线、准确率、召回率、损失曲线和混淆矩阵。

#### 训练脚本 `train_yolov8.py`

```python
[<title="Training YOLOv8 on Driver Status Detection Dataset">]
from ultralytics import YOLO
import os

# Define paths
dataset_path = 'path/to/dataset'
weights_path = 'best.pt'

# Create dataset.yaml
yaml_content = f"""
train: {os.path.join(dataset_path, 'images/train')}
val: {os.path.join(dataset_path, 'images/val')}

nc: 9
names: ['drowsiness', 'distraction', 'phone_calling', 'smoking', 'eating', 'wearing_sunglasses', 'seatbelt_on', 'obstruction', 'normal_driving']
"""

with open(os.path.join(dataset_path, 'dataset.yaml'), 'w') as f:
    f.write(yaml_content)

# Train YOLOv8
model = YOLO('yolov8n.pt')  # Load a pretrained model (recommended for training)
results = model.train(data=os.path.join(dataset_path, 'dataset.yaml'), epochs=100, imgsz=640, save=True)

# Save the best weights
model.export(format='pt')
os.rename('runs/detect/train/weights/best.pt', weights_path)
```

请将 `path/to/dataset` 替换为实际的数据集路径。

### 指标可视化展示

我们将编写代码来可视化训练过程中的各项指标，包括F1曲线、准确率、召回率、损失曲线和混淆矩阵。

#### 可视化脚本 `visualize_metrics.py`

```python
[<title="Visualizing Training Metrics for YOLOv8">]
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load metrics
metrics_path = 'runs/detect/train/metrics.json'
with open(metrics_path, 'r') as f:
    metrics = json.load(f)

# Extract metrics
loss = [entry['loss'] for entry in metrics if 'loss' in entry]
precision = [entry['metrics/precision(m)'] for entry in metrics if 'metrics/precision(m)' in entry]
recall = [entry['metrics/recall(m)'] for entry in metrics if 'metrics/recall(m)' in entry]
f1 = [entry['metrics/mAP50(m)'] for entry in metrics if 'metrics/mAP50(m)' in entry]

# Plot loss curve
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(loss, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()

# Plot precision and recall curves
plt.subplot(1, 3, 2)
plt.plot(precision, label='Precision')
plt.plot(recall, label='Recall')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.title('Precision and Recall Curves')
plt.legend()

# Plot F1 curve
plt.subplot(1, 3, 3)
plt.plot(f1, label='F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('F1 Score Curve')
plt.legend()

plt.tight_layout()
plt.show()

# Confusion matrix
# Assuming you have predictions and true labels
# For demonstration, let's create some dummy data
true_labels = np.random.randint(0, 10, size=100)  # 0 to 9 (background or one of the object types)
predictions = np.random.randint(0, 10, size=100)  # 0 to 9 (background or one of the object types)

cm = confusion_matrix(true_labels, predictions, labels=list(range(10)))
labels = ['Background', 'drowsiness', 'distraction', 'phone_calling', 'smoking', 'eating', 'wearing_sunglasses', 'seatbelt_on', 'obstruction', 'normal_driving']

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
```

### PyQt5设计的界面

我们将使用PyQt5设计一个简单的GUI界面来进行模型预测。

#### GUI代码 `gui_app.py`

```python
[<title="PyQt5 GUI for YOLOv8 Driver Status Detection">]
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QMessageBox, QProgressBar, QTextEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from ultralytics import YOLO
from ui_mainwindow import Ui_MainWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setWindowTitle("Driver Status Detection")
        self.setGeometry(100, 100, 800, 600)

        self.model = YOLO('models/best.pt')
        self.cap = None
        self.timer = QTimer()

        self.ui.pushButton_image.clicked.connect(self.open_image)
        self.ui.pushButton_folder.clicked.connect(self.open_folder)
        self.ui.pushButton_video.clicked.connect(self.open_video)
        self.ui.pushButton_camera.clicked.connect(self.start_camera)
        self.ui.pushButton_stop.clicked.connect(self.stop_camera)

    def open_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "Images (*.jpeg *.jpg);;All Files (*)", options=options)
        if file_name:
            self.process_image(file_name)

    def process_image(self, file_name):
        img = cv2.imread(file_name)  # BGR
        assert img is not None, f'Image Not Found {file_name}'

        results = self.model(img, stream=True)

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                r = box.xyxy[0].astype(int)
                cls = int(box.cls[0])
                conf = box.conf[0]
                label = f'{self.model.names[cls]} {conf:.2f}'
                color = (0, 255, 0)  # Green
                cv2.rectangle(img, r[:2], r[2:], color, 2)
                cv2.putText(img, label, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.ui.label_display.setPixmap(pixmap.scaled(800, 600))

    def open_folder(self):
        folder_name = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_name:
            for filename in os.listdir(folder_name):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(folder_name, filename)
                    self.process_image(file_path)

    def open_video(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "Videos (*.mp4 *.avi);;All Files (*)", options=options)
        if file_name:
            self.cap = cv2.VideoCapture(file_name)
            self.timer.timeout.connect(self.process_frame)
            self.timer.start(30)  # Process frame every 30 ms

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(30)  # Process frame every 30 ms

    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.timer.stop()

    def process_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                results = self.model(frame, stream=True)

                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        r = box.xyxy[0].astype(int)
                        cls = int(box.cls[0])
                        conf = box.conf[0]
                        label = f'{self.model.names[cls]} {conf:.2f}'
                        color = (0, 255, 0)  # Green
                        cv2.rectangle(frame, r[:2], r[2:], color, 2)
                        cv2.putText(frame, label, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.ui.label_display.setPixmap(pixmap.scaled(800, 600))
            else:
                self.cap.release()
                self.cap = None
                self.timer.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
```

### 辅助工具文件 `utils.py`

这个文件可以用来存放一些辅助函数，比如保存结果等。

```python
[<title="Utility Functions for Driver Status Detection">]
import cv2
import os

def save_results(image, detections, output_dir, filename):
    for det in detections:
        r = det['bbox']
        cls = det['class']
        conf = det['confidence']
        label = f'{cls} {conf:.2f}'
        color = (0, 255, 0)  # Green
        cv2.rectangle(image, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), color, 2)
        cv2.putText(image, label, (int(r[0]), int(r[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image)
```

### 界面设计文件 `ui_mainwindow.py`

假设你已经使用 `pyuic5` 将 `ui_mainwindow.ui` 转换为 `ui_mainwindow.py`。这里提供一个简化的示例。

```python
[<title="Generated UI Code from ui_mainwindow.ui">]
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_display = QtWidgets.QLabel(self.centralwidget)
        self.label_display.setText("")
        self.label_display.setObjectName("label_display")
        self.verticalLayout.addWidget(self.label_display)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_image = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_image.setObjectName("pushButton_image")
        self.horizontalLayout.addWidget(self.pushButton_image)
        self.pushButton_folder = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_folder.setObjectName("pushButton_folder")
        self.horizontalLayout.addWidget(self.pushButton_folder)
        self.pushButton_video = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_video.setObjectName("pushButton_video")
        self.horizontalLayout.addWidget(self.pushButton_video)
        self.pushButton_camera = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_camera.setObjectName("pushButton_camera")
        self.horizontalLayout.addWidget(self.pushButton_camera)
        self.pushButton_stop = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_stop.setObjectName("pushButton_stop")
        self.horizontalLayout.addWidget(self.pushButton_stop)
        self.verticalLayout.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_image.setText(_translate("MainWindow", "Open Image"))
        self.pushButton_folder.setText(_translate("MainWindow", "Open Folder"))
        self.pushButton_video.setText(_translate("MainWindow", "Open Video"))
        self.pushButton_camera.setText(_translate("MainWindow", "Start Camera"))
        self.pushButton_stop.setText(_translate("MainWindow", "Stop Camera"))
```

### 资源文件 `resources.qrc` 和 `resources_rc.py`

假设你已经创建了一个 `resources.qrc` 文件，并使用 `pyrcc5` 转换为 `resources_rc.py`。

```xml
<RCC>
    <qresource prefix="/">
        <file>icons/icon.png</file>
    </qresource>
</RCC>
```

### 运行效果展示

假设你已经有了运行效果的图像，可以在 `README.md` 中添加这些图像以供参考。

```markdown
# Driver Status Detection System

## Overview
This project provides a deep learning-based system for detecting various driver states using cameras. The system can identify states such as drowsiness, distraction, phone calling, smoking, eating, wearing sunglasses, seatbelt usage, obstruction, and normal driving.

## Environment Setup
- Software: PyCharm + Anaconda
- Environment: Python=3.9, OpenCV-Python, PyQt5, Torch=1.9

## Features
- Detects 9 types of driver states: ["drowsiness", 'distraction', 'phone_calling', 'smoking', 'eating', 'wearing_sunglasses', 'seatbelt_on', 'obstruction', 'normal_driving']
- Supports detection on images, folders, videos, and live camera feed.
- Batch processing of images.
- Real-time display of detected states with confidence scores and bounding boxes.
- Saving detection results.

## Usage
1. Run the program.
2. Choose an option to detect states in images, folders, videos, or via the camera.

## Screenshots
![Example Screenshot](data/screenshots/example_screenshot.png)
```

### 总结

这样就可以构建一个完整的基于YOLOv8的驾驶员状态检测系统啦！

你看，另外还有这个
包括数据集准备、环境部署、模型训练、指标可视化展示和PyQt5界面设计。以下是所有相关的代码文件：

1. **数据集划分脚本** (`split_dataset.py`)
2. **训练脚本** (`train_yolov8.py`)
3. **指标可视化脚本** (`visualize_metrics.py`)
4. **GUI应用代码** (`gui_app.py`)
5. **辅助工具文件** (`utils.py`)
6. **UI界面源文件** (`ui_mainwindow.py`)
7. **资源文件** (`resources.qrc`, `resources_rc.py`)
8. **文档** (`README.md`)


