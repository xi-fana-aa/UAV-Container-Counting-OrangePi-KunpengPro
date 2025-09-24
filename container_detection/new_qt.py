import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout, QTextEdit
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
import acl
from new_om import YoloV8, init_acl, deinit_acl, DEVICE_ID, trained_model_path, labels_path

class VideoThread(QThread):
    update_original_frame = pyqtSignal(np.ndarray)
    update_detected_frame = pyqtSignal(np.ndarray)
    update_status = pyqtSignal(str)
    finished_signal = pyqtSignal(int)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self._running = True
        self.context = None
        self.det_model = None

    def run(self):
        self.update_status.emit(f"初始化ACL...")
        self.context = init_acl(DEVICE_ID)
        self.det_model = YoloV8(model_path=trained_model_path)
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.update_status.emit(f"无法打开视频 {self.video_path}")
            return

        total_count = 0
        frame_idx = 0
        self.update_status.emit(f"开始检测视频...")

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # 显示原视频
            self.update_original_frame.emit(frame)

            # 推理
            count, img_with_boxes = self.det_model.infer(frame)
            if frame_idx%30==0:
                total_count += count

            # 显示检测视频
            self.update_detected_frame.emit(img_with_boxes)

        cap.release()
        self.det_model.release()
        deinit_acl(self.context, DEVICE_ID)
        self.update_status.emit(f"结束检测，共检测到 {total_count} 个集装箱")
        self.finished_signal.emit(total_count)

    def stop(self):
        self._running = False

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("集装箱视频检测")
        self.video_path = ""
        self.thread = None
        self.total_count = 0
        self.init_ui()

    def init_ui(self):
        # 按钮
        self.select_btn = QPushButton("选择视频")
        self.start_btn = QPushButton("开始检测")
        self.stop_btn = QPushButton("停止检测")

        self.select_btn.clicked.connect(self.select_video)
        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn.clicked.connect(self.stop_detection)

        # 标签显示视频
        self.original_label = QLabel("原视频")
        self.detected_label = QLabel("检测视频")
        self.original_label.setFixedSize(480, 360)
        self.detected_label.setFixedSize(480, 360)
        self.original_label.setStyleSheet("background-color: black")
        self.detected_label.setStyleSheet("background-color: black")

        # 提示框
        self.status_box = QTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setFixedHeight(100)

        # 布局
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.select_btn)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)

        video_layout = QHBoxLayout()
        video_layout.addWidget(self.original_label)
        video_layout.addWidget(self.detected_label)

        main_layout = QVBoxLayout()
        main_layout.addLayout(btn_layout)
        main_layout.addLayout(video_layout)
        main_layout.addWidget(self.status_box)

        self.setLayout(main_layout)

    def select_video(self):
        file_dialog = QFileDialog()
        path, _ = file_dialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi *.mov)")
        if path:
            self.video_path = path
            self.status_box.append(f"选择视频: {self.video_path}")

    def start_detection(self):
        if not self.video_path:
            self.status_box.append("请先选择视频文件")
            return
        self.thread = VideoThread(self.video_path)
        self.thread.update_original_frame.connect(self.update_original_video)
        self.thread.update_detected_frame.connect(self.update_detected_video)
        self.thread.update_status.connect(self.append_status)
        self.thread.finished_signal.connect(self.finished_detection)
        self.thread.start()
        self.status_box.append("开始检测视频...")

    def stop_detection(self):
        if self.thread:
            self.thread.stop()
            self.status_box.append("停止检测请求已发送")

    def update_original_video(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qt_img = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.original_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.original_label.width(), self.original_label.height(), Qt.AspectRatioMode.KeepAspectRatio
        ))

    def update_detected_video(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qt_img = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.detected_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.detected_label.width(), self.detected_label.height(), Qt.AspectRatioMode.KeepAspectRatio
        ))

    def append_status(self, text):
        self.status_box.append(text)

    def finished_detection(self, total_count):
        self.total_count = total_count
        self.status_box.append(f"检测完成，总共检测到 {total_count} 个集装箱")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
