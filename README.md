# 基于Orange Pi Kunpeng Pro 的无人机港口集装箱智能检测与计数系统

本项⽬基于Orange Pi Kunpeng Pro，结合 YOLOv8n ⽬标检测模型，构建集装箱实时检测与计数系统，实现对无人机拍摄视频中的集装箱的⾃动识别与数量统计。交互界面由PyQt5构建。

> container_detection/ 
> ├── det_utils.py   # ⼯具函数模块，封装图像预处理、后处理等函数
>  ├── labels.txt   # 类别标签文件，存放集装箱检测类别定义
>   ├── new_om.py  # 昇腾推理封装 
>   ├── new_qt.py # 基于 PyQt5 的交互界面
>   ├── best.om  # 已转换好的昇腾 OM 模型文件（由 YOLOv8n 训练权重转化而来） 
>   └── requirements.txt # 依赖列

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE2NzI0MjU0NjYsLTg1MDE0MTA3Ml19
-->