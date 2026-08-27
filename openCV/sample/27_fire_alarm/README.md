# 火灾预警程序（OpenCV DNN）

程序用 OpenCV 加载 YOLO ONNX 模型，检测 `fire`/`smoke`，并在最近 10 帧中至少 7 帧为阳性时触发警报。连续帧确认可以降低灯光、夕阳等造成的偶发误报。

## 准备

```bash
python3 -m pip install "numpy<2" opencv-python
```

准备一个已针对火焰/烟雾训练的 YOLO ONNX 模型。模型必须输出未经过 NMS 的 YOLOv5 或 YOLOv8 常见格式，类别顺序通过 `--labels` 指定。

## 运行

```bash
# 检测图片
python3 fire_alarm.py --model fire.onnx --source test.jpg --output result.jpg

# 检测视频，不弹窗
python3 fire_alarm.py --model fire.onnx --source fire.mp4 --no-display

# 使用摄像头；按 q 或 ESC 退出
python3 fire_alarm.py --model fire.onnx --source 0

# 如果模型只有一个 fire 类别
python3 fire_alarm.py --model fire.onnx --source 0 --labels fire
```

重要参数：

- `--conf 0.5`：单个检测框的最低置信度。
- `--window 10 --required 7`：最近 10 帧至少 7 帧检测到目标才报警。
- `--size 640`：必须与模型导出时的输入尺寸一致。
- `--snapshot-dir alarm_snapshots`：首次进入报警状态时保存截图。

程序会输出带检测框的视频 `fire_alarm_output.mp4`。这是视觉预警示例，不能替代合规的烟感、温感和消防报警设备。
