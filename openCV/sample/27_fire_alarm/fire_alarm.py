#!/usr/bin/env python3
"""OpenCV DNN + YOLO ONNX fire/smoke alarm."""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detection:
    box: tuple[int, int, int, int]
    score: float
    label: str


class FireDetector:
    def __init__(self, model: str, labels: list[str], size: int, conf: float, nms: float):
        self.net = cv2.dnn.readNetFromONNX(model)
        self.labels = labels
        self.size = size
        self.conf = conf
        self.nms = nms

    def detect(self, frame: np.ndarray) -> list[Detection]:
        image, scale, pad_x, pad_y = self._letterbox(frame)
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (self.size, self.size), swapRB=True)
        self.net.setInput(blob)
        output = self.net.forward()
        rows = self._rows(output)

        boxes: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        class_count = len(self.labels)

        for row in rows:
            # YOLOv8: xywh + classes; YOLOv5: xywh + objectness + classes.
            if len(row) == class_count + 4:
                class_scores = row[4:]
                class_id = int(np.argmax(class_scores))
                score = float(class_scores[class_id])
            elif len(row) >= class_count + 5:
                class_scores = row[5:5 + class_count]
                class_id = int(np.argmax(class_scores))
                score = float(row[4] * class_scores[class_id])
            else:
                raise ValueError(
                    f"模型每个候选框有 {len(row)} 个值，与 {class_count} 个标签不匹配"
                )
            if score < self.conf:
                continue

            cx, cy, width, height = map(float, row[:4])
            x = int((cx - width / 2 - pad_x) / scale)
            y = int((cy - height / 2 - pad_y) / scale)
            w = int(width / scale)
            h = int(height / scale)
            boxes.append([x, y, w, h])
            scores.append(score)
            class_ids.append(class_id)

        keep = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.nms)
        result: list[Detection] = []
        frame_h, frame_w = frame.shape[:2]
        for index in np.asarray(keep).reshape(-1):
            x, y, w, h = boxes[int(index)]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame_w - 1, x + w), min(frame_h - 1, y + h)
            if x2 > x1 and y2 > y1:
                result.append(Detection((x1, y1, x2, y2), scores[int(index)],
                                        self.labels[class_ids[int(index)]]))
        return result

    def _letterbox(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        height, width = frame.shape[:2]
        scale = min(self.size / width, self.size / height)
        new_w, new_h = round(width * scale), round(height * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        pad_x, pad_y = (self.size - new_w) // 2, (self.size - new_h) // 2
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return canvas, scale, pad_x, pad_y

    @staticmethod
    def _rows(output: np.ndarray) -> np.ndarray:
        data = np.squeeze(output)
        if data.ndim != 2:
            raise ValueError(f"不支持的模型输出形状: {output.shape}")
        # v8 commonly returns [attributes, candidates], v5 returns the reverse.
        if data.shape[0] < data.shape[1]:
            data = data.T
        return data


class AlarmWindow:
    def __init__(self, window: int, required: int):
        if not 1 <= required <= window:
            raise ValueError("--required 必须在 1 到 --window 之间")
        self.samples: deque[bool] = deque(maxlen=window)
        self.required = required

    def update(self, detected: bool) -> bool:
        self.samples.append(detected)
        return len(self.samples) == self.samples.maxlen and sum(self.samples) >= self.required


def draw(frame: np.ndarray, detections: list[Detection], alarm: bool) -> None:
    for item in detections:
        x1, y1, x2, y2 = item.box
        color = (0, 0, 255) if item.label.lower() == "fire" else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{item.label} {item.score:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    status = "FIRE ALARM" if alarm else "monitoring"
    cv2.putText(frame, status, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if alarm else (0, 200, 0), 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 ONNX YOLO 的火灾/烟雾预警")
    parser.add_argument("--model", required=True, help="YOLO ONNX 模型路径")
    parser.add_argument("--source", default="0", help="图片、视频路径或摄像头编号，默认 0")
    parser.add_argument("--labels", default="fire,smoke", help="模型类别，顺序必须与训练一致")
    parser.add_argument("--size", type=int, default=640, help="模型输入尺寸")
    parser.add_argument("--conf", type=float, default=0.5, help="检测置信度阈值")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--window", type=int, default=10, help="判断窗口帧数")
    parser.add_argument("--required", type=int, default=7, help="窗口内报警所需阳性帧数")
    parser.add_argument("--output", default="fire_alarm_output.mp4", help="视频结果路径")
    parser.add_argument("--snapshot-dir", default="alarm_snapshots", help="报警截图目录")
    parser.add_argument("--no-display", action="store_true", help="不打开显示窗口")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    detector = FireDetector(args.model, labels, args.size, args.conf, args.nms)
    source_path = Path(args.source)
    image = cv2.imread(args.source) if source_path.is_file() else None
    if image is not None:
        detections = detector.detect(image)
        draw(image, detections, bool(detections))
        output = Path(args.output).with_suffix(".jpg")
        cv2.imwrite(str(output), image)
        print(f"检测完成: {len(detections)} 个目标，结果保存到 {output}")
        return 0

    source: int | str = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开输入源: {args.source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    history = AlarmWindow(args.window, args.required)
    snapshot_dir = Path(args.snapshot_dir)
    was_alarm = False

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            detections = detector.detect(frame)
            alarm = history.update(bool(detections))
            draw(frame, detections, alarm)
            writer.write(frame)
            if alarm and not was_alarm:
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                filename = snapshot_dir / f"alarm_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(filename), frame)
                print(f"警报触发，截图: {filename}")
            was_alarm = alarm
            if not args.no_display:
                cv2.imshow("Fire alarm", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()
    print(f"处理结束，标注视频保存到 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
