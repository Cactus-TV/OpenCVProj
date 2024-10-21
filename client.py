import cv2
import numpy as np
import time
import grpc
import requests
import circle_detection_pb2 as pb2
import circle_detection_pb2_grpc as pb2_grpc
from ultralytics import YOLO

# параметры для детекции круга
PARAM1 = 50
PARAM2 = 24
MIN_RADIUS = 1
MAX_RADIUS = 15

last_center = None
last_time = None


def send_detection_to_server(detections, frame_id):
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = pb2_grpc.CircleDetectionServiceStub(channel)
        detection_objects = []

        for det in detections:
            bbox = pb2.BoundingBox(x_min=det[0], y_min=det[1], x_max=det[2], y_max=det[3])
            detection = pb2.Detection(object_id=0, bbox=bbox)
            detection_objects.append(detection)

        data = pb2.Data(frame_id=frame_id, detections=detection_objects)
        responses = stub.GetStreamData(data)

        for response in responses:
            print(response)



def detect_circle(frame, frame_id):
    global last_center, last_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3, 3))
    gray = gray[140:300, 193:417]

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=PARAM1, param2=PARAM2, minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)
    
    detections = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            x, y = int(x) + 193, int(y) + 140
            bbox = (x - r, y - r, x + r, y + r)
            detections.append(bbox)
            
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 4)

            if last_center is not None and last_time is not None:
                delta_t = time.time() - last_time
                velocity = ((x - last_center[0]) / delta_t, (y - last_center[1]) / delta_t)
                speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
                end_point = (int(x + velocity[0] / 10), int(y + velocity[1] / 10))  # нормализация для визуализации
                cv2.arrowedLine(frame, (x, y), end_point, (0, 255, 0), 2)
                print(f"Speed: {speed:.2f}, Direction: {np.degrees(np.arctan2(velocity[1], velocity[0])):.2f} degrees")

            last_center = (x, y)
            last_time = time.time()
            break

    send_detection_to_server(detections, frame_id)
    return frame


def detect_circle_with_yolo(frame, frame_id):
    global last_center, last_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
    gray = gray[140:300, 193:417]

    model = YOLO("best.pt")
    results = model.predict(gray)
    detections = []
    for box in results[0].boxes:
        cords = box.xyxy[0].tolist()
        size = abs((cords[0] - cords[2]) * (cords[1] - cords[3]))
        conf = box.conf[0].item()

        if size < 150 or size > 500 or conf < 0.5:
            continue

        x_min, y_min, x_max, y_max = map(int, cords)
        bbox = (x_min + 193, y_min + 140, x_max + 193, y_max + 140)
        detections.append(bbox)

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

        x = (bbox[0] + bbox[2]) // 2
        y = (bbox[1] + bbox[3]) // 2
        if last_center is not None and last_time is not None:
            delta_t = time.time() - last_time
            velocity = ((x - last_center[0]) / delta_t, (y - last_center[1]) / delta_t)
            speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
            end_point = (int(x + velocity[0] / 10), int(y + velocity[1] / 10))  # нормализация для визуализации
            cv2.arrowedLine(frame, (x, y), end_point, (0, 255, 0), 2)
            print(f"Speed: {speed:.2f}, Direction: {np.degrees(np.arctan2(velocity[1], velocity[0])):.2f} degrees")

        last_center = (x, y)
        last_time = time.time()
        break

    send_detection_to_server(detections, frame_id)
    return frame
           

def main():
    frame_id = 0 
    url = "http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg" 
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        _bytes = bytes()
        while True:
            for chunk in response.iter_content(chunk_size=1024):
                _bytes += chunk
                a = _bytes.find(b'\xff\xd8')
                b = _bytes.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = _bytes[a:b+2]
                    _bytes = _bytes[b+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                    frame = detect_circle_with_yolo(frame, frame_id)
                    # frame = detect_circle(frame, frame_id)

                    cv2.imshow("Detection", frame)
                    frame_id += 1

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        print("Не удалось установить соединение")

if __name__ == "__main__":
    main()
