import cv2
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n-seg.pt')
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model(frame)
            annotated_frame = results[0].plot()
            print(results[0].names)
            cv2.imshow('frame', annotated_frame)
            cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()