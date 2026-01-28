import cv2

def main():
    cam_index = 4

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera open failed. Try cam_index different number (current: {cam_index})")

    # (선택) 해상도 설정 — 지원 안 하면 자동으로 무시될 수 있어요
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
    
        if not ret:
            print("Frame read failed.")
            break

        cv2.imshow("webcam", frame)

        # q 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
