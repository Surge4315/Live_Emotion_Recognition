import cv2
from deepface import DeepFace
import numpy as np

cap = cv2.VideoCapture(0)

cv2.namedWindow("LIVE STUDENCIAK REACTION", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("LIVE STUDENCIAK REACTION",
                      cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_GUI_EXPANDED)

panel_width = 250  # side panel width

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # black font for side panel
    panel = np.zeros((frame.shape[0], panel_width, 3), dtype=np.uint8)

    # Add "Press q to quit" at bottom left
    panel_height = panel.shape[0]
    cv2.putText(panel,
                "Press q to quit",
                (10, panel_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA)

    try:
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=True,
            detector_backend="mtcnn"
        )

        data = result[0] if isinstance(result, list) else result

        # drawing rectangle around detected face
        region = data.get("region", None)
        if region:
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        emotions = data.get("emotion", {})
        dominant_emotion = data.get("dominant_emotion", "unknown")

        y_offset = 40
        # Dominant emotion
        cv2.putText(panel,
                    f'Dominant: {dominant_emotion}',
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA)
        y_offset += 50

        # all emotion percentages
        for emo, value in emotions.items():
            cv2.putText(panel,
                        f"{emo}: {value:.1f}%",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA)
            y_offset += 30

    # no face detected case
    except Exception as e:
        y_offset = 40
        cv2.putText(panel,
                    'No face detected',
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA)

    # camera + side panel
    frame = np.hstack((frame, panel))
    cv2.imshow("LIVE STUDENCIAK REACTION", frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# quitting
cap.release()
cv2.destroyAllWindows()
