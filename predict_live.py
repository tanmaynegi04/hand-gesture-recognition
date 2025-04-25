import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ✅ Load the trained model
model = load_model("asl_model.h5")

# ✅ 29 class labels based on your folder structure
labels = [
    'A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I',
    'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R',
    'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

# ✅ Start webcam
cap = cv2.VideoCapture(0)

print("[INFO] Starting live sign detection... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw Region of Interest
    roi = frame[100:300, 100:300]
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)

    # Preprocess the ROI
    img = cv2.resize(roi, (64, 64))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)

    # ✅ Sanity check: shape should match labels
    if preds.shape[1] != len(labels):
        cv2.putText(frame, "Model mismatch!", (100, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("SignLang - Error", frame)
        continue

    preds = preds[0]
    idx = np.argmax(preds)
    confidence = preds[idx]

    # ✅ Only if valid index
    if 0 <= idx < len(labels):
        label = labels[idx]
        cv2.putText(frame, f"{label} ({confidence:.2f})", (100, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Unknown", (100, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show live feed
    cv2.imshow("SignLang - Live ASL Detection", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Cleanup
cap.release()
cv2.destroyAllWindows()