import cv2
import numpy as np
import time

# Camera Feed
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = time.time()

# Virtual boundary box
box_x1, box_y1 = 200, 150
box_x2, box_y2 = 440, 330

# Pre-calc box center
box_cx = (box_x1 + box_x2) // 2
box_cy = (box_y1 + box_y2) // 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result_frame = frame.copy()

    # HSV conversion
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Skin mask (adjustable range)
    lower = np.array([0, 30, 60])
    upper = np.array([20, 150, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological cleanup.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contourS
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    state = "NO HAND"
    distance = -1

    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area > 3000:  
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # Draw tracking point
                cv2.circle(result_frame, (cx, cy), 10, (255, 0, 255), -1)

                # Compute distance to box center
                distance = ((cx - box_cx) ** 2 + (cy - box_cy) ** 2) ** 0.5

                # STATE LOGIC
                if distance > 160:
                    state = "SAFE"
                    color = (0, 255, 0)
                elif distance > 90:
                    state = "WARNING"
                    color = (0, 255, 255)
                else:
                    state = "DANGER"
                    color = (0, 0, 255)

                
                (c_x, c_y), radius = cv2.minEnclosingCircle(cnt)
                cv2.circle(result_frame, (int(c_x), int(c_y)), int(radius), color, 2)

                
                if state == "DANGER":
                    cv2.putText(result_frame, "DANGER DANGER", (30, 60),
                                cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
            else:
                state = "NO HAND"

    
    try:
        cv2.rectangle(result_frame, (box_x1, box_y1), (box_x2, box_y2), color, 3)
    except:
        cv2.rectangle(result_frame, (box_x1, box_y1), (box_x2, box_y2), (255,255,255), 3)

    
    cv2.putText(result_frame, f"STATE: {state}", (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0) if state=="SAFE" else (0,255,255) if state=="WARNING" else (0,0,255),
                2)

    #Distance readout
    if distance != -1:
        cv2.putText(result_frame, f"DIST: {int(distance)}", (430, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # FPS 
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(result_frame, f"FPS: {int(fps)}", (500, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Tracking", result_frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
