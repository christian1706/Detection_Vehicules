import cv2
from ultralytics import YOLO
from collections import defaultdict

# Load the YOLO model
model = YOLO('yolo11l.pt')

class_list = model.names 

# Open the video file
cap = cv2.VideoCapture('test_videos/video2test1.mp4')

#class_list
class_list = model.names 

line_y_red = 430 # Red line position

# Dictionary to store object counts by class
class_counts = defaultdict(int)

# Dictionary to keep track of object IDs that have crossed the line
crossed_ids = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO tracking on the frame
    results = model.track(frame, persist=True, classes = [2,3,4,6,7,8], conf = 0.7) 
    #print(results)

    # Ensure results are not empty
    if results[0].boxes.data is not None:
        # Get the detected boxes, their class indices, and track IDs
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu()

        cv2.line(frame, (400, line_y_red), (1200, line_y_red), (0, 0, 255), 3)
        #cv2.putText(frame, 'Red Line', (690, line_y_red - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Loop through each detected object
        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2  # Calculate the center point
            cy = (y1 + y2) // 2            

            class_name = class_list[class_idx]

            # Uncomment the line below to restore the points on the detected objects.
            #cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            cv2.putText(frame, f"{class_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 


            # Check if the object has crossed the red line
            if cy > line_y_red and track_id not in crossed_ids:
                # Mark the object as crossed
                crossed_ids.add(track_id)
                class_counts[class_name] += 1


        # Display the counts on the frame 
        y_offset = 120
        text_color = (127, 0, 255)
        thickness = 2 
        fontscale = 1
        font = cv2.FONT_HERSHEY_TRIPLEX
        for class_name, count in class_counts.items():
            cv2.putText(frame, f"{class_name}: {count}", (1000, y_offset),
                        font, fontscale, text_color, thickness)
            y_offset += 40

    
    
    # Show the frame
    cv2.imshow("YOLO Object Tracking & Counting", frame)    
    
    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
