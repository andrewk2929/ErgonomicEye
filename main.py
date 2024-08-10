import cv2
import mediapipe as mp
import numpy as np
import time, os

# setup vars
posture_setup_complete = False
setup_frames = 0
initial_shoulder_angles = []
initial_neck_angles = []
shoulder_threshold = 0
neck_threshold = 0

# alert/status vars
alert = 'alert.mp3' # your alert audio
sit = True
sedentary = False
not_sitting_time = 60 # seconds
sitting_time = 1800
start_time = time.time()
time_diff = 0
alert_cooldown = 15 # seconds
last_alert_time = 0

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Mediapipe Object Detection
# mp_object_detection = mp.solutions.object_detection
# object_detection = mp_object_detection.ObjectDetection(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# calculate posture angles
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle 

def poor_posture_detected():
    pass
    # if current_time - last_alert_time > alert_cooldown:
        # if os.path.exists(alert):
        #     playsound(alert)
    #     last_alert_time = current_time

def sitting(current_time):
    while True:
        if nose:
            if time.time() - current_time > sitting_time:
                sedentary = True
        if not nose:
            if time.time() - current_time > not_sitting_time:
                sit = False
                sedentary = False
        
        if sedentary:
            cv2.putText(frame, 
            "You've been sitting for over 30 minutes, try and take a 5 minute active break",
            (50,50), font, 1, (0,0,0), 2, cv2.LINE_AA)


def not_sitting(time):
    while not nose:
        if time.time() - time > not_sitting_time:
            sit = False
            break

# Initialize webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    keyInput = cv2.waitKey(10)
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    font = cv2.FONT_HERSHEY_SIMPLEX
    if setup_frames < 25: 
        cv2.putText(frame, "Analyzing your good posture", (50,50), font, 1, (0, 0, 0), 2, cv2.LINE_4)

    # Convert BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frames with Mediapipe
    pose_results = pose.process(rgb_frame)
    # object_results = object_detection.process(rgb_frame)

    # Draw pose landmarks
    #detects if landmarks are visible

    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
        
        #if all landmarks detected for more than set time, you are asked to take a break
        #there's a snooze option that will reset the timer
        time_diff = time.time() - start_time 
        if time_diff> 30:
            cv2.putText(frame, 
            f"It has been {(time_diff/60):.1f} minutes, Please Take A Break",
            (70,80), font, 1, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Please Press Esc To Reset",
            (270,280), font, 1, (0,0,0), 2, cv2.LINE_AA)
            if keyInput == 27:
                start_time = time.time()

        landmarks = pose_results.pose_landmarks.landmark

        # Get key body parts
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1],
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]
                          ,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]]
        nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x * frame.shape[1],
                landmarks[mp_pose.PoseLandmark.NOSE.value].y * frame.shape[0]]
       
        # Calculate mid point of shoulders
        mid_shoulder = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2]
        
        # Calculate angles
        shoulder_angle = 180 - calculate_angle(left_shoulder, nose, right_shoulder)
        neck_angle = calculate_angle(mid_shoulder, nose, [nose[0], 0])
        # shoulder_angle = calculate_angle(left_shoulder, nose, right_shoulder)
        # neck_angle = calculate_angle(mid_shoulder, nose, [nose[0], mid_shoulder[1]])

        # setting up baseline angles from first 25 frames
        if not posture_setup_complete and setup_frames < 25:
            initial_shoulder_angles.append(shoulder_angle)
            initial_neck_angles.append(neck_angle)
            setup_frames += 1
            cv2.putText(frame, f"Gathering data... {setup_frames}/25", 
                        (10, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        # after 25 frames, take the average angles and set thresholds for 
        # angles
        elif not posture_setup_complete:
            shoulder_threshold = np.mean(initial_shoulder_angles) - 10
            neck_threshold = np.mean(initial_neck_angles) - 10
            posture_setup_complete = True
            os.system('clear')           
            cv2.putText(frame, 
            f"Setup complete! Shoulder threshold: {shoulder_threshold:.1f} and neck threshold: {neck_threshold:.1f}",
            (70,80), font, 1, (0,0,0), 2, cv2.LINE_AA)
            #print(f"Setup complete! Shoulder threshold: {shoulder_threshold:.1f} and neck threshold: {neck_threshold:.1f}")

        # Begin posture detection
        elif posture_setup_complete:
            # poor shoulder posture
            if shoulder_threshold > shoulder_angle and neck_angle > neck_threshold:
                cv2.putText(frame,
                f"Poor shoulder posture detected! Please sit up straight. {shoulder_angle:.1f}/{shoulder_threshold:.1f}",
                (100,50), font, 1, (255,0,0), 2, cv2.LINE_4)
                poor_posture_detected()
            
            # poor neck posture
            if neck_threshold > neck_angle and shoulder_angle > shoulder_threshold:
                cv2.putText(frame,
                f"Poor neck posture detected! Please sit up straight. {neck_angle:.1f}/{neck_threshold:.1f}",
                (100,50), font, 1, (255,0,0), 2, cv2.LINE_4)
                poor_posture_detected()

            if shoulder_threshold > shoulder_angle and neck_threshold > neck_angle:
                cv2.putText(frame,
                f"""Poor neck and shoulder posture detected! Please sit up straight. Shoulder: {shoulder_angle:.1f}/{shoulder_threshold:.1f} Neck: {neck_angle:.1f}/{neck_threshold:.1f}""",
                (100,50), font, 1, (255,0,0), 2, cv2.LINE_4)
                poor_posture_detected()
            print (nose, left_shoulder, right_shoulder)
            
    else: 
        print("here")

            # if person not sitting (not in frame)
        
            #     current_time = time.time()
            #         # just started sitting
            #     if not sit: 
            #         start_time = time.time()
            #         sit = True

            #     sitting(start_time)

            #     else:
            #         if sit:
            #             start_time = time.time()
            #             not_sitting(start_time)

            # if not nose:
            #     current_time = time.time()
            #     not_sitting(current_time)

            # if sit:
            #     current_time = time.time()
            #     sitting(current_time)

    # Draw object detection boxes
    # if object_results.detections:
    #     for detection in object_results.detections:
    #         mp_drawing.draw_detection(frame, detection)

    # Display the frame
    cv2.imshow('MediaPipe Pose Detection', frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam
cap.release()
cv2.destroyAllWindows()

pose.close()
# object_detection.close()
# We may need multithreading to have the sit and posture detection run simultaniously