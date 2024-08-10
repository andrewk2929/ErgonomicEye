import cv2
import mediapipe as mp
import numpy as np
import time, os

# setup vars
font = cv2.FONT_HERSHEY_SIMPLEX
posture_setup_complete = False
setup_frames = 0
initial_shoulder_angles = []
initial_neck_angles = []
shoulder_threshold = 0
neck_threshold = 0

# alert/status vars
ready_to_start = False
alert = 'alert.mp3' # your alert audio
sit = True
sedentary = False
active_threshold = 60 # seconds
sedentary_threshold = 1800 # seconds
start_time = time.time()
time_diff = 0
alert_cooldown = 15 # seconds
last_alert_time = 0

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Mediapipe Drawing
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

# when poor posture is detected
def poor_posture_detected():
    pass
    # if current_time - last_alert_time > alert_cooldown:
        # if os.path.exists(alert):
        #     playsound(alert)
    #     last_alert_time = current_time


# Initialize webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    keyInput = cv2.waitKey(100)
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break
    
    if not ready_to_start:
        cv2.putText(frame, "Hit 'q' At Anytime To Quit Out", (400,250), font, 1, (0, 0, 255), 2, cv2.LINE_4)
        cv2.putText(frame, "Please Sit Straight. When Ready Hit Space To Launch", (250,50), font, 1, (0, 0, 255), 2, cv2.LINE_4)
        if keyInput == ord(' '):
            ready_to_start = True
    
    elif ready_to_start:    

        if setup_frames < 25: 
            cv2.putText(frame, "Analyzing your good posture", (10,50), font, 1, (0, 0, 0), 2, cv2.LINE_4)

        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process RGB frames with Mediapipe
        pose_results = pose.process(rgb_frame)

        # Draw pose landmarks
        # Detects if landmarks are visible
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )
            
            #if all landmarks detected for more than set time, you are asked to take a break
            #there's a snooze option that will reset the timer
            time_diff = time.time() - start_time 
            if time_diff > sedentary_threshold:
                cv2.putText(frame, 
                f"It has been {round(time_diff/60)} minutes, Please Take A Break",
                (250,50), font, 1, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(frame, "Please Press Esc To Reset",
                (400,250), font, 1, (0,0,255), 2, cv2.LINE_AA)
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
            if setup_frames < 25:
                initial_shoulder_angles.append(shoulder_angle)
                initial_neck_angles.append(neck_angle)
                setup_frames += 1
                cv2.putText(frame, f"Gathering data... {setup_frames}/25", 
                            (10, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
            
            # after 25 frames, take the average angles and set thresholds for angles
            elif setup_frames >= 25 and setup_frames < 40:
                shoulder_threshold = np.mean(initial_shoulder_angles) - 10
                neck_threshold = np.mean(initial_neck_angles) - 10
                # os.system('clear')           
                setup_frames += 1
                cv2.putText(frame, 
                f"Setup complete! Shoulder threshold: {shoulder_threshold:.1f} and neck threshold: {neck_threshold:.1f}",
                (70,80), font, 1, (0,0,255), 2, cv2.LINE_AA)
      
                #print(f"Setup complete! Shoulder threshold: {shoulder_threshold:.1f} and neck threshold: {neck_threshold:.1f}")

            # Begin posture detection
            else:
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

                # poor shoulder and neck posture
                if shoulder_threshold > shoulder_angle and neck_threshold > neck_angle:
                    cv2.putText(frame,
                    f"""Poor neck and shoulder posture detected! Please sit up straight. 
                    Shoulder: {shoulder_angle:.1f}/{shoulder_threshold:.1f} Neck: {neck_angle:.1f}/{neck_threshold:.1f}""",
                    (100,50), font, 1, (255,0,0), 2, cv2.LINE_4)
                    poor_posture_detected()
                # print(nose, left_shoulder, right_shoulder)
                
        else: 
            if (time.time() - (start_time + time_diff)) > active_threshold:
                start_time = time.time()

    #Display the frame
    cv2.imshow('ErgonomicEye', frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam
cap.release()
cv2.destroyAllWindows()

pose.close()
# object_detection.close()
# try making thresholds a percent instead for usability
# can then delete the thresholding angles after 25 frames are captured