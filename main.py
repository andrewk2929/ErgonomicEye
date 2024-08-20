import mediapipe as mp
import numpy as np
import time, os
import streamlit as st
# import cv2

# Streamlit setup
st.set_page_config(
    page_title="ErgonomicEye | The Only Way to Work",
    page_icon="🧘‍♀️"
    )

st.title("ErgonomicEye")
st.markdown("***")
st.subheader("Posture Analysis and Sedentary Detection made easy!")

# dynamic setup vars
sedentary_threshold = st.select_slider(
    "Choose sedentary threshold (After how many minutes should we recommend an active break)", 
    options=range(10, 121, 10),
    value=60,
    disabled = False
)

alert_type = st.selectbox("Choose alert type", ["Auditory and Visual", "Visual"])
st.text("Auditory and Visual warning recommended")

st.subheader("Please sit straight. When ready hit start to begin your posture analysis.")

if 'started' not in st.session_state:
    st.session_state.started = False


### OPENCV SETUP
def main():
    st.session_state.started = True

    # setup vars
    font = cv2.FONT_HERSHEY_SIMPLEX
    # posture_setup_complete = False
    setup_frames = 0
    initial_shoulder_angles = []
    initial_neck_angles = []
    neck_slopes = []
    shoulder_slopes = []
    shoulder_threshold = 0
    neck_threshold = 0

    # alert/status vars
    # ready_to_start = False
    # sit = True
    # sedentary = False
    active_threshold = 60 # seconds
    start_time = time.time()
    time_diff = 0
    # key_count = 0
    alert_cooldown = 2 # seconds
    last_alert_time = 0
    pinged = False
    curr = 0

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

    # play alert sound 
    def ping_user(current):
        pass
        # print("BEEP")
        # global pinged
        # # pass
        # if pinged:
        #     if (time.time() - (curr)) > alert_cooldown:
        #         os.system('clear')
        #         print("BEEP ALERT ALERT")
        #         st.write("BEEEP")
        # else:
        #     if (time.time() - (current)) > alert_cooldown and not pinged:
        #         # if os.path.exists("alert.mp3"):
        #         os.system('clear')
        #         print("BEEP ALERT ALERT")
        #         st.write("BEEEP")
        #     else:
        #         pinged = True
        #         curr = current

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Streamlit image display
    image_placeholder = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            st.error("Failed to grab frame")
            break
        
        # if not ready_to_start:
        #     cv2.putText(frame, "Hit 'q' At Anytime To Quit Out", (400,250), font, 1, (0, 0, 255), 2, cv2.LINE_4)
        #     cv2.putText(frame, "Please Sit Straight. When Ready Hit Space To Launch", (250,50), font, 1, (0, 0, 255), 2, cv2.LINE_4)
        #     # dynamic key
        #     key_count += 1
        #     if st.button("Calibrate", key = f"calibration_btn{key_count}"):
        #         ready_to_start = True
        
        # else:    

        if setup_frames < 25: 
            cv2.putText(frame, "Analyzing your good posture", (10,50), font, 1, (255, 0, 0), 2, cv2.LINE_4)

        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process RGB frames with Mediapipe
        pose_results = pose.process(rgb_frame)

        # Draw pose landmarks
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            
            time_diff = time.time() - start_time 
            if time_diff > sedentary_threshold:
                cv2.putText(frame, 
                f"It has been {round(time_diff/60)} minutes, Please Take A Break",
                (250,150), font, 1, (0,0,255), 2, cv2.LINE_AA)
                if alert_type in ["Auditory and Visual"]:
                    ping_user(time.time())
                # sedentary(time_diff)
                # if st.button("Snooze", key = "snooze_btn"):
                #     start_time = time.time()

            landmarks = pose_results.pose_landmarks.landmark

            # Get key body parts
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1],
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x * frame.shape[1],
                    landmarks[mp_pose.PoseLandmark.NOSE.value].y * frame.shape[0]]
        
            # Calculate mid point of shoulders
            mid_shoulder = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2]
            
            # Calculate angles
            shoulder_angle = 180 - calculate_angle(left_shoulder, nose, right_shoulder)
            neck_angle = calculate_angle(mid_shoulder, nose, [nose[0], 0])

            # Calculate slopes (see if neck or shoulders are slanted)
            neck_slope = ((mid_shoulder[1] - nose[1])/(mid_shoulder[0] - nose[0])) # slope between nose and mid shoulder
            shoulder_slope = ((right_shoulder[1] - left_shoulder[1])/(right_shoulder[0]-left_shoulder[0]))

            # setting up baseline angles from first 25 frames
            if setup_frames < 25:
                # append curr angles
                initial_shoulder_angles.append(shoulder_angle)
                initial_neck_angles.append(neck_angle)
                neck_slopes.append(neck_slope)
                shoulder_slopes.append(shoulder_slope)

                setup_frames += 1
                cv2.putText(frame, f"Gathering data... {setup_frames}/25", 
                            (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            # after 25 frames, take the average angles and set thresholds for angles
            elif setup_frames >= 25 and setup_frames < 100:
                shoulder_threshold = np.mean(initial_shoulder_angles) - 10
                neck_threshold = np.mean(initial_neck_angles) - 10

                avg_shoulder_slope = np.mean(shoulder_slopes)
                avg_neck_slope = np.mean(neck_slopes)

                if 0 > avg_neck_slope > -6.5:
                    cv2.putText(frame, 
                f"""NOTE NECK CROOKED TO THE LEFT
                Press start to recalibrate, ignore to continue.""",
                (70,80), font, 1, (255,0,0), 2, cv2.LINE_AA)
                
                elif 0 < avg_neck_slope < 2.3:
                    cv2.putText(frame, 
                f"""NOTE NECK CROOKED TO THE RIGHT
                Press start to recalibrate, ignore to continue.""",
                (70,80), font, 1, (255,0,0), 2, cv2.LINE_AA)
                    
                if avg_shoulder_slope >= 0.1:
                    cv2.putText(frame, 
                f"""NOTE RIGHT SHOULDER HIGHER THAN LEFT SHOULDER
                Press start to recalibrate, ignore to continue.""",
                (70,280), font, 1, (255,0,0), 2, cv2.LINE_AA)
                    
                elif avg_shoulder_slope <= -0.1:
                    cv2.putText(frame, 
                f"""NOTE LEFT SHOULDER HIGHER THAN RIGHT SHOULDER
                Press start to recalibrate, ignore to continue.""",
                (70,280), font, 1, (255,0,0), 2, cv2.LINE_AA)

                setup_frames += 1
                # cv2.putText(frame, 
                # f"Setup complete! Shoulder threshold: {shoulder_threshold:.1f} and neck threshold: {neck_threshold:.1f}",
                # (70,280), font, 1, (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(frame, 
                f"Setup complete! Shoulder slopes: {avg_shoulder_slope:.1f} and neck slopes: {avg_neck_slope:.1f}",
                (270,280), font, 1, (0,255,0), 2, cv2.LINE_AA)

            # Begin posture detection
            else:
                # poor shoulder posture
                if shoulder_threshold > shoulder_angle and neck_angle > neck_threshold:
                    cv2.putText(frame,
                    f"Poor shoulder posture detected! Please sit up straight. {shoulder_angle:.1f}/{shoulder_threshold:.1f}",
                    (100,50), font, 1, (0,0,255), 2, cv2.LINE_4)
                    if alert_type in ["Auditory and Visual"]:
                        ping_user(time.time())
                
                # poor neck posture
                if neck_threshold > neck_angle and shoulder_angle > shoulder_threshold:
                    cv2.putText(frame,
                    f"Poor neck posture detected! Please sit up straight. {neck_angle:.1f}/{neck_threshold:.1f}",
                    (100,50), font, 1, (0,0,255), 2, cv2.LINE_4)
                    if alert_type in ["Auditory and Visual"]:
                        ping_user(time.time())

                # poor shoulder and neck posture
                if shoulder_threshold > shoulder_angle and neck_threshold > neck_angle:
                    cv2.putText(frame,
                    f"Poor neck and shoulder posture detected! Please sit up straight. Shoulder: {shoulder_angle:.1f}/{shoulder_threshold:.1f} Neck: {neck_angle:.1f}/{neck_threshold:.1f}",
                    (100,50), font, 1, (0,0,255), 2, cv2.LINE_4)
                    if alert_type in ["Auditory and Visual"]:
                        ping_user(time.time())
                
        else: 
            if setup_frames < 25:
                cv2.putText(frame,
                    f"Please stay in frame while model gathers data",
                    (10,100), font, 1, (0,0,255), 2, cv2.LINE_4)
            elif (time.time() - (start_time + time_diff)) > active_threshold:
                start_time = time.time()

        # Display the frame in Streamlit
        image_placeholder.image(frame, channels="BGR")

        # Check for stop button
        # if st.button("Stop", key = "stop_btn"):
        #     break

    # Release webcam
    cap.release()
    pose.close()

### END OF POSE AND SEDENTARY ANALYSIS

# 9 columns so col1 and col2 are closer (meaning buttons are closer)
col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)

st.session_state.started = False

# start btn
with col1:
    start_btn = st.button("Start", type = "primary", key = "start_btn", disabled = st.session_state.started)

stop_btn = False

# make stop btn appear when start btn pressed
if start_btn:
    with col2:
        stop_btn = st.button("Stop", type="primary", key = "stop_btn", disabled= False)
    st.info("Press the Start button to recalibrate and the Stop button to exit")

    # start
    st.session_state.started = True
    main()

if stop_btn:
    st.session_state.started = False
    # print("STOPPED")


# try making thresholds a percent instead for usability
# can then delete the thresholding angles after 25 frames are captured
# make an option to press a key to reevaluate thresholds (restart get another 25 frames) reset btn
# maybe add an option to remove the dots on the opencv window
# not working cuz time.time() for the ping user gets updated every frame
