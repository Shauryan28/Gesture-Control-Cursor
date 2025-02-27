"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        Gesture Control System                                 ║
║                                                                              ║
║  A sophisticated hand gesture-based computer control system that lets you    ║
║  control your mouse cursor using natural hand movements.                     ║
║                                                                              ║
║  Made with ❤️ by Shaurya Nandecha                                           ║
║  GitHub: https://github.com/Shauryan28                                      ║
║                                                                              ║
║  Features:                                                                   ║
║  - Mouse control with index finger                                          ║
║  - Left click/drag with index-thumb pinch                                   ║
║  - Right click with middle-thumb pinch                                      ║
║  - Activation/deactivation with "Yo" sign                                   ║
║  - Smooth cursor movement with enhanced gesture detection                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
from collections import deque

# Initialize MediaPipe Hands with higher confidence thresholds
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.85,  # Further increased for better accuracy
    min_tracking_confidence=0.85,   # Further increased for better tracking
    model_complexity=1              # Use more complex model for better accuracy
)

# Initialize OpenCV VideoCapture
cap = cv2.VideoCapture(0)

# Set PyAutoGUI settings
pyautogui.FAILSAFE = False
screen_width, screen_height = pyautogui.size()

# Enhanced gesture detection parameters
class GestureParams:
    def __init__(self):
        self.last_pinch_time = 0
        self.double_click_threshold = 0.3
        self.pinch_threshold = 0.035  # Further refined for more precise detection
        self.activation_cooldown = 1.5  # Increased to prevent accidental toggling
        self.last_activation_time = 0
        self.is_dragging = False
        self.gesture_control_active = False
        
        # Smoothing parameters
        self.cursor_positions = deque(maxlen=12)  # Increased for smoother movement
        self.movement_threshold = 1.5  # Reduced for more precision
        
        # Gesture state
        self.hand_state_buffer = deque(maxlen=7)  # Increased for more stable detection
        
        # Activation gesture confidence
        self.activation_confidence = 0
        self.activation_threshold = 0.6  # Further reduced for easier activation
        
        # New activation stability parameters
        self.yo_sign_start_time = 0
        self.yo_sign_hold_duration = 1.0  # Hold yo sign for 1 second to activate/deactivate
        self.yo_sign_active = False
        self.activation_frames = deque(maxlen=15)  # Increased for more stability
        self.activation_hysteresis = 0.2  # Hysteresis for activation/deactivation
        self.activation_state_change_time = 0
        self.state_change_lockout = 2.0  # Seconds to lock out state changes after a change
        
        # Debug mode
        self.debug_mode = True
        
    def smooth_coordinates(self, x, y):
        self.cursor_positions.append((x, y))
        if len(self.cursor_positions) < 3:
            return x, y
            
        # Enhanced weighted average with more weight to recent positions
        weights = np.linspace(0.3, 1.0, len(self.cursor_positions))
        weights = weights / np.sum(weights)
        
        smooth_x = np.average([p[0] for p in self.cursor_positions], weights=weights)
        smooth_y = np.average([p[1] for p in self.cursor_positions], weights=weights)
        
        # Apply adaptive movement threshold based on speed
        if len(self.cursor_positions) > 1:
            last_x, last_y = self.cursor_positions[-2]
            movement_speed = np.sqrt((smooth_x - last_x)**2 + (smooth_y - last_y)**2)
            
            # Adjust threshold based on movement speed
            adaptive_threshold = self.movement_threshold
            if movement_speed < 5:
                adaptive_threshold = self.movement_threshold * 0.8
            
            if abs(smooth_x - last_x) < adaptive_threshold:
                smooth_x = last_x
            if abs(smooth_y - last_y) < adaptive_threshold:
                smooth_y = last_y
                
        return smooth_x, smooth_y

params = GestureParams()

def is_finger_extended_simple(landmarks, finger_tip_idx, finger_mcp_idx):
    """Simplified method to check if a finger is extended based on y-position only"""
    tip = landmarks[finger_tip_idx]
    mcp = landmarks[finger_mcp_idx]
    
    # Check if tip is above MCP (for vertical orientation)
    return tip.y < mcp.y - 0.05  # Added threshold to ensure clear extension

def is_finger_extended(landmarks, finger_tip_idx, finger_mcp_idx, finger_pip_idx):
    """More accurate method to check if a finger is extended"""
    tip = landmarks[finger_tip_idx]
    mcp = landmarks[finger_mcp_idx]
    pip = landmarks[finger_pip_idx]
    
    # Check if tip is above MCP (for vertical orientation)
    # Relaxed vertical check - just needs to be higher than MCP by a small margin
    vertical_check = tip.y < mcp.y - 0.03
    
    # Check angle at PIP joint
    vec1 = np.array([pip.x - mcp.x, pip.y - mcp.y])
    vec2 = np.array([tip.x - pip.x, tip.y - pip.y])
    
    # Normalize vectors
    if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        
        # Calculate angle
        angle = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))
        angle_check = np.degrees(angle) > 140  # Relaxed from 160 to 140 degrees
    else:
        angle_check = False
    
    # Distance check (tip should be far from MCP)
    dist = np.sqrt((tip.x - mcp.x)**2 + (tip.y - mcp.y)**2)
    dist_check = dist > 0.08  # Reduced from 0.1 to 0.08
    
    # For index and pinky, we'll be more lenient
    if finger_tip_idx == mp_hands.HandLandmark.INDEX_FINGER_TIP or finger_tip_idx == mp_hands.HandLandmark.PINKY_TIP:
        return vertical_check and (angle_check or dist_check)
    else:
        return vertical_check and angle_check and dist_check

def is_finger_folded(landmarks, finger_tip_idx, finger_mcp_idx):
    """Check if a finger is folded down"""
    tip = landmarks[finger_tip_idx]
    mcp = landmarks[finger_mcp_idx]
    
    # Tip should be below MCP for folded finger
    vertical_check = tip.y > mcp.y
    
    # Tip should be close to palm
    dist = np.sqrt((tip.x - mcp.x)**2 + (tip.y - mcp.y)**2)
    dist_check = dist < 0.12  # Increased from 0.1 to 0.12
    
    return vertical_check or dist_check  # Either condition can indicate folded finger

def detect_yo_sign(landmarks):
    """
    Detect the 'Yo' sign - index finger up, middle and ring fingers down with thumb overlapping them, 
    and pinky finger up
    """
    # Check index finger extended - using both methods for robustness
    index_extended_simple = is_finger_extended_simple(
        landmarks, 
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_MCP
    )
    
    index_extended_full = is_finger_extended(
        landmarks, 
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.INDEX_FINGER_PIP
    )
    
    index_extended = index_extended_simple or index_extended_full
    
    # Check pinky finger extended - using both methods for robustness
    pinky_extended_simple = is_finger_extended_simple(
        landmarks, 
        mp_hands.HandLandmark.PINKY_TIP,
        mp_hands.HandLandmark.PINKY_MCP
    )
    
    pinky_extended_full = is_finger_extended(
        landmarks, 
        mp_hands.HandLandmark.PINKY_TIP,
        mp_hands.HandLandmark.PINKY_MCP,
        mp_hands.HandLandmark.PINKY_PIP
    )
    
    pinky_extended = pinky_extended_simple or pinky_extended_full
    
    # Check middle finger folded
    middle_folded = is_finger_folded(
        landmarks,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP
    )
    
    # Check ring finger folded
    ring_folded = is_finger_folded(
        landmarks,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_MCP
    )
    
    # Check thumb position - should be near middle and ring fingers
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
    
    # Calculate distances from thumb to middle and ring fingers
    thumb_to_middle = np.sqrt((thumb_tip.x - middle_pip.x)**2 + (thumb_tip.y - middle_pip.y)**2)
    thumb_to_ring = np.sqrt((thumb_tip.x - ring_pip.x)**2 + (thumb_tip.y - ring_pip.y)**2)
    
    # Thumb should be close to either middle or ring finger
    thumb_overlapping = thumb_to_middle < 0.1 or thumb_to_ring < 0.1  # Increased from 0.08 to 0.1
    
    # Calculate confidence score (0-1)
    confidence = 0
    if index_extended: confidence += 0.35  # Increased weight for index
    if pinky_extended: confidence += 0.35  # Increased weight for pinky
    if middle_folded: confidence += 0.1
    if ring_folded: confidence += 0.1
    if thumb_overlapping: confidence += 0.1
    
    # Debug information
    if params.debug_mode:
        debug_info = {
            "index_extended_simple": index_extended_simple,
            "index_extended_full": index_extended_full,
            "index_extended": index_extended,
            "pinky_extended_simple": pinky_extended_simple,
            "pinky_extended_full": pinky_extended_full,
            "pinky_extended": pinky_extended,
            "middle_folded": middle_folded,
            "ring_folded": ring_folded,
            "thumb_overlapping": thumb_overlapping,
            "confidence": confidence
        }
        return confidence > 0.65, confidence, debug_info
    
    return confidence > 0.65, confidence

def detect_activation_gesture(landmarks, hand_landmarks, frame=None):
    """Enhanced activation gesture detection using the 'Yo' sign with hold-to-activate"""
    current_time = time.time()
    
    # Check if we're in the lockout period after a state change
    if current_time - params.activation_state_change_time < params.state_change_lockout:
        # Display lockout countdown if in debug mode
        if params.debug_mode and frame is not None:
            lockout_remaining = params.state_change_lockout - (current_time - params.activation_state_change_time)
            cv2.putText(frame, f"Lockout: {lockout_remaining:.1f}s", (10, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return False
    
    # Get yo sign detection and confidence
    if params.debug_mode and frame is not None:
        is_yo_sign, confidence, debug_info = detect_yo_sign(landmarks)
        
        # Display debug information on frame
        h, w, _ = frame.shape
        y_pos = 60
        cv2.putText(frame, f"Index up (simple): {debug_info['index_extended_simple']}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
        cv2.putText(frame, f"Index up (full): {debug_info['index_extended_full']}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
        cv2.putText(frame, f"Pinky up (simple): {debug_info['pinky_extended_simple']}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
        cv2.putText(frame, f"Pinky up (full): {debug_info['pinky_extended_full']}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
        cv2.putText(frame, f"Middle down: {debug_info['middle_folded']}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
        cv2.putText(frame, f"Ring down: {debug_info['ring_folded']}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
        cv2.putText(frame, f"Thumb overlap: {debug_info['thumb_overlapping']}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        is_yo_sign, confidence = detect_yo_sign(landmarks)
    
    # Add to activation frames buffer
    params.activation_frames.append(is_yo_sign)
    
    # Calculate percentage of frames with yo sign
    if len(params.activation_frames) >= 10:  # Need at least 10 frames for stability
        activation_percentage = sum(params.activation_frames) / len(params.activation_frames)
        
        # Apply hysteresis to prevent rapid toggling
        # If currently not in yo sign mode, need higher threshold to enter
        if not params.yo_sign_active:
            if activation_percentage > params.activation_threshold + params.activation_hysteresis:
                params.yo_sign_active = True
                params.yo_sign_start_time = current_time
        # If already in yo sign mode, need lower threshold to exit
        else:
            if activation_percentage < params.activation_threshold - params.activation_hysteresis:
                params.yo_sign_active = False
                params.yo_sign_start_time = 0
    
    # Hold-to-activate logic
    if params.yo_sign_active:
        hold_duration = current_time - params.yo_sign_start_time
        
        # Display hold progress if in debug mode
        if params.debug_mode and frame is not None:
            # Calculate progress as percentage
            progress = min(hold_duration / params.yo_sign_hold_duration, 1.0) * 100
            
            # Draw progress bar
            bar_width = 200
            bar_height = 20
            bar_x = 10
            bar_y = 240
            
            # Background bar (gray)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            
            # Progress bar (green)
            progress_width = int((progress / 100) * bar_width)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
            
            # Text
            cv2.putText(frame, f"Hold: {progress:.0f}%", (bar_x + 70, bar_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Check if held long enough
        if hold_duration >= params.yo_sign_hold_duration:
            # Reset for next activation
            params.yo_sign_active = False
            params.yo_sign_start_time = 0
            params.activation_frames.clear()
            
            # Set state change time for lockout period
            params.activation_state_change_time = current_time
            
            # Toggle activation state
            return True
    
    return False

def draw_hand_landmarks(frame, hand_landmarks):
    """Enhanced visualization of hand landmarks with custom styling"""
    # Draw connections
    mp.solutions.drawing_utils.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
        mp.solutions.drawing_utils.DrawingSpec(color=(0, 121, 255), thickness=2)
    )
    
    # Add custom visualization for activation gesture
    landmarks = hand_landmarks.landmark
    h, w, _ = frame.shape
    
    # Get finger positions
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    
    # Check finger states using both simple and full methods
    index_extended_simple = is_finger_extended_simple(
        landmarks, 
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_MCP
    )
    
    index_extended_full = is_finger_extended(
        landmarks, 
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.INDEX_FINGER_PIP
    )
    
    index_extended = index_extended_simple or index_extended_full
    
    pinky_extended_simple = is_finger_extended_simple(
        landmarks, 
        mp_hands.HandLandmark.PINKY_TIP,
        mp_hands.HandLandmark.PINKY_MCP
    )
    
    pinky_extended_full = is_finger_extended(
        landmarks, 
        mp_hands.HandLandmark.PINKY_TIP,
        mp_hands.HandLandmark.PINKY_MCP,
        mp_hands.HandLandmark.PINKY_PIP
    )
    
    pinky_extended = pinky_extended_simple or pinky_extended_full
    
    middle_folded = is_finger_folded(
        landmarks,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP
    )
    
    ring_folded = is_finger_folded(
        landmarks,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_MCP
    )
    
    # Draw circles on index and pinky tips if extended (green)
    if index_extended:
        cv2.circle(frame, (int(index_tip.x * w), int(index_tip.y * h)), 12, (0, 255, 0), -1)
    
    if pinky_extended:
        cv2.circle(frame, (int(pinky_tip.x * w), int(pinky_tip.y * h)), 12, (0, 255, 0), -1)
    
    # Draw circles on middle and ring tips if folded (red)
    if middle_folded:
        cv2.circle(frame, (int(middle_tip.x * w), int(middle_tip.y * h)), 8, (0, 0, 255), -1)
    
    if ring_folded:
        cv2.circle(frame, (int(ring_tip.x * w), int(ring_tip.y * h)), 8, (0, 0, 255), -1)
    
    # Draw circle on thumb if overlapping middle/ring (blue)
    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
    thumb_to_middle = np.sqrt((thumb_tip.x - middle_pip.x)**2 + (thumb_tip.y - middle_pip.y)**2)
    thumb_to_ring = np.sqrt((thumb_tip.x - ring_pip.x)**2 + (thumb_tip.y - ring_pip.y)**2)
    
    if thumb_to_middle < 0.1 or thumb_to_ring < 0.1:
        cv2.circle(frame, (int(thumb_tip.x * w), int(thumb_tip.y * h)), 10, (255, 0, 0), -1)
    
    # If proper Yo sign, draw a line between index and pinky
    if index_extended and pinky_extended and middle_folded and ring_folded:
        cv2.line(frame, 
                (int(index_tip.x * w), int(index_tip.y * h)), 
                (int(pinky_tip.x * w), int(pinky_tip.y * h)), 
                (255, 0, 255), 3)
        
        # If in hold-to-activate mode, make the line pulse
        if params.yo_sign_active:
            pulse_intensity = int(127 * (1 + np.sin(time.time() * 10))) + 128
            cv2.line(frame, 
                    (int(index_tip.x * w), int(index_tip.y * h)), 
                    (int(pinky_tip.x * w), int(pinky_tip.y * h)), 
                    (pulse_intensity, 0, pulse_intensity), 5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    # Display status with smoother color transition
    if params.gesture_control_active:
        color = (0, 255, 0)  # Green when active
        status = "Active"
    else:
        color = (0, 0, 255)  # Red when inactive
        status = "Inactive"
    
    status_text = f"Gesture Control: {status}"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Add instruction text
    instruction = "Hold 'Yo' sign for 1 second to toggle control"
    cv2.putText(frame, instruction, (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks
            landmarks = hand_landmarks.landmark
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Check for activation gesture (Yo sign)
            if detect_activation_gesture(landmarks, hand_landmarks, frame if params.debug_mode else None):
                params.gesture_control_active = not params.gesture_control_active
                # Play a beep sound to indicate activation/deactivation
                if params.gesture_control_active:
                    print("\a")  # System beep
                continue

            if params.gesture_control_active:
                # Enhanced distance calculations
                pinch_distance = np.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)
                right_pinch_distance = np.sqrt((middle_tip.x - thumb_tip.x)**2 + (middle_tip.y - thumb_tip.y)**2)
                
                # Improved cursor positioning with better screen mapping
                # Use a smaller region of the frame for more precise control
                cursor_x = np.interp(index_tip.x, [0.2, 0.8], [0, screen_width])
                cursor_y = np.interp(index_tip.y, [0.15, 0.75], [0, screen_height])
                
                # Apply enhanced smoothing
                smooth_x, smooth_y = params.smooth_coordinates(cursor_x, cursor_y)

                # Enhanced right-click detection
                if right_pinch_distance < params.pinch_threshold:
                    pyautogui.rightClick()
                    time.sleep(0.3)  # Reduced delay
                    continue

                # Enhanced pinch detection for left-click and drag
                if pinch_distance < params.pinch_threshold:
                    current_time = time.time()
                    if current_time - params.last_pinch_time < params.double_click_threshold:
                        pyautogui.doubleClick()
                        params.last_pinch_time = 0
                        params.is_dragging = False
                    else:
                        if not params.is_dragging:
                            pyautogui.mouseDown()
                            params.is_dragging = True
                        params.last_pinch_time = current_time
                else:
                    if params.is_dragging:
                        pyautogui.mouseUp()
                        params.is_dragging = False
                    
                    # Move cursor only if not dragging
                    if not params.is_dragging:
                        pyautogui.moveTo(smooth_x, smooth_y)

            # Enhanced visualization with custom highlighting for the Yo sign
            draw_hand_landmarks(frame, hand_landmarks)

    cv2.imshow('Gesture Control', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
