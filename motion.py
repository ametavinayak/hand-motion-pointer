import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time
import ctypes
from enum import Enum
from ctypes import wintypes

# Initialize MediaPipe components
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
CLICK_DISTANCE_THRESHOLD = 0.05  # Distance threshold for click gesture
ZOOM_SENSITIVITY = 10  # Higher value = more zoom per pinch
SCROLL_SENSITIVITY = 15  # Higher value = more scroll per movement
SMOOTHING_FACTOR = 0.6  # Mouse movement smoothing factor
PINCH_HOLD_TIME = 0.3  # Time to hold pinch for click (seconds)

# Windows API for minimizing windows
user32 = ctypes.WinDLL('user32')
user32.GetForegroundWindow.restype = wintypes.HWND
user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)

class PointerMode(Enum):
    POINTING = 1
    CLICKING = 2
    ZOOMING = 3
    SCROLLING = 4
    MINIMIZE = 5
    IDLE = 6

class HandPointer:
    def __init__(self):
        self.hands = mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1
        )
        self.prev_x, self.prev_y = 0, 0
        self.pinch_start_time = 0
        self.mode = PointerMode.IDLE
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.click_cooldown = False
        self.cooldown_timer = 0
        self.prev_pinch_distance = 0
        self.prev_scroll_position = None
        self.three_finger_down_start = 0
        self.three_finger_active = False

    def process_hand_landmarks(self, landmarks):
        """Process hand landmarks and return relevant points"""
        return {
            'wrist': landmarks[0],
            'thumb_tip': landmarks[4],
            'index_tip': landmarks[8],
            'middle_tip': landmarks[12],
            'ring_tip': landmarks[16],
            'pinky_tip': landmarks[20],
            'index_pip': landmarks[6],
            'middle_pip': landmarks[10]
        }

    def is_pinch_gesture(self, index_tip, thumb_tip):
        """Check if the user is making a pinch gesture"""
        distance = math.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)
        return distance < CLICK_DISTANCE_THRESHOLD

    def is_pointing_gesture(self, landmarks):
        """Check if the user is making a pointing gesture"""
        return (
            landmarks['index_tip'].y < landmarks['index_pip'].y and  # Index extended
            landmarks['middle_tip'].y > landmarks['middle_pip'].y and  # Middle closed
            landmarks['ring_tip'].y > landmarks['middle_pip'].y and    # Ring closed
            landmarks['pinky_tip'].y > landmarks['middle_pip'].y       # Pinky closed
        )

    def is_scroll_gesture(self, landmarks):
        """Check if index and middle fingers are extended"""
        return (
            landmarks['index_tip'].y < landmarks['index_pip'].y and  # Index extended
            landmarks['middle_tip'].y < landmarks['middle_pip'].y and  # Middle extended
            landmarks['ring_tip'].y > landmarks['middle_pip'].y and    # Ring closed
            landmarks['pinky_tip'].y > landmarks['middle_pip'].y       # Pinky closed
        )

    def is_three_finger_down(self, landmarks):
        """Check if index, middle and ring fingers are extended downward"""
        return (
            landmarks['index_tip'].y > landmarks['index_pip'].y and  # Index down
            landmarks['middle_tip'].y > landmarks['middle_pip'].y and  # Middle down
            landmarks['ring_tip'].y > landmarks['middle_pip'].y and    # Ring down
            landmarks['pinky_tip'].y > landmarks['middle_pip'].y       # Pinky closed
        )

    def update_mouse_position(self, x, y):
        """Update mouse position with smoothing"""
        if self.prev_x == 0 and self.prev_y == 0:
            self.prev_x, self.prev_y = x, y
            
        smoothed_x = self.prev_x * SMOOTHING_FACTOR + x * (1 - SMOOTHING_FACTOR)
        smoothed_y = self.prev_y * SMOOTHING_FACTOR + y * (1 - SMOOTHING_FACTOR)
        
        pyautogui.moveTo(smoothed_x, smoothed_y)
        self.prev_x, self.prev_y = smoothed_x, smoothed_y

    def handle_zoom(self, index_tip, thumb_tip):
        """Handle zoom in/out based on pinch distance"""
        current_distance = math.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)
        
        if self.prev_pinch_distance == 0:
            self.prev_pinch_distance = current_distance
            return
            
        distance_change = current_distance - self.prev_pinch_distance
        zoom_amount = distance_change * ZOOM_SENSITIVITY * 100
        
        if abs(zoom_amount) > 1:  # Deadzone to prevent tiny movements
            pyautogui.keyDown('ctrl')
            pyautogui.scroll(int(zoom_amount))
            pyautogui.keyUp('ctrl')
        
        self.prev_pinch_distance = current_distance

    def handle_scroll(self, index_tip, middle_tip):
        """Handle scrolling based on finger movement"""
        current_position = (index_tip.y + middle_tip.y) / 2
        
        if self.prev_scroll_position is None:
            self.prev_scroll_position = current_position
            return
            
        position_change = current_position - self.prev_scroll_position
        scroll_amount = position_change * SCROLL_SENSITIVITY * 100
        
        if abs(scroll_amount) > 1:  # Deadzone to prevent tiny movements
            pyautogui.scroll(int(-scroll_amount))  # Negative because screen scroll is inverted
        
        self.prev_scroll_position = current_position

    def minimize_current_window(self):
        """Minimize the current foreground window"""
        hwnd = user32.GetForegroundWindow()
        user32.ShowWindow(hwnd, 6)  # 6 = SW_MINIMIZE

    def run(self):
        """Main application loop"""
        while self.running:
            success, image = self.cap.read()
            if not success:
                continue
                
            image = cv2.flip(image, 1)
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(image)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            self.mode = PointerMode.IDLE
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    landmarks = self.process_hand_landmarks(hand_landmarks.landmark)
                    
                    # Three finger down (minimize gesture)
                    if self.is_three_finger_down(landmarks):
                        current_time = time.time()
                        if not self.three_finger_active:
                            self.three_finger_down_start = current_time
                            self.three_finger_active = True
                        
                        if current_time - self.three_finger_down_start > 0.5:
                            self.mode = PointerMode.MINIMIZE
                            self.minimize_current_window()
                            self.three_finger_active = False
                            time.sleep(0.5)  # Prevent repeated triggers
                    else:
                        self.three_finger_active = False
                    
                    # Pointing gesture (cursor movement)
                    if self.is_pointing_gesture(landmarks):
                        self.mode = PointerMode.POINTING
                        x_pos = int(landmarks['index_tip'].x * SCREEN_WIDTH)
                        y_pos = int(landmarks['index_tip'].y * SCREEN_HEIGHT)
                        self.update_mouse_position(x_pos, y_pos)
                        
                        # Click on pinch
                        if self.is_pinch_gesture(landmarks['index_tip'], landmarks['thumb_tip']):
                            self.mode = PointerMode.CLICKING
                            current_time = time.time()
                            if not self.click_cooldown:
                                self.pinch_start_time = current_time
                                self.click_cooldown = True
                            
                            if current_time - self.pinch_start_time > PINCH_HOLD_TIME:
                                pyautogui.click()
                                self.click_cooldown = False
                                self.cooldown_timer = current_time
                        else:
                            self.click_cooldown = False
                    
                    # Scroll gesture (index + middle finger)
                    elif self.is_scroll_gesture(landmarks):
                        self.mode = PointerMode.SCROLLING
                        self.handle_scroll(landmarks['index_tip'], landmarks['middle_tip'])
                    
                    # Zoom gesture (pinch in/out while not pointing)
                    elif self.is_pinch_gesture(landmarks['index_tip'], landmarks['thumb_tip']):
                        self.mode = PointerMode.ZOOMING
                        self.handle_zoom(landmarks['index_tip'], landmarks['thumb_tip'])
                    else:
                        self.prev_pinch_distance = 0
                        self.prev_scroll_position = None
            
            # Display mode information
            mode_text = f"Mode: {self.mode.name}"
            cv2.putText(image, mode_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display instructions
            cv2.putText(image, "Point: Move cursor", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, "Pinch: Click", (10, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, "Pinch move: Zoom", (10, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, "Index+Middle: Scroll", (10, 160), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, "3 fingers down: Minimize", (10, 190), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, "Press 'Q' to quit", (10, 220), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Advanced Hand Pointer', image)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                self.running = False
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pointer = HandPointer()
    pointer.run()
