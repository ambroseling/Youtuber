import pyautogui
import time
import random
import string
import numpy as np
import math

# makes mistakes as it types
def type_code_naturally(code, speed=0.01):
    for char in code:

        # Randomly make a typo, correct it, and continue only if the char is a letter
        if char in string.ascii_letters and random.random() < 0.15:
            typo_char = random.choice(string.ascii_letters)  # Choose a random letter to simulate a typo
            pyautogui.typewrite(typo_char, interval=random.uniform(speed * 0.5, speed * 2))  # Type the random letter
            pyautogui.press('backspace')  # Press backspace to correct the typo
        
        # types out the actual Character
        pyautogui.typewrite(char, interval=random.uniform(speed * 0.5, speed * 2))


        time.sleep(random.uniform(speed * 0.5, speed * 1.5))  # Random delay between keystrokes


def bezier_curve(p0, p1, p2, t):
    return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
# tries to move naturally (isnt really working rn)
def move_and_click_mouse_naturally(target_x, target_y, speed=30.0):
    current_x, current_y = pyautogui.position()
    control_x = (current_x + target_x) / 2 + random.randint(-100, 100)
    control_y = (current_y + target_y) / 2 + random.randint(-100, 100)
    distance = np.hypot(target_x - current_x, target_y - current_y)
    steps = int(distance / 10)

    base_duration = max(0.01 / speed, 0.005)  # Ensure minimum duration for very high speeds
    base_delay = max(0.01 / speed, 0.005)

    for i in range(steps + 1):
        t = i / steps
        new_x = bezier_curve(current_x, control_x, target_x, t)
        new_y = bezier_curve(current_y, control_y, target_y, t)
        pyautogui.moveTo(new_x, new_y, duration=random.uniform(base_duration, base_duration * 2))
        time.sleep(random.uniform(base_delay, base_delay * 2))  # Add a small random delay to mimic human movement

    pyautogui.click(target_x, target_y, duration=random.uniform(0.05 / speed, 0.1 / speed))

# just draws a circle
def circle(X, Y, R=20, STEP = 50):

    for angle_deg in range(0, 360, STEP):
        angle_rad = math.radians(angle_deg)
        pyautogui.moveTo(
                X + R * math.cos(angle_rad),
                Y + R * math.sin(angle_rad))

# method: 
# able to take in multiple screenshots, and tries each different one out when one fails
def advancedMoveTo(lstImages, confidence=0.8):
    for image_path in lstImages:
        try:
            res = pyautogui.locateCenterOnScreen(image_path, confidence=confidence)
            if res:
                print(f"Image found at: {res}")
                pyautogui.click(res.x, res.y, duration= 0.8)
                return True  # Exit the function once an image is found
            else:
                print(f"Image not found on the screen: {image_path}")
        except pyautogui.ImageNotFoundException:
            print(f"Image could not be found: {image_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
    print("None of the images were found.")
    return False  # Return False if no image was found



















print('hello world!')
print('WE CAN NOW CODE!!!')










