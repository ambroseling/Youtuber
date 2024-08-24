
import subprocess
import pyautogui
import time
# Given a list of images, loops till it finds the image and clicks on it
def advancedMoveTo(list_Images, confidence=0.8):
    '''
    list_Images: list of images is sequentially looped. When one image is not found, goes to the next one
                 When it finds the image, moves to it at a speed of .5 second and single left clicks it
    
    condifence: Confidence when finding and clicking an image.
    '''
    for image_path in list_Images:
        try:
            res = pyautogui.locateCenterOnScreen(image_path, confidence=confidence)
            if res:
                print(f"Image found at: {res}")
                pyautogui.click(res.x, res.y, duration= .5)
                return True  # Exit the function once an image is found
            else:
                print(f"Image not found on the screen: {image_path}")
        except pyautogui.ImageNotFoundException:
            print(f"Image could not be found: {image_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
    print("None of the images were found.")
    return False  # Return False if no image was found

def image_exists(image_path, confidence = .8):
    try:
        res = pyautogui.locateCenterOnScreen(image_path, confidence=confidence)
        if res:
            return True  # Exit the function once an image is found
        else:
            return False
    except pyautogui.ImageNotFoundException:
        print(f"Image could not be found: {image_path}")

# Opens vs code by doing ctrl+shift+n given you ran the code in terminal
def open_vscode():
    '''
    Opens vs code by doing ctrl+shift+n given you ran the code in terminal
    '''
    pyautogui.hotkey('ctrl', 'shift', 'n')
    time.sleep(4)  # Wait for VSCode to open
    # pyautogui.press('enter', interval=0.1)

def maximize_vscode():
    """
    maximizes the vs_code window
    """
    # Simulate pressing just F11 to enter full screen
    pyautogui.click(pyautogui.locateCenterOnScreen("vs_pics/maximize.png"), duration = .4)
    time.sleep(2)  # Wait for the action to complete

def create_project_folder(folder_name):
    
    pyautogui.hotkey('ctrl', 'k', 'o')
    time.sleep(2)
    # pyautogui.click(pyautogui.locateCenterOnScreen("vs_pics/folder_window.png", confidence=0.8), duration = 1)
    pyautogui.hotkey('ctrl', 'shift', 'n')
    time.sleep(1)
    pyautogui.typewrite(folder_name, interval=0.1)
    pyautogui.press('enter')
    time.sleep(2)
    if image_exists("vs_pics/confirm_folder_replace.png"):
        time.sleep(1)
        pyautogui.press('enter')
        time.sleep(3)
        pyautogui.click(pyautogui.locateCenterOnScreen("vs_pics/folder_window.png", confidence=.5), duration=.5)
        time.sleep(1)
        pyautogui.hotkey('ctrl', 'a')
        pyautogui.press('backspace')
        pyautogui.typewrite(folder_name, interval=0.1)
        pyautogui.press('enter')
        pyautogui.press('enter')

    else:
        pyautogui.click(pyautogui.locateCenterOnScreen("vs_pics/select_folder.png", confidence=.6), duration=.5)
    time.sleep(1)

def create_new_file(file_name):
    pyautogui.hotkey('ctrl', 'shift', 'p')
    time.sleep(1)
    pyautogui.typewrite('new file', interval=0.1)
    pyautogui.press('enter', interval=.1)
    time.sleep(1)
    pyautogui.typewrite(file_name, interval=0.1)
    time.sleep(1)
    pyautogui.press('enter', interval=.1)
    time.sleep(1)
    pyautogui.press('enter', interval=.1)
    time.sleep(1)

def test_typing():
    pyautogui.typewrite('Hello, World!', interval=0.1)

def new_terminal():
    pyautogui.hotkey('ctrl', 'shift', 'p')
    time.sleep(1)
    pyautogui.typewrite('New Terminal', interval=0.1)
    pyautogui.press('enter')
