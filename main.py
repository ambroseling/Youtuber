import time
import os
import argparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Set up argument parser
parser = argparse.ArgumentParser(description="YouTube Auto Upload Bot")
parser.add_argument('answer', type=int, choices=[1, 2], help="Press 1 if you want to spam same video or Press 2 if you want to upload multiple videos")
parser.add_argument('video_name', type=str, help="Put the name of the video you want to upload (Ex: vid.mp4 or myshort.mp4 etc..)")
parser.add_argument('howmany', type=int, nargs='?', default=1, help="How many times you want to upload this video (only for spamming the same video)")
args = parser.parse_args()

options = Options()
options.add_argument("--log-level=3")
options.add_argument("user-data-dir=/home/myname/.config/google-chrome-beta")
options.binary_location = "/usr/bin/google-chrome-beta"

# Update this path to where your chromedriver.exe is located for Chrome Beta
chromedriver_path = "/home/myname/projects/youtube-autoupload-bot/chromedriver.exe"

if args.answer == 1:
    nameofvid = args.video_name
    howmany = args.howmany

    for i in range(howmany):
        service = Service(chromedriver_path)
        bot = webdriver.Chrome(service=service, options=options)

        bot.get("https://studio.youtube.com")
        time.sleep(3)
        upload_button = bot.find_element(By.XPATH, '//*[@id="upload-icon"]')
        upload_button.click()
        time.sleep(1)

        file_input = bot.find_element(By.XPATH, '//*[@id="content"]/input')
        simp_path = 'videos/{}'.format(str(nameofvid))
        abs_path = os.path.abspath(simp_path)
        file_input.send_keys(abs_path)

        time.sleep(7)

        next_button = bot.find_element(By.XPATH, '//*[@id="next-button"]')
        for _ in range(3):
            next_button.click()
            time.sleep(1)

        done_button = bot.find_element(By.XPATH, '//*[@id="done-button"]')
        done_button.click()
        time.sleep(5)
        bot.quit()

elif args.answer == 2:
    print("\033[1;31;40m IMPORTANT: Please make sure the name of the videos are like this: vid1.mp4, vid2.mp4, vid3.mp4 ...  etc")
    dir_path = './videos'
    count = 0

    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
    print("   ", count, " Videos found in the videos folder, ready to upload...")
    time.sleep(6)

    for i in range(count):
        service = Service(chromedriver_path)
        bot = webdriver.Chrome(service=service, options=options)

        bot.get("https://studio.youtube.com")
        time.sleep(3)
        upload_button = bot.find_element(By.XPATH, '//*[@id="upload-icon"]')
        upload_button.click()
        time.sleep(1)

        file_input = bot.find_element(By.XPATH, '//*[@id="content"]/input')
        simp_path = 'videos/vid{}.mp4'.format(str(i + 1))
        abs_path = os.path.abspath(simp_path)
        
        file_input.send_keys(abs_path)

        time.sleep(7)

        next_button = bot.find_element(By.XPATH, '//*[@id="next-button"]')
        for _ in range(3):
            next_button.click()
            time.sleep(1)

        done_button = bot.find_element(By.XPATH, '//*[@id="done-button"]')
        done_button.click()
        time.sleep(5)
        bot.quit()
