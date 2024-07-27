# youtube-autoupload-bot
This document provides step-by-step instructions to set up Google Chrome Beta version 127.0.6533.57 
and ChromeDriver on an Ubuntu system. It also includes a Python script for automating video uploads to YouTube.

Step 1: Add Google Repository and Install Google Chrome Beta
1.1 Add Google Repository
This command adds the Google repository to your system's package sources:
For this you need to be in: the usr/bin/ directory, like so:
    cd ~
    cd /usr/bin/
Then run these commands:

    wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
    sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'

1.2 Update Package List
Update your package list to include the new repository:

    sudo apt-get update

1.3 Install the specific version of Google Chrome Beta (127.0.6533.57):
    1. Find the available versions:
        apt-cache showpkg google-chrome-beta
    2. Install the specific version: (this is easier, as the chrome driver is already in the repo)
        sudo apt-get install google-chrome-beta=127.0.6533.57-1 
1.4 Verify the Installation
Verify the installation by checking the version of Google Chrome Beta:

    google-chrome-beta --version

The output should be Google Chrome 127.0.6533.57 beta.

if you downloaded the same chrome version as above, as the chrome driver is already in the repo
Give the chromedriver permissions:
    chmod +x /home/myname/projects/youtube-autoupload-bot/chromedriver.exe

Change the following paths in the main.py file (change where it says "myname"):
                                                ##
    options.add_argument("user-data-dir=/home/myname/.config/google-chrome-beta")
    options.binary_location = "/usr/bin/google-chrome-beta" #(dont change this)

    # Update this path to where your chromedriver.exe is located for Chrome Beta
                                ##
    chromedriver_path = "/home/myname/projects/youtube-autoupload-bot/chromedriver.exe"

search chrome beta ubuntu in your computer search (where it shows all the applications in your computer)
open and log in to your youtube page.

then try running the code like so:
    python main.py 1 vid1.mp4 1
or you can just do this command and go through the options yourself:
    python main.py

this should upload 1 video to the channel you signed in with