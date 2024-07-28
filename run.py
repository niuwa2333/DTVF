import subprocess
import os

def install_requirements():
    try:
        subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'])
        print("Requirements installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        exit(1)

def run_app():
    try:
        subprocess.check_call(['python', 'app.py'])
    except subprocess.CalledProcessError as e:
        print(f"Error running app: {e}")
        exit(1)

if __name__ == "__main__":
    install_requirements()
    run_app()