# main.py
import subprocess
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Face Registration and Recognition System")
    parser.add_argument('--action', type=str, required=True, choices=['add_face', 'recognize_face'], help="Choose the action: 'add_face' to register a new face or 'recognize_face' for real-time recognition.")

    args = parser.parse_args()

    if args.action == 'add_face':
        print("Starting face registration process...")
        subprocess.run([sys.executable, 'add_face.py'])
    elif args.action == 'recognize_face':
        print("Starting face recognition process...")
        subprocess.run([sys.executable, 'recognize_face.py'])

if __name__ == "__main__":
    main()
