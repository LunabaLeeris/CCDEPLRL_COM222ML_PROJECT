import cv2
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk, ImageOps
import threading
import time
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from rembg import remove
from PIL import Image
from playsound import playsound
import pyttsx3
import io
import pygame

# Load the trained model once at the top of your script
model = load_model("rock_paper_scissors_cnn.h5")

# Class labels mapping
labels = {0: "rock", 1: "paper", 2: "scissors"}

# Make output directories
os.makedirs("../left", exist_ok=True)
os.makedirs("../right", exist_ok=True)

# Global variables
left_score = 0
right_score = 0
left_action = ""
right_action = ""
running = False

# Engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

pygame.mixer.init()
pygame.mixer.music.load("sound/theme_song.wav")
pygame.mixer.music.set_volume(0.5)
pygame.mixer.music.play(-1)


# Sound
sounds = {
    "ready": pygame.mixer.Sound("sound/ready.wav"),
    "rock": pygame.mixer.Sound("sound/rock.wav"),
    "paper": pygame.mixer.Sound("sound/paper.wav"),
    "scissor": pygame.mixer.Sound("sound/scissor.wav"),
    "shoot": pygame.mixer.Sound("sound/shoot.wav"),
    "left_wins": pygame.mixer.Sound("sound/left_wins.wav"),
    "right_wins": pygame.mixer.Sound("sound/right_wins.wav"),
    "tie": pygame.mixer.Sound("sound/tie.wav"),
}

def predict_action(image_path, left=True):
    # Remove background using rembg
    with open(image_path, "rb") as input_file:
        input_data = input_file.read()
        output_data = remove(input_data)

    # Load the transparent image
    img = Image.open(io.BytesIO(output_data)).convert("RGBA")

    # Create a white background image
    white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(white_bg, img).convert("RGB")  # Merge and convert to RGB

    # Apply conditional rotation and flip
    if left:
        img = img.rotate(90, expand=True)
    else:
        img = img.rotate(-90, expand=True)
        img = ImageOps.mirror(img)

    # Save (optional)
    img.save(image_path)

    # Resize and preprocess
    img = img.resize((150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    return labels[class_index]


# Determine winner
def decide_winner(left, right):
    if left == right:
        return "draw"
    elif (left == "rock" and right == "scissors") or \
         (left == "scissors" and right == "paper") or \
         (left == "paper" and right == "rock"):
        return "left"
    else:
        return "right"

# Start game logic
def start_game():
    global running
    if running:
        return
    running = True
    threading.Thread(target=game_round).start()

def game_round():
    global left_score, right_score, left_action, right_action, running

    pygame.mixer.music.pause()  # 🔇 Pause theme song

    status_label.config(text="Ready...", fg="orange")
    sounds["ready"].play()
    time.sleep(1)

    status_label.config(text="Rock...", fg="orange")
    sounds["rock"].play()
    time.sleep(1)

    status_label.config(text="Paper...", fg="orange")
    sounds["paper"].play()
    time.sleep(1)

    status_label.config(text="Scissors...", fg="orange")
    sounds["scissor"].play()
    time.sleep(1)

    status_label.config(text="SHOOOT!", fg="orange")
    sounds["shoot"].play()
    time.sleep(1)

    status_label.config(text="Judging...", fg="orange")
    # Capture frame and crop
    ret, frame = cap.read()
    if not ret:
        running = False
        return

    h, w, _ = frame.shape
    (left_x1, left_y1, left_x2, left_y2), (right_x1, right_y1, right_x2, right_y2) = get_bounding_boxes(w, h)

    left_box = frame[left_y1:left_y2, left_x1:left_x2]
    right_box = frame[right_y1:right_y2, right_x1:right_x2]

    left_path = f"left/left_{int(time.time())}.png"
    right_path = f"right/right_{int(time.time())}.png"

    cv2.imwrite(left_path, left_box)
    cv2.imwrite(right_path, right_box)

    left_action = predict_action(left_path)
    right_action = predict_action(right_path, left=False)

    winner = decide_winner(left_action, right_action)

    if winner == "left":
        left_score += 1
        status_label.config(text="Left Wins!", fg="green")
        sounds["left_wins"].play()

    elif winner == "right":
        right_score += 1
        status_label.config(text="Right Wins!", fg="yellow")
        sounds["right_wins"].play()

    else:
        status_label.config(text="It's a Tie!", fg="gray")
        sounds["tie"].play()

    tts_engine.runAndWait()  # Speak the winner
    update_labels()
    running = False

    pygame.mixer.music.unpause()  # 🟢 Resume theme song

def update_labels():
    left_action_label.config(text=f"Action: {left_action}")
    left_score_label.config(text=f"Score: {left_score}")
    right_action_label.config(text=f"Action: {right_action}")
    right_score_label.config(text=f"Score: {right_score}")

def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    # Draw bounding boxes
    # Draw bounding boxes with padding
    box_width, box_height = 300, 350
    box_spacing = 100  # space between left and right box
    frame_height, frame_width, _ = frame.shape

    # Total width occupied by both boxes and spacing
    total_width = 2 * box_width + box_spacing

    # Starting X coordinate to center everything
    start_x = (frame_width - total_width) // 2

    # Left and right box position
    (left_x1, left_y1, left_x2, left_y2), (right_x1, right_y1, right_x2, right_y2) = get_bounding_boxes(frame_width, frame_height)

    # Draw centered boxes
    cv2.rectangle(frame, (left_x1, left_y1), (left_x2, left_y2), (0, 255, 0), 2)
    cv2.rectangle(frame, (right_x1, right_y1), (right_x2, right_y2), (0, 255, 255), 2)

    # Convert to RGB and display in Tkinter
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, update_frame)


def get_bounding_boxes(frame_width, frame_height):
    box_width, box_height = 300, 350
    box_spacing = 50  # space between left and right box

    total_width = 2 * box_width + box_spacing
    start_x = (frame_width - total_width) // 2

    left_x1 = start_x + 25
    left_y1 = 100
    left_x2 = left_x1 + box_width - 25
    left_y2 = left_y1 + box_height

    right_x1 = left_x2 + box_spacing
    right_y1 = 100
    right_x2 = right_x1 + box_width - 25
    right_y2 = right_y1 + box_height

    return (left_x1, left_y1, left_x2, left_y2), (right_x1, right_y1, right_x2, right_y2)

# GUI Setup
root = tk.Tk()
root.title("Rock Paper Scissors Game")
root.configure(bg="#1e1e1e")  # dark background
root.geometry("800x700")     # optional fixed size for consistency

# Top header label
status_label = Label(
    root,
    text="ROCK PAPER SCISSORS",
    font=("Helvetica", 24, "bold"),
    fg="#ffa500",   # orange
    bg="#1e1e1e"
)
status_label.pack(pady=(20, 10))

# Video feed label
video_label = Label(root, bg="#1e1e1e")
video_label.pack(pady=10)

# Frame for scores and actions
score_frame = tk.Frame(root, bg="#1e1e1e")
score_frame.pack(pady=20)

label_style = {"font": ("Helvetica", 14, "bold"), "bg": "#1e1e1e", "fg": "#ffa500"}

# Left side labels
left_action_label = Label(score_frame, text="Action: ", **label_style)
left_action_label.grid(row=0, column=0, padx=40, pady=5)

left_score_label = Label(score_frame, text="Score: 0", **label_style)
left_score_label.grid(row=1, column=0, padx=40, pady=5)

# Styled Start button
start_button = Button(
    score_frame,
    text="Start Game",
    font=("Helvetica", 14, "bold"),
    bg="#ffa500",
    fg="#1e1e1e",
    relief="raised",
    activebackground="#ffcc00",
    activeforeground="black",
    borderwidth=3,
    width=12,
    command=start_game
)
start_button.grid(row=0, column=1, rowspan=2, padx=40, pady=10)

# Right side labels
right_action_label = Label(score_frame, text="Action: ", **label_style)
right_action_label.grid(row=0, column=2, padx=40, pady=5)

right_score_label = Label(score_frame, text="Score: 0", **label_style)
right_score_label.grid(row=1, column=2, padx=40, pady=5)

# Start video capture
cap = cv2.VideoCapture(0)

update_frame()
root.mainloop()

# Release webcam after GUI closes
cap.release()
cv2.destroyAllWindows()
