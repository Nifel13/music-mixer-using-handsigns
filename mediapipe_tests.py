import cv2
import mediapipe as mp
from playsound import playsound
import os
from pyo import *
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

sound_folder = "audios"
song_filename = "drugs.wav"  
song_path = os.path.join(sound_folder, song_filename)

# functions outside of mainloop

def draw_handline(rightx,righty, leftx, lefty, image, color = (255,0,0), width = 2):
	cv2.line(image, (int(leftx * image.shape[1]), int(lefty * image.shape[0])),
						 (int(rightx * image.shape[1]), int(righty * image.shape[0])),
						 color, width)
	

s = Server().boot()
s.start()

vol = SigTo(value=0.5, time=0.5)
transpo = SigTo(value=1, time=0.2)
def create_chain(filename):
    global vol, transpo
    sf = SfPlayer(filename, speed=1.0, loop=True)
    fft = PVAnal(sf, size=1024)
    pitch = PVTranspose(fft, transpo=transpo)
    synth = PVSynth(pitch)
    amp = synth * vol
    amp.out()
    return sf, pitch, amp
# define all sounfiles

playlist = os.listdir("audios")
track = 0

# finger positions

left_index = None
right_index = None
left_thumb = None
right_thumb = None

# sound control

sf, pitch, amp = create_chain(f"audios/{playlist[track]}")

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
	model_complexity=0,
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5) as hands:
	while cap.isOpened():
		success, image = cap.read()
		if not success:
			print("Ignoring empty camera frame.")
			# If loading a video, use 'break' instead of 'continue'.
			continue

		# To improve performance, optionally mark the image as not writeable to
		# pass by reference.
		image.flags.writeable = False
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		results = hands.process(image)

		# Draw the hand annotations on the image.
		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		if results.multi_hand_landmarks:
			for i,hand_landmarks in enumerate(results.multi_hand_landmarks):
				# mp_drawing.draw_landmarks(
				# 	image,
				# 	hand_landmarks,
				# 	mp_hands.HAND_CONNECTIONS,
				# 	mp_drawing_styles.get_default_hand_landmarks_style(),
				# 	mp_drawing_styles.get_default_hand_connections_style())
				# if the hand is the right hand, draw a line in the middle of the screen
				# get the right index finger tip and left index finger tip
				hand_label = results.multi_handedness[i].classification[0].label
				if hand_label == 'Left':
					left_index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
					left_thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
					
				elif hand_label == 'Right':
					right_index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
					right_thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
					
			if left_index and right_index:
				draw_handline((right_index.x+right_thumb.x)/2, (right_index.y+right_thumb.y)/2,
				    (left_index.x+left_thumb.x)/2, (left_index.y+left_thumb.y)/2, image)
				
				dist_index = np.sqrt((right_index.x-left_index.x)**2+(right_index.y-left_index.y)**2)
				print(dist_index)
				vol.value = float(dist_index)

			if left_thumb and right_thumb:
				draw_handline(right_index.x, right_index.y, right_thumb.x, right_thumb.y, image)
				draw_handline(left_index.x, left_index.y, left_thumb.x, left_thumb.y, image)

				dist_left = np.sqrt((left_index.x-left_thumb.x)**2+(left_index.y-left_thumb.y)**2)
				pitch.transpo = float(dist_left)*5

				dist_right = np.sqrt((right_index.x-right_thumb.x)**2+(right_index.y-right_thumb.y)**2)
				sf.speed = 1+float(dist_right)

				if left_index.y> left_thumb.y:
					amp.stop()
					track += 1
					sf, pitch, amp = create_chain(f"audios/{playlist[track]}")
					time.sleep(0.7)

				if right_index.y> right_thumb.y:
					amp.stop()
					track += -1
					sf, pitch, amp = create_chain(f"audios/{playlist[track]}")
					time.sleep(0.7)
					
		# Flip the image horizontally for a selfie-view display.
		cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
		if cv2.waitKey(5) & 0xFF == 27:
			break
cap.release()
s.stop()
s.shutdown()
