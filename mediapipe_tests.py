import cv2
import mediapipe as mp
import os
from pyo import *
import numpy as np
from spectrum_analyzer import SimpleSpectrumAnalyzer

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# functions outside of mainloop

def draw_handline(rightx,righty, leftx, lefty, image, color = (255,0,0), width = 2):
	cv2.line(image, (int(leftx * image.shape[1]), int(lefty * image.shape[0])),
						 (int(rightx * image.shape[1]), int(righty * image.shape[0])),
						 color, width)

def draw_spectrum(rightx, righty, leftx, lefty, image, color=(255, 0, 0), width=2, spectrum=None, compression_factor=4):
	if spectrum is None or len(spectrum) == 0:
		return

	# Correct distance calculation
	spectrum_compressed = spectrum[::compression_factor]
	num_bars = len(spectrum_compressed)

    # Calculate start and end pixel positions
	x_start = int(round(leftx * image.shape[1]))
	y_base = int(round(lefty * image.shape[0]))
	x_end = int(round(rightx * image.shape[1]))


	dist_y = (righty - lefty)
	dist_z = (rightx - leftx)
	theta = np.arctan2(dist_z, dist_y)

	# print(f"dist_y: {dist_y}, dist_z: {dist_z}, theta: {theta}")

	# Calculate the interval between bars
	if num_bars > 1:
		interval = (x_end - x_start) / (num_bars - 1)
	else:
		interval = 0

	for i in range(num_bars):
		hsv_color = np.uint8([[[i, 160, 255]]])
		bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0].tolist()
		x = int(round(x_start + i * interval))
		y = y_base
		bar_height = int(round(spectrum_compressed[i])**2*0.0002)
		y2 = max(0, y - bar_height)
		invy2 = min(image.shape[0], y + bar_height)
		if 0 <= x < image.shape[1]:
			# Create an overlay
			overlay = image.copy()
			prime_z = int((interval * i)/(dist_z/(dist_y + 0.000001)))
			cv2.line(overlay, (x, y+prime_z), (x, y2+prime_z), bgr_color, width)
			cv2.line(overlay, (x, y+prime_z), (x, invy2+prime_z), bgr_color, width)
			# Blend overlay with the original image (alpha=0.3 for transparency)
			alpha = 0.6
			image[:] = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


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

# Create analyzer
analyzer = SimpleSpectrumAnalyzer(sf)
analyzer.start()
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
	model_complexity=0,
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5) as hands:
	while cap.isOpened():
		success, image = cap.read()
		if not success:
			# print("Ignoring empty camera frame.")
			# If loading a video, use 'break' instead of 'continue'.
			continue

		image_flipped = cv2.flip(image, 1)
		cv2.putText(image_flipped, f"Now Playing: {playlist[track%len(playlist)]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 173, 101), 2)
		# Flip back to original orientation
		image = cv2.flip(image_flipped, 1)

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
				# draw_handline((right_index.x+right_thumb.x)/2, (right_index.y+right_thumb.y)/2,
				#     (left_index.x+left_thumb.x)/2, (left_index.y+left_thumb.y)/2, image)
				
				dist_index = np.sqrt((right_index.x-left_index.x)**2+(right_index.y-left_index.y)**2)
				vol.value = float(dist_index)

			if left_thumb and right_thumb:

				spectrum = analyzer.get_spectrum()  # List of floats
				if spectrum is not None:
					draw_spectrum((right_index.x+right_thumb.x)/2, (right_index.y+right_thumb.y)/2,
				    (left_index.x+left_thumb.x)/2, (left_index.y+left_thumb.y)/2, image, spectrum=spectrum)

				draw_handline(right_index.x, right_index.y, right_thumb.x, right_thumb.y, image, color=(255, 50, 50))
				draw_handline(left_index.x, left_index.y, left_thumb.x, left_thumb.y, image, color=(50, 50, 255))

				dist_left = np.sqrt((left_index.x-left_thumb.x)**2+(left_index.y-left_thumb.y)**2)
				pitch.transpo = float(dist_left)*5

				dist_right = np.sqrt((right_index.x-right_thumb.x)**2+(right_index.y-right_thumb.y)**2)
				sf.speed = 1+float(dist_right)

				if left_index.y> left_thumb.y:
					amp.stop()
					track += 1
					sf, pitch, amp = create_chain(f"audios/{playlist[track%len(playlist)]}")
					time.sleep(0.7)

				if right_index.y> right_thumb.y:
					amp.stop()
					track += -1
					sf, pitch, amp = create_chain(f"audios/{playlist[track%len(playlist)]}")
					time.sleep(0.7)
					
		# Flip the image horizontally for a selfie-view display.
		cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
		if cv2.waitKey(5) & 0xFF == 27:
			break
cap.release()
s.stop()
s.shutdown()
