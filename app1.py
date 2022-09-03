import streamlit as st
import cv2
import numpy as np


PAGE_CONFIG = {"page_title":"StColab.io","page_icon":":smiley:","layout":"centered"}
#st.beta_set_page_config(**PAGE_CONFIG)
def main():
	st.title("Gigmo Solutions - Remote Identity Verification")
	#st.subheader("Yeah... trying Mini project")
	menu = ["Remote Identity Verification","onboarding"]
	choice = st.sidebar.selectbox('Menu',menu)
	if choice == 'onboarding':
		st.subheader("Onboarding Initiation")	
		picture = st.camera_input("Take a picture")
		if picture:
     			st.image(picture)
	
	
	
	cap = cv2.VideoCapture(1)

	currentFrame = 0
	while(True):
    		print(currentFrame)
   
    		ret, frame = cap.read()
		frame = cv2.flip(frame,1)
		cv2.imshow('frame',frame)
    		if cv2.waitKey(1) & 0xFF == ord('q'):
        		break
		currentFrame += 1
cap.release()
cv2.destroyAllWindows()
	
if __name__ == '__main__':
	main()
