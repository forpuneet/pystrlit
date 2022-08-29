import streamlit as st
PAGE_CONFIG = {"page_title":"StColab.io","page_icon":":smiley:","layout":"centered"}
#st.beta_set_page_config(**PAGE_CONFIG)
def main():
	st.title("Gigmo Solutions - Remote Identity Verification")
	#st.subheader("Yeah... trying Mini project")
	menu = ["Remote Identity Verification","onboarding"]
	choice = st.sidebar.selectbox('Menu',menu)
	if choice == 'Home':
		st.subheader("Tweet Sentiment Analysis")	
if __name__ == '__main__':
	main()
