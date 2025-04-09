import streamlit as st
from PIL import Image
import os
import time

# Folder containing the character images
image_folder = "sign_folder"

# Input sentence
input_sentence = "HELLO"

# Streamlit app
def main():
    st.title("Sign Language Video Stream")

    # Input sentence from the user
    sentence = st.text_input("Enter a sentence:", input_sentence)

    # Button to start the video stream
    if st.button("Start Streaming"):
        # Placeholder for the video stream
        image_placeholder = st.empty()

        # Loop through each character in the sentence
        for char in sentence.upper():
            if char == " ":
                char = "space"  # Assuming the space image is named "space.jpg"
            image_path = os.path.join(image_folder, f"{char}.jpg")

            try:
                # Open the image using PIL
                img = Image.open(image_path)

                # Display the image in the placeholder
                image_placeholder.image(img, caption=f"Character: {char}", use_container_width=True)

                # Wait for 1 second
                time.sleep(1)

            except FileNotFoundError:
                st.error(f"Image for character '{char}' not found.")
            except Exception as e:
                st.error(f"Error processing image for character '{char}': {e}")

        # Clear the placeholder after streaming is done
        image_placeholder.empty()
        st.success("Streaming completed!")

# Run the Streamlit app
if __name__ == "__main__":
    main()