import streamlit as st

import numpy as np
import pandas as pd
import os



st.markdown("""# That's how monet would have done it!
## transform any pic into a Monet painting
""")



uploaded_file = st.file_uploader("Choose a jpg file", type="jpg")

image_path = uploaded_file
#st.echo(image_path)
st.write(image_path)

from PIL import Image
if image_path:
    image = Image.open(image_path)
    st.image(image, caption='Image before transformation')

image = Image.open(image_path)
st.image(image, caption='Image after transformation')

# def file_selector(folder_path='.'):
#     filenames = os.listdir(folder_path)
#     selected_filename = st.selectbox('Select a file', filenames)
#     return os.path.join(folder_path, selected_filename)


# if __name__ == '__main__':
#     # Select a file
#     if st.checkbox('Select a file in current directory'):
#         folder_path = '.'
#         if st.checkbox('Change directory'):
#             folder_path = st.text_input('Enter folder path', '.')
#         filename = file_selector(folder_path=folder_path)
#         st.write('You selected `%s`' % filename)