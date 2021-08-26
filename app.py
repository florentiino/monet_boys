import streamlit as st

import numpy as np
import pandas as pd
import os



# st.markdown("""# That's how monet would have done it!
# ## transform any pic into a Monet painting
# """)



# uploaded_file = st.file_uploader("Choose a jpg file", type="jpg")

# image_path = uploaded_file
# #st.echo(image_path)
# st.write(image_path)

# from PIL import Image
# if image_path:
#     image = Image.open(image_path)
#     st.image(image, caption='Image before transformation')

# image = Image.open(image_path)
# st.image(image, caption='Image after transformation')

#SELECT A FILE 
# def file_selector(folder_path='.'):
#     filenames = os.listdir(folder_path)
#     selected_filename = st.selectbox('Select a file', filenames)
#     # st.write(os.path.join(folder_path, selected_filename))
#     return os.path.join(folder_path, selected_filename)

                                                       
# if __name__ == '__main__':
#     if st.checkbox('Select an image'):
#         folder_path = '.'
#     if st.checkbox('Change directory'):
#         folder_path = st.text_input('Enter folder path', '.')
#         filename = file_selector(folder_path=folder_path)
#         st.write(type(filename))
#         st.write('You selected `%s`' % filename)                                                        
                                                    

   
#########UPLOAD TO GCP#####################################        
# import io 
# from io import BytesIO
# from google.cloud import storage


# client = storage.Client.from_service_account_json("batch-672-gan-monet-59f43b305ef6.json")
# BUCKET_NAME = 'bucket-monet-gan'
# bucket = client.get_bucket(BUCKET_NAME)  
# blob = bucket.blob(filename)    

# with open(filename, 'rb') as f: #open file as binary - create an object of file
#     blob.upload_from_file(f)   
          

# def upload_to_bucket(blob_name, path_to_file, bucket_name):
# """ Upload data to a google cloud bucket and get public URL"""
# # Explicitly use service account credentials by specifying the private key
# # file.
# storage_client = storage.Client.from_service_account_json(
#     'XXXXXX_gcloud.json'
# )
# # print(buckets = list(storage_client.list_buckets())
# bucket = storage_client.get_bucket(bucket_name)
# blob = bucket.blob(blob_name)
# blob.upload_from_filename(path_to_file)
# # returns a public url
# blob.make_public()
# return blob.public_url


from google.cloud import storage
from google.cloud.storage import bucket

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'batch-672-gan-monet-59f43b305ef6.json'

storage_client = storage.Client()

bucket_name = 'bucket-monet-gan'
bucket = storage_client.bucket(bucket_name)

vars(bucket)

'''Accessing the bucket'''
my_bucket = storage_client.get_bucket(bucket_name)

'''Uploading files'''
def upload_to_bucket(blob_name, file_path, bucket_name):
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)
        return True

    except:
        return False


file_path = r'../raw_data/test/'
#upload_to_bucket('test_monet', os.path.join(file_path, 'test_monet.jpg'), 'bucket-monet-gan')




def hauptsache():
    #print(dir(storage_client)) #print bucket details
    #print(vars(my_bucket)) # print bucket details
    upload_to_bucket('frontend_upload_images/test_frontend', os.path.join(file_path,
                     'impression_Sunrise.jpg'), 'bucket-monet-gan')


if __name__ == '__main__':
    hauptsache()
    
  