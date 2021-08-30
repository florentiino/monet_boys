import os
import streamlit as st
from PIL import Image
from google.cloud import storage

    

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


if __name__ == '__main__':
    # Select a file
    st.markdown("""# Transform any image into a monet-painting
    ## Upload an image
    """)
    
    
    
    if st.checkbox('Select a file in current directory'):
        folder_path = '.'
        if st.checkbox('Change directory'):
            folder_path = st.text_input('Enter folder path', '.')
        filepath = file_selector(folder_path=folder_path)
        st.write('You selected `%s`' % filepath)
        
        if filepath.endswith("jpg"):
            image = Image.open(filepath)
            st.image(image, caption='Uploaded File', use_column_width=False)
            
       
        
        # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'batch-672-gan-monet.json'

        # storage_client = storage.Client()

        # bucket_name = 'bucket-monet-gan'


        # #Accessing the bucket
        
        # my_bucket = storage_client.get_bucket(bucket_name)

        # #Uploading files
        
        # def upload_to_bucket(blob_name, file_path, bucket_name):
        #     try:
        #         bucket = storage_client.get_bucket(bucket_name)
        #         blob = bucket.blob(blob_name)
        #         blob.upload_from_filename(file_path)
        #         return True

        #     except Exception as e:
        #         print(e)
        #         return False
       
        # #calling the upload function   
        
        # filename_upload = 'frontend_upload_images/streamlit_test2'
        
        # upload_to_bucket(filename_upload,filepath,bucket_name)

        # #downloading file from bucket 
        # def download_from_bucket(blob_name, file_path, bucket_name):
        #     try:
        #         bucket = storage_client.get_bucket(bucket_name)
        #         blob = bucket.blob(blob_name)
        #         with open(file_path, 'wb') as f:
        #             storage_client.download_blob_to_file(blob, f)
        #         return True

        #     except Exception as e:
        #         print(e)
        #         return False
            
        # #calling the download function
        # filename_download = 'impression_Sunrise.jpg'
        # folder_name = 'frontend_download_images'
        # download_from_bucket('frontend_download_images/impression_Sunrise.jpg',os.getcwd(),bucket_name)
        
        # image2 = Image.open(filename_download)
        # st.image(image2, caption='DownloadedImage', use_column_width=False)
        
        
        
            
            
            
        