import os 
from google.cloud import storage

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'batch-672-gan-monet-59f43b305ef6.json'

storage_client = storage.Client()

bucket_name = 'bucket-monet-gan'


'''Accessing the bucket'''
my_bucket = storage_client.get_bucket(bucket_name)

'''Uploading files'''
def upload_to_bucket(blob_name, file_path, bucket_name):
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)
        return True

    except Exception as e:
        print(e)
        return False
    
filepath = ''
upload_to_bucket('frontend_upload_images/upload2',os.path.join(filepath,'pic.jpg'),bucket_name)

storage.bucket(bucket_name).object()
