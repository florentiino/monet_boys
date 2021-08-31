import os
from google.cloud import storage
from google.cloud.storage import bucket
from dotenv import load_dotenv

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../raw_data/batch-672-gan-monet.json'

storage_client = storage.Client()

bucket_name = 'bucket-monet-gan'
bucket = storage_client.bucket(bucket_name)
file_path = r'../raw_data/test/'

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


'''Downloading files'''


def download_from_bucket(blob_name, file_path, bucket_name):
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    with open(file_path, 'wb') as f:
        storage_client.download_blob_to_file(blob, f)
    print(file_path)
    return True


def main():
    download_from_bucket('weights/cezanne_weights.h5', os.path.join(
        file_path, 'download_cezanne_weights.h5'), bucket_name)


if __name__ == '__main__':
    main()

    #print(dir(storage_client)) #print bucket details
    #print(vars(my_bucket)) # print bucket details
    #upload_to_bucket('test/test_monet', os.path.join(file_path,
    #                  'test_monet.jpg'), 'bucket-monet-gan')
    #print('upload donesies')
    #download_from_bucket('test_monet', os.path.join(os.getcwd(),'download1.jpg'), bucket_name)
    #print('download donesise')
