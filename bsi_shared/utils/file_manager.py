import xmltodict as xd
import json
import os
import cv2
import boto3
import numpy as np
import botocore


class FileManager:
    def __init__(self, s3_bucket_name=None):
        if s3_bucket_name is None :
            self.s3 = False
        else :
            self.s3 = True
            session = boto3.Session(profile_name='bsi')
            self.s3_resource = session.resource('s3')
            self.s3_client = session.client('s3')
            self.s3_bucket_name = s3_bucket_name
            self.s3_bucket = self.s3_resource.Bucket(self.s3_bucket_name)

    def get_folder_list(self, folder, extension_filter=None) :
        if self.s3 :
            plist = [obj.key[len(folder):] if obj.key.startswith(folder) else obj.key for obj in self.s3_bucket.objects.filter(Prefix=folder)]
            if isinstance(extension_filter, str):
                return sorted(filter(lambda x: x.endswith(extension_filter), plist))
            else:
                return sorted(plist)
        else :
            return sorted(os.listdir(folder))
    
    def read_image_file(self,folder,filename) :
        if self.s3 :
            obj = self.s3_client.get_object(Bucket=self.s3_bucket_name, Key=folder+filename)
            x = np.fromstring(obj['Body'].read(), dtype='uint8')
            img = cv2.imdecode(x, cv2.IMREAD_UNCHANGED)
            return img
        else :
            img = cv2.imread(os.path.join(folder, filename))
            return img


    def write_file_to_s3(self, filename, localpath='', s3path=''):
        if not self.s3:
            return
        local_filepath = os.path.join(localpath, filename)
        remote_filepath = os.path.join(s3path, filename)
        with open(local_filepath, 'rb') as f:
            self.s3_bucket.put_object(Key=remote_filepath, Body=f)


    def read_image_dict(self,folder,filename) :

        x_imageDict = None
        if self.s3 :
            obj = self.s3_client.get_object(Bucket=self.s3_bucket_name, Key=folder+filename)
            if obj is not None :
                if filename.endswith('.xml'):
                    x_imageDict = xd.parse(obj['Body'])
                elif filename.endswith('.json'):
                    x_imageDict = json.load(obj['Body'])
        else:
            with open(folder + filename, 'r+') as f:
                if filename.endswith('.xml'):
                    x_imageDict = xd.parse(f)
                elif filename.endswith('.json'):
                    x_imageDict = json.load(f)
                f.close()
        return x_imageDict

    def save_image_dict(self,folder,filename,data) :

        if self.s3 :
            self.s3_client.put_object(Body=json.dumps(data), Bucket=self.s3_bucket_name, Key=folder+filename)

        else:
            with open(os.path.join(folder, filename), 'w+') as f:
                json.dump(data, f)
                f.close()

    def download_file(self, folder, key, local_path):
        if not self.s3:
            print('Requires s3 connection')
            return
        try:
            self.s3_client.download_file(self.s3_bucket_name, folder+key, local_path)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise

    def download_dir(self, dir_name, local='.', force_redownload=False, force_redownload_jsons=False):
        new_downloaded_files = 0
        already_downloaded_files = 0
        redownloaded_files = 0
        paginator = self.s3_client.get_paginator('list_objects')
        for result in paginator.paginate(Bucket=self.s3_bucket_name, Delimiter='/', Prefix=dir_name):
            if result.get('CommonPrefixes') is not None:
                for subdir in result.get('CommonPrefixes'):
                    self.download_dir(subdir.get('Prefix'), local, force_redownload, force_redownload_jsons)
            if result.get('Contents') is not None:
                for file in result.get('Contents'):
                    fname = local + os.sep + file.get('Key')
                    if os.path.exists(fname):
                        if fname.endswith('json'):
                            if not force_redownload_jsons and not force_redownload:
                                already_downloaded_files += 1
                                continue
                            else:
                                redownloaded_files += 1
                        elif not force_redownload:
                            already_downloaded_files += 1
                            continue
                        else: 
                            redownloaded_files += 1

                    if not os.path.exists(os.path.dirname(local + os.sep + file.get('Key'))):
                        print('Found new dir: {}'.format(file.get('Key')))
                        if os.path.isdir(file.get('Key')):
                            os.makedirs(local + os.sep + file.get('Key'))
                        else:
                            os.makedirs(os.path.dirname(local + os.sep + file.get('Key')))
                    new_downloaded_files += 1
                    if not os.path.isdir(file.get('Key')):
                        self.s3_resource.meta.client.download_file(self.s3_bucket_name, file.get('Key'), local + os.sep + file.get('Key'))
        print("Downloaded {}/{} files from {}".format(new_downloaded_files, already_downloaded_files+new_downloaded_files, dir_name))
        print("Redownloaded {} files".format(redownloaded_files))

    def compare_local_remote_dirs(self, local_dir, remote_dir):
        if not self.s3:
            return False
        
        local_dir_list = os.listdir(local_dir)
        remote_dir_list = self.get_folder_list("")

if __name__ == '__main__':

    # fm = FileManager("sagemaker-us-east-1-359098845427")
    # print fm.get_folder_list('')
    # fm.get_checkpoint_from_arn("sagemaker-us-east-1-359098845427")
    # fm = FileManager("bsivisiondata")
    # fm.download_dir('', './data')
    pass