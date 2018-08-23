from utils.data_extractor import DataExtractor
import os
import csv
import random

def write_data_to_CSV(folderNames, s3_bucket_name, path_root="", train_test_ratio=.7):
    gens = []
    for f in folderNames:
        gens.append(DataExtractor(f, s3_bucket=s3_bucket_name).get_annotation_generator(True))

    csv_data = [] #[[filename,x1,y1,x2,y2,class_label]...]
    val_csv_data = []
    for i,gen in enumerate(gens):
        for data in gen:
            im_dict = data['data']
            fname = data['filename']
            
            #check if bbox objects in im_dict
            objects = im_dict.get('annotation', {}).get('object', {})
            im_name = im_dict['annotation']['filename']
            path = os.path.join( path_root, folderNames[i] ,'frames', im_name)
            if objects:
                # if so:
                # csv_data.append(line)
                for obj in objects:
                    line = [path, []]
                    line[1].append(obj.get('bndbox', {}).get('xmin', '-1')) 
                    line[1].append(obj.get('bndbox', {}).get('ymin', '-1'))
                    line[1].append(obj.get('bndbox', {}).get('xmax', '-1'))
                    line[1].append(obj.get('bndbox', {}).get('ymax', '-1'))
                    line[1].append(obj.get('name', 'Unnamed'))
                    if '-1' not in line[1] and 'Unnamed' not in line[1]:
                        if random.random() > train_test_ratio:
                            val_csv_data.append(line)
                        else:
                            csv_data.append(line)       
            # if not, append filename line


    def write_csv_data_to_file(fname, data):
        with open(fname, 'wb') as f:
            wr = csv.writer(f)
            for filename, M in data:
                row = [x for x in M]
                row = [filename] + row
                wr.writerow(row)

    write_csv_data_to_file("data/annotations.csv", csv_data)
    write_csv_data_to_file("data/val_annotations.csv", val_csv_data)
    print("Done.")



