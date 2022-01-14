"""
This files aim to detect net_mobile_vlad based on real enviroment
We test the performance based on ground truth info obtained from lidar localization pos
no local files only compare image

"""
 
import tensorflow as tf  
import faiss
import cv2
import numpy as np
from datetime import datetime
import time
import math
import os
from os.path import join, exists, isfile, realpath, dirname, basename, isdir
from os import mkdir, makedirs, removedirs, remove, chdir, environ, listdir, path, getcwd
import argparse
 
parser = argparse.ArgumentParser(description='test mobile net vlad on real environment')
parser.add_argument('--eg', type = str, help = 'example to use script')
parser.add_argument('--base', type = str, default = '/home/tang/tang/netvlad_tf_open/data/img/memory', help = 'path to the base images folder')
parser.add_argument('--test', type = str, default = '/home/tang/tang/netvlad_tf_open/data/img/query', help = 'path to the test images folder')
parser.add_argument('--weight', type = str, default = '/home/tang/tang/netvlad_tf_open/mobilenetvlad_depth-0.35', help = 'path to the weights folder')
parser.add_argument('--size', type = int, default = '4096', help = 'the size of output feature')
parser.add_argument('--thresh_trans', type = float, default = '1.0', help = 'the threshold of trans error(below this will be judged as positive match)')
parser.add_argument('--thresh_rot', type = float, default = '30.0', help = 'the threshold of rot error(below this will be judged as positive match)')
parser.add_argument('--save_result', action = 'store_true', help = 'to save results into txt')
 
def get_error_theta(theta_1, theta_2):
    relative_theta = math.fabs(theta_1 - theta_2)
    if(relative_theta > 180):
        return 360-relative_theta
    else:
        return relative_theta
 
def get_imgfiles_and_locmat(folder_path):
    if folder_path == '':
        raise Exception("no folder path given")
    print("folder path: ", folder_path)
    img_path_lists = []
    img_name_lists = []
    for image in os.listdir(folder_path):
        img_path = join(folder_path, image)
        if exists(img_path):
            img_path_lists.append(img_path)
            img_name_lists.append(image.rstrip('.jpg'))
        else:
            raise Exception("this image does not exist, which should not happen")
    return img_path_lists, img_name_lists
 
class Whole_base_test_ground_truth():
    def __init__(self, mat_base, mat_test):
        self.data_theta_base = mat_base[:, 3]
        self.data_theta_test = mat_test[:, 3]
        self.data_xy_base = mat_base[:, 1:3].astype('float32')
        self.data_xy_test = mat_test[:, 1:3].astype('float32')
        faiss_index = faiss.IndexFlatL2(2)
        faiss_index.add(np.ascontiguousarray(self.data_xy_base))
        n_values = [1, 2, 3, 4, 5]
        self.distances, self.predictions = faiss_index.search(self.data_xy_test, max(n_values))
 
 
    def get_positives(self, test_idx):
        distance = []
        index = []
        sub_distance = self.distances[test_idx, :]
        sub_corrs_index = self.predictions[test_idx, :]
        theta_1 = self.data_theta_test[test_idx]
 
        for i, dist in enumerate(sub_distance):
            if dist < 1:
                predict_index = sub_corrs_index[i]
                theta_2 = self.data_theta_base[predict_index]
                delta = get_error_theta(theta_1, theta_2)
                if delta < 30:
                    distance.append(dist)
                    index.append(predict_index)
 
        return distance, index


def list2matrix(input_list):
    output_array = np.zeros((len(input_list), 2), dtype = np.double)
    for i, list in enumerate(input_list):
        output_array[i][0] = list[0]
        output_array[i][1] = list[1]
    return output_array


def cp_img(base_name, base_name_path, query_name, query_name_path):
    folder = "/home/tang/tang/netvlad_tf_open/data/corrspond"
    folder_path = os.path.join(folder, query_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    base_img_path_new = os.path.join(folder_path, base_name + '.jpg')
    query_img_path_new = os.path.join(folder_path, query_name + '.jpg')
    cp_cmd1 = "cp " + query_name_path + " " + query_img_path_new
    cp_cmd2 = "cp " + base_name_path + " " + base_img_path_new
    os.system(cp_cmd1)
    os.system(cp_cmd2)


if __name__ == "__main__":
    params = parser.parse_args()
    base_folder = params.base
    test_folder = params.test
    weight_folder = params.weight
    feature_size = params.size
    if os.path.exists(base_folder):
        father_folder = os.path.dirname(base_folder)
        database = os.path.join(father_folder, "database.npy")
    # prerapre datas 
    base_img_path_lists, base_img_name_lists = get_imgfiles_and_locmat(base_folder)
    test_img_path_lists, test_img_name_lists = get_imgfiles_and_locmat(test_folder)
 
    base_length = len(base_img_path_lists)
    test_legnth = len(test_img_path_lists)
 
    base_features = np.empty((base_length, feature_size))
    test_features = np.empty((test_legnth, feature_size))
 
    #prepare = Whole_base_test_ground_truth(base_data_txt, test_data_txt)
 
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ['serve'], weight_folder)
        graph = tf.get_default_graph
        y = sess.graph.get_tensor_by_name('descriptor:0')
        x = sess.graph.get_tensor_by_name('image:0')
 
        # extract features from base dataset
        start_time = time.time()
        if not os.path.exists(database):
            for i, name in enumerate(base_img_path_lists):
                print("process " , i , "base frame")
                src = cv2.imread(name)
                src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                src = cv2.resize(src, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                input = np.expand_dims(src, axis = 0)
                input = np.expand_dims(input, axis = 3)
                output = sess.run(y, feed_dict={x: input})
                base_features[i, :] = output
            np.save(database, base_features)
            print("database is not existed")
        else:
            print("database is already exists")
            base_features = np.load(str(database))
        end_time = time.time()
        cost_time = (end_time - start_time) * 1000
        print("extract features from base dataset cost:{} ms ".format(cost_time))
        # extract features from test dataset 
        for i, img in enumerate (test_img_path_lists):
            print("process " , i , "test frame")
            src = cv2.imread(img)
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            src = cv2.resize(src, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            input = np.expand_dims(src, axis = 0)
            input = np.expand_dims(input, axis = 3)
            output = sess.run(y, feed_dict={x: input})
            test_features[i, :] = output
 
    base_features =  base_features.astype('float32')
    test_features =  test_features.astype('float32')
    faiss_descriptor_index = faiss.IndexFlatL2(feature_size)
    faiss_descriptor_index.add(np.ascontiguousarray(base_features))
    n_values = [1, 2, 3, 4, 5]
    start_time = time.time()
    distances, predictions = faiss_descriptor_index.search(test_features, max(n_values))
    end_time = time.time()
    cost_time = (end_time - start_time) * 1000
    print("Search candidate from base dataset cost:{} ms ".format(cost_time))
    cooresponding_txt = []
    recall_test_values = [1, 3, 5]
    correct_at_n = np.zeros(int(recall_test_values[-1]))
    correct_at_n_final = np.zeros(len(recall_test_values))
    #groundtruth is index 1 of database is according to index 1 of query
    interval_threshold = 0.3
    for row_index in range(0, test_legnth):
        query_name = test_img_name_lists[row_index]
        query_name_path = test_img_path_lists[row_index]
        predict_index = predictions[row_index, 1]
        base_name = base_img_name_lists[predict_index]
        base_name_path = base_img_path_lists[predict_index]
        if abs(int(query_name) - int(base_name)) > 120 * 1e6 and distances[row_index, 1] < interval_threshold:
            coorespond = []
            coorespond.append(query_name)
            coorespond.append(base_name)
            cooresponding_txt.append(coorespond)
            cp_img(base_name, base_name_path, query_name, query_name_path)

    output_array = list2matrix(cooresponding_txt)
    #print(output_array)
    cooresponding_txt_array = np.array(cooresponding_txt)
    coorespond_txt = join(os.getcwd(), "coorespond.txt")
    with open(coorespond_txt, 'w') as fid:
        fid.write("{}\n".format(cooresponding_txt_array))


