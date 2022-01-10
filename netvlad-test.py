"""
This files aim to detect net_mobile_vlad based on real enviroment
We test the performance based on ground truth info obtained from lidar localization pos
"""
 
import tensorflow as tf  
import faiss
import cv2
import numpy as np
from datetime import datetime
import time
import math
from os.path import join, exists, isfile, realpath, dirname, basename, isdir
from os import mkdir, makedirs, removedirs, remove, chdir, environ, listdir, path, getcwd
import argparse
 
parser = argparse.ArgumentParser(description='test mobile net vlad on real environment')
parser.add_argument('--eg', type = str, help = 'example to use script')
parser.add_argument('--base', type = str, default = '/home/tang/tang/netvlad_tf_open/test_images/memory', help = 'path to the base images folder')
parser.add_argument('--test', type = str, default = '/home/tang/tang/netvlad_tf_open/test_images/live', help = 'path to the test images folder')
parser.add_argument('--weight', type = str, default = '/home/tang/tang/netvlad_tf_open/mobilenet_v2_0.35_224/mobilenet_v2_0.35_224_frozen.pb', help = 'path to the weights folder')
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
    loc_txt = join(folder_path, "loc.txt")
    img_path_lists = []
    if exists(loc_txt):
        data_txt = np.loadtxt(loc_txt)
        length = data_txt.shape[0]
        for i in range(length):
            timestamp = str((int)(data_txt[i, :][0]))
            img_path = join(folder_path, timestamp) + '.png'
            if exists(img_path):
                img_path_lists.append(img_path)
            else:
                raise Exception("this image does not exist, which should not happen")
 
    return img_path_lists, data_txt
 
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
 
if __name__ == "__main__":
    params = parser.parse_args()
    base_folder = params.base
    test_folder = params.test
    weight_folder = params.weight
    feature_size = params.size
    # prerapre datas 
    base_img_path_lists, base_data_txt = get_imgfiles_and_locmat(base_folder)
    test_img_path_lists, test_data_txt = get_imgfiles_and_locmat(test_folder)
 
    base_length = len(base_img_path_lists)
    test_legnth = len(test_img_path_lists)
 
    base_features = np.empty((base_length, feature_size))
    test_features = np.empty((test_legnth, feature_size))
 
    prepare = Whole_base_test_ground_truth(base_data_txt, test_data_txt)
 
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ['serve'], weight_folder)
        graph = tf.get_default_graph
        y = sess.graph.get_tensor_by_name('descriptor:0')
        x = sess.graph.get_tensor_by_name('image:0')
 
        # extract features from base dataset 
        for i, name in enumerate(base_img_path_lists):
            print("process " , i , "base frame")
            src = cv2.imread(name)
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            src = cv2.resize(src, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            input = np.expand_dims(src, axis = 0)
            input = np.expand_dims(input, axis = 3)
            output = sess.run(y, feed_dict={x: input})
            base_features[i, :] = output
 
        # extract features from test dataset 
        for i, name in enumerate (test_img_path_lists):
            print("process " , i , "test frame")
            src = cv2.imread(name)
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
    distances, predictions = faiss_descriptor_index.search(test_features, max(n_values))
 
    recall_test_values = [1, 5, 10, 20]
    correct_at_n = np.zeros(len(recall_test_values))
    correct_score = np.zeros((3, test_legnth), dtype = np.float)
 
    for qIx, predict in enumerate(predictions):
        distance_truth, predict_truth = prepare.get_positives(qIx)
        print("==================================================")
        print("truth ", predict_truth)
        print("predict ", predict)
        for i, n in enumerate(recall_test_values):
            if np.any(np.in1d(predict[:n], predict_truth)):
                correct_at_n[i:] += 1
                break
 
    # prapare data save into txt
    for qIx, predict in enumerate(predictions):
        distance_truth, predict_truth = prepare.get_positives(qIx)
        if np.any(np.in1d(predict[0], predict_truth)):
            correct_score[0][qIx]= 1
        else:
            correct_score[0][qIx]= 0
        correct_score[2][qIx] = predict[0]  
 
    for qIx, distance in enumerate(distances):
        correct_score[1][qIx] = distance[0]
 
 
    recall_at_n = correct_at_n / test_legnth
    for i, n in enumerate(recall_test_values):
        print("====> Recall@{}: ({}/{}) {:.4f}".format(n, correct_at_n[i], test_legnth , recall_at_n[i]))
 
    # save into txt 
    PRC_Matches_writePath = datetime.now().strftime('%b%d_%H-%M-%S_') + 'Mobilevlad_Presicion_recall_Data.txt'
    np.savetxt(PRC_Matches_writePath, correct_score, fmt='%f', delimiter=',')
