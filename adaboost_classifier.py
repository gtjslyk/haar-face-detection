#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: adaboost_classifier
Author: lyk
Date: 2025-02-12
Description: implemention of adaboost classifier and it's training and predicting function
"""
import numpy as np
from tqdm import tqdm
import pickle
import multiprocessing
import os
import time
import warnings
warnings.filterwarnings('error', category=RuntimeWarning)


def compute_integral_image(image):
    """compute the integral image"""
    integral = np.cumsum(np.cumsum(image, axis=0), axis=1)
    integral = np.pad(integral, ((1,0), (1,0)), mode='constant', constant_values=0)
    return integral

def compute_integral_image_vectorized(images):
    """compute the integral image"""
    integrals = np.cumsum(np.cumsum(images, axis=1), axis=2)
    integrals = np.pad(integrals, ((0, 0), (1,0), (1,0)), mode='constant', constant_values=0)
    return integrals

def haar_feature_value_vectorized(integrals, means, stds, feature, training = False):
    """
    compute haar feature using integral image
    for training, normalization is canceled because it's done already
    """
    def sum_rect(x1, y1, w, h):
        x2 = x1 + w
        y2 = y1 + h
        stds[stds == 0] = 1
        if training == False:
            ret = ((integrals[:, y2, x2] - integrals[:, y1, x2] - integrals[:, y2, x1] + integrals[:, y1, x1]) - (means * w * h)) / stds
        else:
            ret = (integrals[:, y2, x2] - integrals[:, y1, x2] - integrals[:, y2, x1] + integrals[:, y1, x1])
        return ret
    
    feat_type, x, y, w, h = feature
    
    if feat_type == 'two_horizontal':
        w_half = w // 2
        return sum_rect(x, y, w_half, h) - sum_rect(x + w_half, y, w_half, h)
    
    elif feat_type == 'two_vertical':
        h_half = h // 2
        return sum_rect(x, y + h_half, w, h_half) - sum_rect(x, y, w, h_half)
    
    elif feat_type == 'three_horizontal':
        w_third = w // 3
        return (sum_rect(x, y, w_third, h) 
                + sum_rect(x + 2*w_third, y, w_third, h)
                - sum_rect(x + w_third, y, w_third, h))
    
    elif feat_type == 'three_vertical':
        h_third = h // 3
        return (sum_rect(x, y, w, h_third) 
                + sum_rect(x, y + 2*h_third, w, h_third)
                - sum_rect(x, y + h_third, w, h_third))
    
    elif feat_type == 'four_rect':
        w_half = w // 2
        h_half = h // 2
        return ((sum_rect(x, y, w_half, h_half) + sum_rect(x + w_half, y + h_half, w_half, h_half))
                - (sum_rect(x + w_half, y, w_half, h_half) + sum_rect(x, y + h_half, w_half, h_half)))
    
    else:
        raise ValueError("未知特征类型")
    

def haar_feature_value(integral, mean, std, feature, training = False):
    """
    compute haar feature using integral image
    for training, normalization is canceled because it's done already
    """
    def sum_rect(x1, y1, w, h):
        x2 = x1 + w
        y2 = y1 + h
        if training == False:
            if std != 0:
                ret = ((integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]) - (mean * w * h)) / std
            else:
                ret = ((integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]) - (mean * w * h))
        else:
            ret = (integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1])
        return ret
    
    feat_type, x, y, w, h = feature
    
    if feat_type == 'two_horizontal':
        w_half = w // 2
        return sum_rect(x, y, w_half, h) - sum_rect(x + w_half, y, w_half, h)
    
    elif feat_type == 'two_vertical':
        h_half = h // 2
        return sum_rect(x, y + h_half, w, h_half) - sum_rect(x, y, w, h_half)
    
    elif feat_type == 'three_horizontal':
        w_third = w // 3
        return (sum_rect(x, y, w_third, h) 
                + sum_rect(x + 2*w_third, y, w_third, h)
                - sum_rect(x + w_third, y, w_third, h))
    
    elif feat_type == 'three_vertical':
        h_third = h // 3
        return (sum_rect(x, y, w, h_third) 
                + sum_rect(x, y + 2*h_third, w, h_third)
                - sum_rect(x, y + h_third, w, h_third))
    
    elif feat_type == 'four_rect':
        w_half = w // 2
        h_half = h // 2
        return ((sum_rect(x, y, w_half, h_half) + sum_rect(x + w_half, y + h_half, w_half, h_half))
                - (sum_rect(x + w_half, y, w_half, h_half) + sum_rect(x, y + h_half, w_half, h_half)))
    
    else:
        raise ValueError("未知特征类型")




def get_all_haar_features(window_width: int, window_height: int):
    features = []
    # 两矩形水平特征：宽度为偶数，高度任意
    for w in range(2, window_width + 1, 2):
        for h in range(1, window_height + 1):
            for x in range(window_width - w + 1):
                for y in range(window_height - h + 1):
                    features.append(('two_horizontal', x, y, w, h))
    
    # 两矩形垂直特征：高度为偶数，宽度任意
    for h in range(2, window_height + 1, 2):
        for w in range(1, window_width + 1):
            for x in range(window_width - w + 1):
                for y in range(window_height - h + 1):
                    features.append(('two_vertical', x, y, w, h))
    
    # 三矩形水平特征：宽度为3的倍数，高度任意
    for w in range(3, window_width + 1, 3):
        for h in range(1, window_height + 1):
            for x in range(window_width - w + 1):
                for y in range(window_height - h + 1):
                    features.append(('three_horizontal', x, y, w, h))
    
    # 三矩形垂直特征：高度为3的倍数，宽度任意
    for h in range(3, window_height + 1, 3):
        for w in range(1, window_width + 1):
            for x in range(window_width - w + 1):
                for y in range(window_height - h + 1):
                    features.append(('three_vertical', x, y, w, h))
    
    # 四矩形特征：宽度和高度均为偶数
    for w in range(2, window_width + 1, 2):
        for h in range(2, window_height + 1, 2):
            for x in range(window_width - w + 1):
                for y in range(window_height - h + 1):
                    features.append(('four_rect', x, y, w, h))
    
    return features

class adaboost_classifier:
    def __init__(self, classifier_name: str):
        self.windowX = 24
        self.windowY = 24
        self.N = None  # max processes
        self.T = None  # max iterations
        self.classifiers = []
        self.positive_weights_factor = None

        self.classifier_name = classifier_name
        self.output_dir = './output'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"create dir :{self.output_dir}")
        self.classifier_output_dir = 'adaboost_classifier_1'
    

    def _predict_weak(self, integrals, means, stds, feature, threshold, polarity, scale_factor = 1.0):
        """predict of single weak classifier"""
        if scale_factor != 1:
            feat_type, x, y, w, h = feature
            x, y, w, h = int(x * scale_factor), int(y * scale_factor), int(w * scale_factor), int(h * scale_factor)
            feature = feat_type, x, y, w, h
        integrals, means, stds = np.array(integrals), np.array(means), np.array(stds)
        f_values = haar_feature_value_vectorized(integrals, means, stds, feature)
        if scale_factor != 1:
            f_values = f_values / scale_factor / scale_factor
        if polarity == 1:
            return (f_values >= threshold).astype(int)
        return (f_values <= threshold).astype(int)
    

    def predict(self, X):
        """predict of strong classifier"""
        # assert X.shape[1] == X.shape[2]
        # window_width = X.shape[1]
        # assert self.windowX == self.windowY
        scale_factor = X.shape[1] / self.windowX
        # integrals_ = [compute_integral_image(img) for img in X.astype(np.float32)]
        # square_integrals_ = [compute_integral_image(img ** 2) for img in X.astype(np.float32)]
        # means_ = [integral[-1, -1]/(window_width ** 2) for integral in integrals_]

        
        integrals = compute_integral_image_vectorized(X.astype(np.float32))
        # square_integrals = compute_integral_image_vectorized((X.astype(np.float32)**2))
        # means = integrals[:, -1, -1]/(window_width ** 2)

        # # #stds = [np.sqrt((square_integral[-1, -1] / (window_width ** 2)) - (mean **2)) for square_integral, mean in zip(square_integrals, means)]

        # stds = []

        # for square_integral, mean, x in zip(square_integrals, means, X):
        #     tmp = (square_integral[-1, -1] / (window_width ** 2)) - (mean ** 2)
        #     if tmp < 0:
        #         tmp = 0
        #     std = np.sqrt(tmp)
        #     stds.append(std)

        means = np.mean(X, axis=(1, 2))
        stds = np.std(X, axis=(1, 2))

        sum_pred = np.zeros(len(X))
        
        for classifier in self.classifiers:
            pred = self._predict_weak(integrals, means, stds, 
                                    classifier['feature'],
                                    classifier['threshold'],
                                    classifier['polarity'],
                                    scale_factor = scale_factor)
            sum_pred += classifier['alpha'] * (2 * pred - 1)    # convert to ±1 vote
        # print(f'sum_pred: {sum_pred}')
        return (sum_pred >= 0).astype(int)
    
    
    

    def calculate_accuracy(self, y_test, y_pred):
        
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        
        accuracy = np.mean(y_test == y_pred)
        return accuracy

    
    def calculate_false_negative_rate(self, y_test, y_pred):
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        
        fn = np.sum((y_test == 1) & (y_pred == 0))
        tp = np.sum((y_test == 1) & (y_pred == 1))
        
        denominator = fn + tp
        if denominator == 0:
            return np.inf
        
        fn_rate = fn / denominator
        return fn_rate

    def calculate_true_negative_rate(self, y_test, y_pred):
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        
        fp = np.sum((y_test == 0) & (y_pred == 1))
        tn = np.sum((y_test == 0) & (y_pred == 0))
        
        denominator = fp + tn
        if denominator == 0:
            return np.inf
        
        tn_rate = tn / denominator
        return tn_rate



    def blocked_feature_select(self, features, weights, integrals, means, stds, y, queue = None, progress = None, error_type = 'accuracy', beta = 1):

        best_error = float('inf')
        best_feature = None
        best_threshold = 0
        best_polarity = 1
        _i = 0

        integrals, means, stds = np.array(integrals), np.array(means), np.array(stds),

        for feature in features:
            # calculate feature value of all integrals
            # f_values = np.array([haar_feature_value(integral, mean, std, feature, training=True) for integral, mean, std in zip(integrals, means, stds)])
            f_values = haar_feature_value_vectorized(integrals, means, stds, feature, training=True)
            
            # sorted all list
            sorted_indices = np.argsort(f_values)
            sorted_weights = weights[sorted_indices]
            sorted_labels = y[sorted_indices]
            sorted_values = f_values[sorted_indices]

            # calculate cumulative weights
            pos_fn = np.cumsum(sorted_weights * (sorted_labels == 1)) # 前i个中正的
            pos_tn = np.cumsum(sorted_weights * (sorted_labels == 0)) # 前i个中负的
            # for reverse the cumulation
            total_pos = pos_fn[-1]
            total_neg = pos_tn[-1]
            pos_fp = total_neg - pos_tn
            pos_tp = total_pos - pos_fn

            neg_fp = pos_tp
            neg_tn = pos_fn
            neg_fn = pos_tn
            neg_tp = pos_fp
            if error_type == 'accuracy':
                # calculatie the errors
                errors_positive = pos_fn + pos_fp # 前i个中正的 + 后i个中负的
                errors_negative = neg_fn + neg_fp # 前i个中负的 + 后i个中正的
                
                
            elif error_type == 'fbeta':

                pos_precision = pos_tp / (pos_tp + pos_fp + 1e-20)
                pos_recall = pos_tp / (pos_tp + pos_fn)
                errors_positive = 1 - ((1 + beta ** 2) * (pos_precision * pos_recall) / ((beta ** 2) * pos_precision + pos_recall + 1e-20))

                neg_precision = neg_tp / (neg_tp + neg_fp + 1e-20)
                neg_recall = neg_tp / (neg_tp + neg_fn)
                errors_negative = 1 - ((1 + beta ** 2) * (neg_precision * neg_recall) / ((beta ** 2) * neg_precision + neg_recall + 1e-20))
            else:
                ValueError('unknow error type')

            
            all_errors = np.minimum(errors_positive, errors_negative)
            min_idx = np.argmin(all_errors)
            min_error = all_errors[min_idx]
            if min_error < best_error:
                best_error = min_error
                if best_error < 0:
                    best_error = 0.0
                best_feature = feature
                best_threshold = sorted_values[min_idx]
                best_polarity = 1 if errors_positive[min_idx] < errors_negative[min_idx] else -1
            
            if progress is not None and _i % 100 == 0:
                progress.value += 100   # update progress
            _i += 1
        # return result
        if queue != None:
            queue.put((best_error, best_feature, best_threshold, best_polarity))

        return best_error, best_feature, best_threshold, best_polarity

    def load_classifier(self, path):
        with open(path, 'rb+') as f:
            self.classifiers = pickle.load(f)

    
    def train(self, X_train, y_train, X_validate, y_validate, 
              min_accuracy = 0.6, max_fn_rate = 0.01, positive_weights_factor = 10, fn_weights_factor = 0, N = 16, T = 200, cnt_num = 0, error_type = 'accuracy', beta = 1):
        self.N, self.T, self.positive_weights_factor = N, T, positive_weights_factor

        reached = False
        cnt = 0
        """calculate one strong classifier"""

        # integrals and means, stds for normalization using post-mutiplying
        integrals = [compute_integral_image(img) for img in tqdm(X_train.astype(np.float32), desc="computing integrals")]
        """for C"""
        # square_integrals = [compute_integral_image(img ** 2) for img in tqdm(X_train.astype(np.float32), desc="computing squared integrals")]
        # means = [integral[-1, -1]/(self.windowX*self.windowY) for integral in integrals]
        # stds = [np.sqrt((square_integral[24, 24] / (self.windowX*self.windowY)) - (mean **2)) for square_integral, mean in zip(square_integrals, means)]
        """python version"""
        means = np.mean(X_train, axis=(1, 2))
        stds = np.std(X_train, axis=(1, 2))


        # initialize weights
        weights_pos = self.positive_weights_factor*(y_train / np.sum(y_train)).astype(np.float64)
        y_ = np.abs(1 - y_train)    # inverse y
        weights_neg = (y_ / np.sum(y_)).astype(np.float64)
        weights = weights_pos + weights_neg
        
        # compute all features
        features = get_all_haar_features(self.windowX, self.windowX)
        for t in tqdm(range(self.T), desc= "iteration: "):
            # normalize weights
            weights /= weights.sum()

            best_info_each = []

            # divide features into N parts
            start, stop = 0, len(features)
            points = np.linspace(start, stop, self.N, dtype = np.int64)
            ranges = [(int(points[i]), int(points[i+1])) for i in range(len(points) - 1)]

            # process related
            
            processes = []
            queue = multiprocessing.Queue()
            manager = multiprocessing.Manager()
            progress = manager.Value('i', 0)  # sharing progress
            with tqdm(total=len(features), desc="features: ", leave=False) as pbar:
                for _range in ranges:
                    features_part = features[_range[0]: _range[1]]
                    process = multiprocessing.Process(target=self.blocked_feature_select, args = (features_part, weights, integrals, means, stds, y_train, queue, progress, error_type, beta))
                    processes.append(process)
                    process.start() 
                # update the bar
                while any(p.is_alive() for p in processes):
                    pbar.n = progress.value # update the value
                    pbar.last_print_n = progress.value  #
                    pbar.update(0)  # update the bar
                
                for process in processes:
                    process.join()

            # collect results
            while not queue.empty():
                best_info_each.append(queue.get())
                

            list.sort(best_info_each, key = lambda x : x[0])
            best_error, best_feature, best_threshold, best_polarity = best_info_each[0]

            # calculate alpha
            alpha = 0.5 * np.log((1 - best_error) / (best_error + 1e-200))
            
            # add weak classifier
            self.classifiers.append({
                'feature': best_feature,
                'threshold': best_threshold,
                'polarity': best_polarity,
                'alpha': alpha,
                'error': best_error,
            })
            # update weights of each X
            preds = self._predict_weak(integrals, means, stds, best_feature, best_threshold, best_polarity)
            correct = (preds == y_train)
            weights /= np.exp(alpha * correct)

            # output statistic
            y_pred = self.predict(X_train)
            acc = self.calculate_accuracy(y_train, y_pred)
            fn = self.calculate_false_negative_rate(y_train, y_pred)
            tn = self.calculate_true_negative_rate(y_train, y_pred)
            info_txt = f'{self.classifier_name}--iter: {t}, train: acc: {acc:.5f}, fnr: {fn:.5f} tnr: {tn:.5f}'
            print('')
            print(info_txt)

            with open(os.path.join(self.output_dir, 'log.txt'), 'a+') as f:
                f.write(f'{time.asctime()}: {info_txt}\n')
                f.close()


            y_pred = self.predict(X_validate)
            acc = self.calculate_accuracy(y_validate, y_pred)
            fn = self.calculate_false_negative_rate(y_validate, y_pred)
            tn = self.calculate_true_negative_rate(y_validate, y_pred)
            info_txt = f'{self.classifier_name}--iter: {t},   val: acc: {acc:.5f}, fnr: {fn:.5f} tnr: {tn:.5f}'
            print(info_txt)

            with open(os.path.join(self.output_dir, 'log.txt'), 'a+') as f:
                f.write(f'{time.asctime()}: {info_txt}\n')
                f.close()

            # adjust weights according to the false negative samples
            
            false_negative = ((preds == 0) & (y_train == 1))
            weights *= (1 + false_negative * fn_weights_factor)
            
            
            # save weights and classifiers
            
            
            # save middle info
            if not os.path.exists(os.path.join(self.output_dir, 'checkpoint')):
                os.makedirs(os.path.join(self.output_dir, 'checkpoint'))

            with open(os.path.join(self.output_dir, 'checkpoint', f'{self.classifier_name}_{t:03d}_weights.pkl'), 'wb') as f:
                pickle.dump(weights, f)
                f.close()
                
            with open(os.path.join(self.output_dir, 'checkpoint', f'{self.classifier_name}_{t:03d}_classifiers.pkl'), 'wb') as f:
                pickle.dump(self.classifiers, f)
                f.close()



            

            if acc > min_accuracy and fn < max_fn_rate:
                reached = True
            if reached :
                cnt += 1
                if cnt >= cnt_num:
                    print('metrics met, finished')
                    break

        # save final strong classifier
        output_path = os.path.join(self.output_dir, f'{self.classifier_name}_classifier.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(self.classifiers[:len(self.classifiers)], f)
            f.close()

        print(f'classifier saved: {output_path}')

        return self