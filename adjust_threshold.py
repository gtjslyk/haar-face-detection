from cascade_classifier import *
from adaboost_classifier import *
import data_loader
import argparse
import os
from sklearn.metrics import precision_score, recall_score, accuracy_score



def _predict_weak_fvalue(integrals, means, stds, feature, threshold, polarity, scale_factor = 1.0):
        """predict of single weak classifier"""
        if scale_factor != 1:
            feat_type, x, y, w, h = feature
            x, y, w, h = int(x * scale_factor), int(y * scale_factor), int(w * scale_factor), int(h * scale_factor)
            feature = feat_type, x, y, w, h
        integrals, means, stds = np.array(integrals), np.array(means), np.array(stds)
        f_values = haar_feature_value_vectorized(integrals, means, stds, feature)
        if scale_factor != 1:
            f_values = f_values / scale_factor / scale_factor
        return f_values

def predict_value_mean(self, X, y):
        """predict of strong classifier"""
        assert X.shape[1] == X.shape[2]
        window_width = X.shape[1]
        assert self.windowX == self.windowY
        scale_factor = X.shape[1] / self.windowX
        # integrals_ = [compute_integral_image(img) for img in X.astype(np.float32)]
        # square_integrals_ = [compute_integral_image(img ** 2) for img in X.astype(np.float32)]
        # means_ = [integral[-1, -1]/(window_width ** 2) for integral in integrals_]

        
        true_indices = np.where(y == 1)


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

        f_means_list = []
        
        for classifier in self.classifiers:
            f_values = _predict_weak_fvalue(integrals, means, stds, 
                                    classifier['feature'],
                                    classifier['threshold'],
                                    classifier['polarity'],
                                    scale_factor = scale_factor)
            mean = np.mean(f_values[true_indices])
            f_means_list.append(float(mean))
            
        return f_means_list

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(  
        description='adjust threshold of each claasifier.')  
    parser.add_argument(  
        '-s', '--stage', default = 1, metavar='int', type=int,  
        help='stage')  
    
    

    args = parser.parse_args()
    stage = args.stage

    X_train, y_train = data_loader.load_dataset(os.path.join('training_data', f'validating_data_s{stage}.pkl'))
    X_val, y_val = data_loader.load_dataset(os.path.join('training_data', f'validating_data_s{stage}.pkl'))


    c = cascade_classifier()
    for s in range(stage, stage + 1):
        tmp = adaboost_classifier(classifier_name=f'stage_{s}')
        tmp.load_classifier(os.path.join('output',f'stage_{s}_classifier.pkl'))
        c.add_adaboost_calssifier(tmp)

    with open(os.path.join('output',f'stage_{s}_classifier.pkl'), 'rb') as f:
        old_file = f.read()

    fvalue_scales = []
    print("fvalue_scales: ", end = ' ')
    for _strong_clsfr in c.classifier_list:
        f_means = predict_value_mean(_strong_clsfr, X_train, y_train)
        for m, _weak_clsfr in zip(f_means, _strong_clsfr.classifiers):
            _scale = abs(m - _weak_clsfr['threshold'])
            fvalue_scales.append(_scale)
            print(f"{_scale: 6f}", end = ' ')
    print('')

    while True:
        y_pred = c.predict(X_val)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        accuracy = accuracy_score(y_val, y_pred)
        print(f'precision: {precision}, recall: {recall}, accuracy: {accuracy}')
        ipt = input()
        if ipt == 'q' or ipt == 'quit':
            exit()
        elif ipt == 'save':
            with open(os.path.join('output',f'stage_{stage}_classifier_old.pkl'), 'wb') as f:
                f.write(old_file)
            old_path = os.path.join('output',f'stage_{stage}_classifier_old.pkl')
            print(f'old file saved to {old_path}')
            with open(os.path.join('output',f'stage_{stage}_classifier.pkl'), 'wb') as f:
                pickle.dump(c.classifier_list[0].classifiers, f)
            exit()
        elif ipt == '':
            pass
        else:
            try:
                n = float(eval(ipt))
            except:
                print('invalid input')
                continue
        for clsfr, scale in zip(c.classifier_list[0].classifiers, fvalue_scales):
            clsfr['threshold'] -= clsfr['polarity'] * n * scale * clsfr['alpha']
