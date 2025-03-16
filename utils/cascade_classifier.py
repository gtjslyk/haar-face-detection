from utils.adaboost_classifier import *
import utils.data_loader as data_loader


class cascade_classifier:
    def __init__(self, classifier_name='annonymous_cc'):
        self.calssifier_name = classifier_name
        self.classifier_list = []
    
    def load_adaboost_classifier(self, classifier_list: list):
        self.classifier_list = classifier_list

    def add_adaboost_calssifier(self, classifier):
        self.classifier_list.append(classifier)
    
    def get_stages_num(self):
        return len(self.classifier_list)
    
    def predict(self, X):
        y_ret = np.zeros(X.shape[0])
        next_X = X
        indices_in_origin = np.arange(0, X.shape[0])
        for classifier in self.classifier_list:
            
            y_pred = classifier.predict(next_X)
            next_X = next_X[y_pred == 1]
            indices_in_origin = indices_in_origin[y_pred == 1]
            if not next_X.any():
                return y_ret
            
        y_ret[indices_in_origin] = 1
        return y_ret

