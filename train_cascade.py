
import neg_generator
import cv2
import multiprocessing
from cascade_classifier import *
import argparse



def blocked_get_false_true_samples(ret_list, total_neg_num, stage, c, img_list):
    for i in range(stage):
        X_neg, y_neg = neg_generator.generate_negative(img_list, num = 1000)
        if stage == 1:
            X_neg_next = X_neg
            ret_list.append(X_neg_next)
        else:
            y_pred = c.predict(X_neg)
            X_fp = X_neg[y_pred == 1]
            ret_list.append(X_fp)
            total_neg_num.value += 1000
    return

stage = 1
neg_N = 10000
pos_N = 10000
window_N = 24
train_val_ratio = 0.8


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(  
        description='sum the integers at the command line')  
    parser.add_argument(  
        '-s', '--stage', default = 1, metavar='int', type=int,  
        help='stage to train')  
    parser.add_argument(
        '-v', '--view', default=False, type=bool,  
        help='a short view before generating dataset')  
    parser.add_argument('--min_accuracy', default=0.7, type=float)
    parser.add_argument('--max_fn_rate', default=0.005, type=float)
    parser.add_argument('--positive_weights_factor', default=5, type=float)
    parser.add_argument('--fn_weights_factor', default=2, type=float)
    parser.add_argument('--N', default=12, type=int)
    parser.add_argument('--T', default=1, type=int)
    parser.add_argument('--posN', default=4000, type=int)
    parser.add_argument('--negN', default=6000, type=int)
    parser.add_argument('--train_val_ratio', default=0.8, type=float)
    parser.add_argument('--cnt', default=0, type=int)
    parser.add_argument('--data_from_file', default=False, type=bool)
    parser.add_argument('--error_type', default='accuracy', type=str)
    parser.add_argument('--beta', default=1, type=float)
    parser.add_argument('--only_genfile', default=None, type=int)
    parser.add_argument('--from_genedfile', default=None, type=bool)
    

    args = parser.parse_args()
    stage = args.stage
    view = args.view
    min_accuracy = args.min_accuracy
    max_fn_rate = args.max_fn_rate
    positive_weights_factor = args.positive_weights_factor
    fn_weights_factor = args.fn_weights_factor
    N = args.N
    T = args.T
    pos_N = args.posN
    neg_N = args.negN
    train_val_ratio = args.train_val_ratio
    error_type = args.error_type
    beta = args.beta
    only_genfile = args.only_genfile

    # positive data 
    X_pos, y_pos = data_loader.load_dataset(os.path.join('data', 'pos_data_normalized.pkl'))
    X_pos, y_pos = X_pos[:pos_N], y_pos[:pos_N]
    pivot = int(train_val_ratio * y_pos.shape[0])
    X_train_pos, y_train_pos = X_pos[:pivot], y_pos[:pivot]
    X_val_pos, y_val_pos = X_pos[pivot:], y_pos[pivot:]

    c = cascade_classifier()
    X_neg_next = np.ndarray((0, window_N, window_N))
    for s in range(1, stage):
        tmp = adaboost_classifier(classifier_name=f'stage_{s}')
        tmp.load_classifier(os.path.join('output',f'stage_{s}_classifier.pkl'))
        c.add_adaboost_calssifier(tmp)
    total_neg_num = 0

    
    
    manager = multiprocessing.Manager()
    ret_list = manager.list()
    total_neg_num = multiprocessing.Value('i', 0)  # sharing progress

    
    with open('data/negative_list.pkl', 'rb') as f:
        img_list = pickle.load(f)
        f.close()

    if args.from_genedfile:
        root, ds, fs = os.walk('file_gen').__next__()
        for f in tqdm(fs, desc="loading generated files: "):
            p = os.path.join(root, f)
            with open(p, 'rb') as f:
                X, y = pickle.load(f)

                y_pred = c.predict(X)
                X_fp = X[y_pred == 1]

                X_neg_next = np.concatenate((X_neg_next, X_fp), axis = 0)

    if args.data_from_file == False:
        while True:
            processes = []
            for i in range(1):
                process = multiprocessing.Process(target=blocked_get_false_true_samples, args = (ret_list, total_neg_num, stage, c, img_list))
                processes.append(process)
                process.start() 
            while any(p.is_alive() for p in processes):
                i+=1
                time.sleep(0.01)
            
            for process in processes:
                process.join()
            for l in ret_list:
                X_neg_next = np.concatenate((X_neg_next, l), axis=0)
            ret_list = manager.list()
            
            print(f'\rfalse true samples num: {X_neg_next.shape[0]:-5d} / {total_neg_num.value:-10d}', end = '')
            if X_neg_next.shape[0] > neg_N:
                break
        print('')
        # while True:
        #     X_neg, y_neg = neg_generator.generate_negative(num = 40000)
        #     total_neg_num += 40000
        #     if stage == 1:
        #         X_neg_next = X_neg
        #         break
        #     else:
        #         y_pred = c.predict(X_neg)
        #         X_fp = X_neg[y_pred == 1]
        #         X_neg_next = np.concatenate((X_neg_next, X_fp), axis=0)
        #         if X_neg_next.shape[0] > neg_N:
        #             break
        #     print(f'\rfalse true samples num: {X_neg_next.shape[0]:05d}', end='')
        # print(f'\ntotal negative samples tested: {total_neg_num}')
        X_neg_next = X_neg_next[:neg_N]
        y_neg_next = np.zeros(X_neg_next.shape[0])
        pivot = int(train_val_ratio * y_neg_next.shape[0])
        X_train_neg, y_train_neg = X_neg_next[:pivot], y_neg_next[:pivot]
        X_val_neg, y_val_neg = X_neg_next[pivot:], y_neg_next[pivot:]


        X_train, y_train = data_loader.merge_dataset(X_train_pos, y_train_pos, X_train_neg, y_train_neg)
        X_val, y_val = data_loader.merge_dataset(X_val_pos, y_val_pos, X_val_neg, y_val_neg)
        if only_genfile is not None:
            assert type(only_genfile) == int
            if not os.path.exists('./file_gen'):
                os.mkdir('./file_gen')
            data_loader.dump_dataset(os.path.join('./file_gen', f'training_data_s{stage}_{only_genfile}.pkl'), X_train, y_train)
            data_loader.dump_dataset(os.path.join('./file_gen', f'validating_data_s{stage}_{only_genfile}.pkl'), X_val, y_val)
            exit()
    else:
        X_train, y_train = data_loader.load_dataset(os.path.join('training_data', f'training_data_s{stage}.pkl'))
        X_val, y_val = data_loader.load_dataset(os.path.join('training_data', f'validating_data_s{stage}.pkl'))


    if view is True:
        for c in X_train[::1000]:
            cv2.namedWindow('training data view', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('training data view', 500, 500)
            cv2.imshow('training data view', c)
            k = cv2.waitKey(0)
            if k == ord('q'):
                break
            elif k == ord('s'):
                print('stop generation. exit')
                exit()
        cv2.destroyAllWindows()
        
        for c in X_val[::1000]:
            cv2.namedWindow('validating data view', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('validating data view', 500, 500)
            cv2.imshow('validating data view', c)
            k = cv2.waitKey(0)
            if k == ord('q'):
                break
            elif k == ord('s'):
                print('stop generation. exit')
                exit()
        cv2.destroyAllWindows()

    data_loader.dump_dataset(os.path.join('training_data', f'training_data_s{stage}.pkl'), X_train, y_train)
    data_loader.dump_dataset(os.path.join('training_data', f'validating_data_s{stage}.pkl'), X_val, y_val)


    classifier = adaboost_classifier(f'stage_{stage}')

    """final confirmation"""
    train_pos_num = np.sum(y_train.astype(np.uint8))
    val_pos_num = np.sum(y_val.astype(np.uint8))
    print(f'pos/neg: train: {train_pos_num}/{y_train.shape[0]} val: {val_pos_num}/{y_val.shape[0]}')
    print(f'min accuracy: {min_accuracy}')
    print(f'max fn rate: {max_fn_rate}')
    print(f'positive weights factor: {positive_weights_factor}')
    print(f'process num: {N}')
    print(f'max iteration: {T}')
    # if input('press y to start: ') != 'y':
    #     exit()
    classifier.train(X_train, y_train, X_val, y_val, 
                     min_accuracy=min_accuracy, max_fn_rate=max_fn_rate, 
                     positive_weights_factor=positive_weights_factor, 
                     fn_weights_factor=fn_weights_factor,
                     N=N, T=T,
                     cnt_num=args.cnt,
                     error_type = error_type,
                     beta = beta)