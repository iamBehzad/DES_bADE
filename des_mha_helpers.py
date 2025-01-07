from imports import *

def initialize_ds(pool_classifiers, X_DSEL, y_DSEL, k=20):
    des_mha = DES_MHA(pool_classifiers, k=k)
    # fit the ds techniques
    des_mha.fit(X_DSEL, y_DSEL)
    return des_mha

def model_setup(datasetName):
    pools = helpers.load_pool(datasetName)
    des_matrix = []
    for itr in range(config.no_itr):
        t1 = time.time()
        pool_classifiers = pools[itr]
        try:
            print(f"Setup Number {itr} just started")
            data = np.load('./Experiment/Datasets/' + datasetName +  '/' + datasetName + str(itr) + '.npz', allow_pickle=True)
            X_train = data['X_train']
            X_test = data['X_test']
            X_DSEL = data['X_DSEL']
            y_train = data['y_train']
            y_test = data['y_test']
            y_DSEL = data['y_DSEL']
            ds=initialize_ds(pool_classifiers,X_DSEL,y_DSEL)
            des_matrix.append(ds)
            print(f"Setup Number {itr} - Duration = {time.time()-t1}")
        except EOFError as e:
            print(f"Error loading file: {datasetName +  '/' + datasetName + str(itr) + '.npz'}. Skipping...")
            continue
    ds_mha = []
    for itr in range(config.no_itr):
        ds_mha.append(des_matrix[itr])
    helpers.save_model("DES_MHA", datasetName,ds_mha)

def evaluate_model(datasetName):
    results = []
    labels = []
    yhat = []
    ds_mha = helpers.load_model("DES_MHA", datasetName)
    for itr in range(config.no_itr):
        t1 = time.time()
        try:
            print(f"Evaluate Number {itr} just started")
            data = np.load('./Experiment/Datasets/' + datasetName +  '/' + datasetName + str(itr) + '.npz', allow_pickle=True)
            X_train = data['X_train']
            X_test = data['X_test']
            X_DSEL = data['X_DSEL']
            y_train = data['y_train']
            y_test = data['y_test']
            y_DSEL = data['y_DSEL']
            labels.append(y_test)
            score, y_hat = ds_mha[itr].score(X_test, y_test)
            results.append(score * 100)
            yhat.append(y_hat)
            print(f"Evaluation Number {itr} - Duration = {time.time()-t1}")
        except EOFError as e:
            print(f"Error loading file: {datasetName +  '/' + datasetName + str(itr) + '.npz'}. Skipping...")
            continue
    helpers.save_results2("DES_MHA", datasetName,results,labels,yhat)