from imports import *

# Prepare the DS techniques. Changing k value to 7.

def initialize_ds(pool_classifiers, X_DSEL, y_DSEL, k=7):
    knorau = KNORAU(pool_classifiers, k=k)
    kne = KNORAE(pool_classifiers, k=k)
    desknn = DESKNN(pool_classifiers, k=k)
    ola = OLA(pool_classifiers, k=k)
    lca = LCA(pool_classifiers, k=k)
    mla = MLA(pool_classifiers, k=k)
    mcb = MCB(pool_classifiers, k=k)
    rank = Rank(pool_classifiers, k=k)
    knop = KNOP(pool_classifiers, k=k)
    meta = METADES(pool_classifiers, k=k)

    single_best = SingleBest(pool_classifiers, n_jobs=-1)
    oracle = Oracle(pool_classifiers)
    majority_voting = pool_classifiers

    list_ds = [knorau, kne, desknn, ola, lca, mla, mcb, rank, knop, meta, single_best, oracle]
    methods_names = ['KNORA-U', 'KNORAE', 'DESKNN', 'OLA', 'LCA', 'MLA', 'MCB', 'Rank', 'KNOP', 'META-DES', 'SingleBest', 'Oracle']

    # fit the ds techniques
    for ds in list_ds:
        if ds != majority_voting:
            ds.fit(X_DSEL, y_DSEL)

    return list_ds, methods_names

def model_setup(datasetName):
    global methods_names
    pools = helpers.load_pool(datasetName)
    ds_matrix = []
    for itr in range(config.no_itr):
        pool_classifiers = pools[itr]
        try:
            #[X_train, X_test, X_DSEL, y_train, y_test, y_DSEL] = np.load('/content/drive/MyDrive/Experiment1/Datasets/' + datasetName + str(itr) + '.npy', allow_pickle=True)
            data = np.load('./Experiment/Datasets/' + datasetName + '/' + datasetName + str(itr) + '.npz', allow_pickle=True)
            X_train = data['X_train']
            X_test = data['X_test']
            X_DSEL = data['X_DSEL']
            y_train = data['y_train']
            y_test = data['y_test']
            y_DSEL = data['y_DSEL']
            list_ds, methods_names = initialize_ds(pool_classifiers,X_DSEL,y_DSEL)
            ds_matrix.append(list_ds)
        except EOFError as e:
            print(f"Error loading file: {datasetName +  '/' + datasetName + str(itr) + '.npz'}. Skipping...")
            continue
    for tec in range(config.NO_techniques):
        ds_tec = []
        for itr in range(config.no_itr):
            ds_tec.append(ds_matrix[itr][tec])
        helpers.save_model(methods_names[tec], datasetName,ds_tec)

def evaluate_model(datasetName):
    for tec in range(config.NO_techniques):
        results = []
        labels = []
        yhat = []
        ds_tec = helpers.load_model(methods_names[tec], datasetName)
        for itr in range(config.no_itr):
            try:
                #[X_train, X_test, X_DSEL, y_train, y_test, y_DSEL] = np.load('/content/drive/MyDrive/Experiment1/Datasets/' + datasetName + str(itr) + '.npz',  allow_pickle=True)
                data = np.load('./Experiment/Datasets/' + datasetName + '/' + datasetName + str(itr) + '.npz', allow_pickle=True)
                X_train = data['X_train']
                X_test = data['X_test']
                X_DSEL = data['X_DSEL']
                y_train = data['y_train']
                y_test = data['y_test']
                y_DSEL = data['y_DSEL']
                labels.append(y_test)
                results.append(ds_tec[itr].score(X_test, y_test) * 100)
                if methods_names[tec] == 'Oracle':
                    yhat.append(ds_tec[itr].predict(X_test,y_test))
                else:
                    yhat.append(ds_tec[itr].predict(X_test))
            except EOFError as e:
                print(f"Error loading file: {datasetName + str(itr) + '.npz'}. Skipping...")
                continue
        helpers.save_results2(methods_names[tec],datasetName,results,labels,yhat)

