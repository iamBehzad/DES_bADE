from imports import *

def save_pool(datasetName,pools):
    path = config.ExperimentPath + "Pools/" + datasetName + '/' + datasetName + "_pools.p"
    poolspec = open(path, mode="wb")
    pickle.dump(pools, poolspec)
    poolspec.close() 

def load_pool(datasetName):
    path = config.ExperimentPath + "Pools/" + datasetName +  '/' + datasetName + "_pools.p"
    poolspec = open(path, mode="rb")
    return pickle.load(poolspec)

def save_model(tec_name,datasetName,ds):
    path = config.ExperimentPath + "Models/" + datasetName +  '/' + tec_name +"_"+datasetName + "_model.p"
    poolspec = open(path, mode="wb")
    pickle.dump(ds, poolspec)
    poolspec.flush()
    poolspec.close()

def load_model(tec_name,datasetName):
    path = config.ExperimentPath + "Models/" + datasetName +  '/' + tec_name +"_"+ datasetName + "_model.p"
    poolspec = open(path, mode="rb")
    return pickle.load(poolspec)

def save_results(tec_name,datasetName,accuracy,labels,yhat):
    path = config.ExperimentPath + "Results/" + datasetName + '/' +tec_name +"_"+ datasetName + "_result.p"
    poolspec = open(path, mode="wb")
    pickle.dump(accuracy, poolspec)
    pickle.dump(labels, poolspec)
    pickle.dump(yhat, poolspec)
    poolspec.flush()
    poolspec.close()

def save_results2(tec_name, datasetName, accuracy, labels, yhat):
    # Save .p file (pickle)
    p_path = os.path.join(config.ExperimentPath, "Results", datasetName, f"{tec_name}_{datasetName}_result.p")
    with open(p_path, mode="wb") as poolspec:
        pickle.dump(accuracy, poolspec)
        pickle.dump(labels, poolspec)
        pickle.dump(yhat, poolspec)
        #print(len(accuracy), len(labels), len(yhat))

    # Save .xlsx file
    df = pd.DataFrame({
        'Accuracy': accuracy,
        'Labels': labels,
        'Predictions': yhat
    })
    excel_path = os.path.join(config.ExperimentPath, "Results_xls", datasetName, f"{tec_name}_{datasetName}_results.xlsx")
    df.to_excel(excel_path, index=False, engine='openpyxl')

def pool_generator(datasetName):
    state = 0
    pools = []
    for itr in range(0, config.no_itr):
        rng = np.random.RandomState(state)
        #[X_train, X_test, X_DSEL, y_train, y_test, y_DSEL] =  np.load('/content/drive/MyDrive/Experiment1/Datasets/' + datasetName +str(itr)+'.npz',allow_pickle=True)
        # Load the data
        try:              
            data = np.load(config.ExperimentPath + 'Datasets/' + datasetName + '/' + datasetName + str(itr) + '.npz', allow_pickle=True)
            X_train = data['X_train']
            X_test = data['X_test']
            X_DSEL = data['X_DSEL']
            y_train = data['y_train']
            y_test = data['y_test']
            y_DSEL = data['y_DSEL']

            learner = Perceptron(max_iter=100, tol=10e-3, alpha=0.001, penalty=None, random_state=rng)
            model = CalibratedClassifierCV(learner, cv=5,method='isotonic')
            pool_classifiers = BaggingClassifier(model, n_estimators=config.NO_classifiers, bootstrap=True, max_samples=1.0, random_state=rng)

            pool_classifiers.fit(X_train,y_train)
            pools.append(pool_classifiers)
        except EOFError as e:
            print(f"Error loading file: {datasetName +  '/' + datasetName + str(itr) + '.npz'}. Skipping...")
            continue
    path = config.ExperimentPath + "/Pools/" + datasetName +  '/' + datasetName + "_pools.p"
    poolspec = open(path, mode="wb")
    pickle.dump(pools, poolspec)

def convert_datasets(datasetName):
    redata = sio.loadmat("./datasets/" + datasetName + ".mat")
    data = redata['dataset']
    X = data[:, 0:-1]
    y = data[:, -1]
    print(datasetName, "is readed.")
    state = 0
    print(datasetName, ': ', X.shape)

    # ### ### ### ### ### ### ### ### ###
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    X[np.isnan(X)] = 0
    # #### **** #### **** #### **** #### **** #### **** #### ****
    # scaler = preprocessing.MinMaxScaler()
    # X = scaler.fit_transform(X)
    # #### **** #### **** #### **** #### **** #### **** #### ****

    yhat = np.zeros((config.no_itr, math.ceil(len(y) / 4)))
    for itr in range(0, config.no_itr):
        # rand = np.random.randint(1,10000,1)
        rng = np.random.RandomState(itr)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y,
                                                            random_state=rng)  # stratify=y
        X_DSEL, X_test, y_DSEL, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test,
                                                          random_state=rng)  # stratify=y_test
        yhat[itr, :] = y_test

        scaler = preprocessing.MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_DSEL = scaler.transform(X_DSEL)

        # Save the data
        np.savez( config.ExperimentPath + 'Datasets/' + datasetName + '/' + datasetName + str(itr) + '.npz',
                X_train=X_train, X_test=X_test, X_DSEL=X_DSEL,
                y_train=y_train, y_test=y_test, y_DSEL=y_DSEL)


def collecting_results():
    methods_names = ['DES_MHA', 'KNORA-U', 'KNORAE', 'DESKNN', 'OLA', 'LCA', 'MLA', 'MCB', 'Rank', 'KNOP', 'META-DES', 'SingleBest', 'Oracle']

    for d in config.datasets:
        accuracy_dict = {}
        for m in methods_names:
            accuracy_list = []
            path = config.ExperimentPath + "/Results/" + d + "/" + m + "_" + d + "_result.p"
            with open(path, mode="rb") as f:
                accuracy = pickle.load(f)
            accuracy_list.append(accuracy)
            accuracy_dict[m] = np.array(accuracy_list).reshape(-1,)
        df = pd.DataFrame(accuracy_dict)
        # Calculate mean, std, and rank of mean for each column
        mean = df.mean()
        std = df.std()
        rank = mean.rank(ascending=False)
        df.loc['mean'] = mean
        df.loc['std'] = std
        df.loc['rank'] = rank
        save_path = config.ExperimentPath + '/Results_xls/' + d + '/'
        writer = pd.ExcelWriter(save_path + d + "_accuracy_results.xlsx")
        df.to_excel(writer)
        writer.close()  # Use close() instead of save()