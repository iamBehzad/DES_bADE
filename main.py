from imports import *

config.OTHERS = False
if __name__ == "__main__":
    if config.OTHERS:
        for datasetName in config.datasets:
            print(datasetName)
            helpers.convert_datasets(datasetName)
            if config.generate_pools:
                helpers.pool_generator(datasetName)
            if config.do_train:
                t1 = time.time()
                other_methods_helper.model_setup(datasetName)
                print("Train time",time.time()-t1)
            if config.do_evaluate:
                t1 = time.time()
                other_methods_helper.evaluate_model(datasetName)
                print("Test time", time.time() - t1)
                
    # ================================== DES_MHA ==================================
    for datasetName in config.datasets:
        if config.do_train:
            t1 = time.time()
            des_mha_helpers.model_setup(datasetName)
            print("Train time",time.time()-t1)
        if config.do_evaluate:
            t1 = time.time()
            des_mha_helpers.evaluate_model(datasetName)
            print("Test time", time.time() - t1)
            
    # ================================== Results Collection ==================================
    for d in config.datasets:
        accuracy_dict = {}
        for m in config.methods_names:
            accuracy_list = []
            path = config.ExperimentPath + "/Results/" + m + "_" + d + "_result.p"
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
        save_path='/content/drive/MyDrive/Experiment1/Results_xls/'
        writer = pd.ExcelWriter(save_path + d + "_accuracy_results.xlsx")
        df.to_excel(writer)
        writer.close()  # Use close() instead of save()
