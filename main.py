from imports import *

config.OTHERS = True
if __name__ == "__main__":
    # =============================== Other Methods ===============================
    if config.OTHERS:
        for datasetName in config.datasets:
            print(datasetName)
            helpers.convert_datasets(datasetName)
            if config.generate_pools:
                helpers.pool_generator(datasetName)
            if config.do_train:
                t1 = time.time()
                helpers.others_model_setup(datasetName)
                print("Train time",time.time()-t1)
            if config.do_evaluate:
                t1 = time.time()
                helpers.others_evaluate_model(datasetName)
                print("Test time", time.time() - t1)
    # ================================== DES_MHA ==================================
    for datasetName in config.datasets:
        if config.do_train:
            t1 = time.time()
            helpers.des_mha_model_setup(datasetName, k=20)
            print("Train time",time.time()-t1)
        if config.do_evaluate:
            t1 = time.time()
            helpers.des_mha_evaluate_model(datasetName)
            print("Test time", time.time() - t1)
    # ============================= collecting_results =============================
    helpers.collecting_results()
