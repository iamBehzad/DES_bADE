from imports import *

config.OTHERS = False
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
    # ============================= collecting_results =============================
    helpers.collecting_results()
