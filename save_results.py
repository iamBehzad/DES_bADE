from imports import *
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