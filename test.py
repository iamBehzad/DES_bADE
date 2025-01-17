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
    # Set the directory where the file will be created  
    directory = os.path.join(config.ExperimentPath, 'Results_xls', d)  
    os.makedirs(directory, exist_ok=True)  

    # Set the base filename  
    base_filename = f"{d}_accuracy_results.xlsx"  
    base_filename_with_path = os.path.join(directory, base_filename)  

    # Check if the file already exists  
    i = 1  
    new_filename = base_filename_with_path  
    while os.path.exists(new_filename):  
        # Generate a unique filename by appending a number  
        new_filename = os.path.join(directory, f"{os.path.splitext(base_filename)[0]}_{i}{os.path.splitext(base_filename)[1]}")  
        i += 1  

    # Save the DataFrame to the new file  
    with pd.ExcelWriter(new_filename) as writer:  
        df.to_excel(writer, index=True)  