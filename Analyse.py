from imports import * 

def load_accuracy_data(datasets, methods, base_path):
    """
    Loads accuracy data from pickle files and organizes them into a dictionary.
    """
    all_data = {}
    for dataset in datasets:
        accuracy_dict = {}
        for method in methods:
            path = f"{base_path}/Results/{dataset}/{method}_{dataset}_result.p"
            if os.path.exists(path):
                with open(path, "rb") as f:
                    accuracy = pickle.load(f)
                accuracy_dict[method] = np.array(accuracy).reshape(-1,)
            else:
                print(f"File not found: {path}")
        all_data[dataset] = pd.DataFrame(accuracy_dict)
    return all_data

def calculate_win_tie_loss(data, target_method, methods):
    """
    Creates a win-tie-loss summary for the target method against other methods.
    """
    summary = {"Dataset": [], "Method": [], "Wins": [], "Ties": [], "Losses": []}
    for dataset, values in data.items():
        target_results = np.mean(values[target_method])
        for method in methods:
            if method != target_method and method in values:
                method_results = np.mean(values[method])
                wins = np.sum(target_results > method_results)
                ties = np.sum(target_results == method_results)
                losses = np.sum(target_results < method_results)
                summary["Dataset"].append(dataset)
                summary["Method"].append(method)
                summary["Wins"].append(wins)
                summary["Ties"].append(ties)
                summary["Losses"].append(losses)
    return pd.DataFrame(summary)

def save_win_tie_loss_plot(summary, dataset_name, save_dir="./Results"):
    """
    Saves the win-tie-loss summary plot as a high-quality image.
    """
    os.makedirs(save_dir, exist_ok=True)
    grouped = summary.groupby("Method")[["Wins", "Ties", "Losses"]].sum()
    ax = grouped.plot(kind="bar", stacked=True, figsize=(10, 6))
    plt.title(f"Win-Tie-Loss")
    plt.xlabel("Methods")
    plt.ylabel("Count")
    #plt.grid(axis="y")
    plt.xticks(rotation=0)
    
    
    # Display values on their respective sections of the bars  
    for i in range(len(grouped)):  
        # The cumulative height to position the text appropriately  
        win_height = grouped.iloc[i]["Wins"]  
        tie_height = grouped.iloc[i]["Ties"]  
        loss_height = grouped.iloc[i]["Losses"]  

        # Annotate Wins (blue part)  
        ax.annotate(f'{int(win_height)}',  
                     (i, win_height / 2),  # Position it on the blue part  
                     ha='center', va='center', fontsize=10, color='white')  

        # Annotate Ties (green part)  
        if int(tie_height) > 0 :
            ax.annotate(f'{int(tie_height)}',  
                        (i, win_height + tie_height / 2),  # Position it on the green part  
                        ha='center', va='center', fontsize=10, color='white')  

        # Annotate Losses (red part)  
        ax.annotate(f'{int(loss_height)}',  
                     (i, win_height + tie_height + loss_height / 2),  # Position it on the red part  
                     ha='center', va='center', fontsize=10, color='white')  
        
    save_path = os.path.join(save_dir, f"{dataset_name}_WinTieLoss.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return grouped

def friedman_test(accuracy_dict):
    """Perform Friedman test and return results with ranks."""
    methods = list(next(iter(accuracy_dict.values())).keys())
    data = []
    for dataset, accuracies in accuracy_dict.items():
        data.append([accuracies[method].mean() for method in methods])
    data = np.array(data).T
    statistic, p_value = friedmanchisquare(*data)

    # Calculate ranks
    ranks = np.argsort(np.argsort(-np.mean(data, axis=1))) + 1
    
    ranks_dict = {method: int(rank) for method, rank in zip(methods, ranks)}

    return {"statistic": statistic, "p-value": p_value, "ranks": ranks_dict}

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

def generate_summary_table(data, methods, target_method, output_file):
    summary = {"Dataset": []}
    temp_summary = {}
    ranks = {method: [] for method in methods}

    for dataset, df in data.items():
        summary["Dataset"].append(dataset)

        # Calculate mean(std) for each method
        for method in methods:
            if method in df.columns:
                mean = np.mean(df[method])
                std = np.std(df[method])
                temp_summary.setdefault(method, []).append(mean)
                summary.setdefault(method, []).append(f"{mean:.2f}({std:.2f})")
            else:
                summary.setdefault(method, []).append("N/A")

        # Calculate ranks for valid methods
        valid_methods = [method for method in methods if method in df.columns]
        mean_values = [np.mean(df[method]) for method in valid_methods]        
        rank = rankdata([-v for v in mean_values], method="average")  # Higher is better
        for i, method in enumerate(valid_methods):
            ranks[method].append(rank[i])
            
    #Perform Wilcoxon test
    p_values = []
    for method, df in temp_summary.items():
        if method == target_method:
            target_df = df
            p_values.append("--")
        elif method != target_method:
            stats, p_value = wilcoxon(target_df, df)
            p_values.append(f"{p_value:.6f}")

    # calculate average ranks 
    avg_rank = [
        f"{np.mean(ranks[method]):.2f}" if ranks[method] else "N/A"
        for method in methods
    ]

           
    summary["Dataset"].append("Ave rank")
    summary["Dataset"].append("𝑝-value")
    # Create DataFrame from the summary dictionary
    for index , method in enumerate(methods):
        summary.setdefault(method, []).append(avg_rank[index])
        summary.setdefault(method, []).append(p_values[index])

        
    #print(summary)
    summary_df = pd.DataFrame(summary)

    # Save to Excel
    summary_df.to_excel(output_file, index=False, engine="openpyxl")
    print(f"Summary table saved to {output_file}")


def save_results_to_excel(win_tie_loss ,friedman_results, base_path):
    """Save results to Excel."""
    
    filename = base_path + "Win_Tie_Loss Results.xlsx"
    df_results = pd.DataFrame(win_tie_loss)
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df_results.to_excel(writer, index=False, sheet_name="win_tie_loss Results")
    
    filename = base_path + "Friedman Results.xlsx"
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        # Save Friedman Results
        friedman_df = pd.DataFrame({
            "Statistic": [friedman_results["statistic"]],
            "p-value": [friedman_results["p-value"]],
            "Ranks": [friedman_results["ranks"]]
        })
        filename 
        friedman_df.to_excel(writer, sheet_name="Friedman Results", index=False)


def save_boxplots(data, target_method, save_path):
    """Save box plots for DES_MHA across datasets."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(len(data) * 0.5, 10))
    results = [values[target_method] for values in data.values()]
    labels = list(data.keys())
    plt.boxplot(results, vert=True, labels=labels)
    plt.title(f"Box Plots for {target_method}")
    plt.xlabel("Datasets")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.xticks(rotation=90)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Box plots saved to {save_path}")

# Example Usage
if __name__ == "__main__":
    # Simulated data for demonstration
    
    methods = ['DES_MHA', 'KNORA-U', 'KNORAE', 'DESKNN', 'OLA', 'LCA', 'MLA', 'MCB', 'Rank', 'KNOP', 'META-DES']
    target = "DES_MHA"
    save_path= config.ExperimentPath + "Results"
    accuracy_data = load_accuracy_data(config.datasets, methods, config.ExperimentPath)
    
    # Win-Tie-Loss Summary and Plot
    wtl_summary = calculate_win_tie_loss(accuracy_data, target, methods)
    win_tie_loss = save_win_tie_loss_plot(wtl_summary, "All_Datasets")

    # Friedman Test
    friedman_results = friedman_test(accuracy_data)
    
    # Generate table
    generate_summary_table(accuracy_data, methods, "DES_MHA", "Results/summary_table.xlsx")

    # Save Results
    save_results_to_excel(win_tie_loss, friedman_results, "Results/")

    # Box Plots
    save_boxplots(accuracy_data, target, "./Results/des_mha_boxplots.png")
