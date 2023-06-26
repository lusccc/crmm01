import pandas as pd
import matplotlib.pyplot as plt

data_dirs = ['data/cr_cls2_mixed_st10_kw20', 'data/cr2_cls2_mixed_st10_kw20']

def merge_csv_files(data_dir):
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    val_df = pd.read_csv(f"{data_dir}/val.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")
    return pd.concat([train_df, val_df, test_df], ignore_index=True)

dfs = [merge_csv_files(data_dir) for data_dir in data_dirs]

def plot_rating_distribution(dfs, labels):
    width = 0.35
    fig, ax = plt.subplots()
    for idx, (df, label) in enumerate(zip(dfs, labels)):
        counts = df['Rating'].value_counts().sort_index()
        ax.bar([x + idx * width for x in counts.index], counts.values, width, label=label)
        for i, v in enumerate(counts.values):
            ax.text(i + idx * width, v, str(v), ha='center', va='bottom')

    ax.set_xticks([x + width/2 for x in counts.index])
    ax.set_xticklabels(['Junk', 'Good'])
    ax.set_xlabel('Rating')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Rating Distribution in Datasets')
    ax.legend()
    plt.show()

plot_rating_distribution(dfs, ['CCR-S', 'CCR-L'])