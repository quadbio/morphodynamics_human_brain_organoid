import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import anndata as ad
from pathlib import Path

# Define import function
def convert_to_h5ad(sample:str,
                    dir_input:str,
                    dir_output:str,
                    stat:str = 'median',
                    log_transform=True) -> ad.AnnData:
    """
    Convert csv file to h5ad file.
    :param stat:
    :param sample:
    :param dir_input:
    :param dir_output:
    :param log_transform:
    :param dim:
    :param filter_area:
    :return:
    """
    # generate output directory
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    file = Path(dir_input, f'{sample}.csv')
    if file.is_file():
        df = pd.read_csv(file)
        df.rename(columns={'ID': 'label'}, inplace=True)
        df_counts = df.loc[:, df.columns[df.columns.str.contains(f'{stat}')]]
        df_counts.columns = df_counts.columns.str.replace(f'intensity_{stat}_', '')
        colnames_sorted = sorted(df_counts.columns)
        df_counts = df_counts[colnames_sorted].values
        if log_transform:
            df_counts = np.log1p(df_counts)
        adata = sc.AnnData(X=df_counts)
        adata.var_names = colnames_sorted
        adata.obs['ID'] = df['label'].values
        adata.obs['sample'] = sample
        adata.obs['area'] = df['area'].values
        adata.obs['X'] = df['centroid-1'].values
        adata.obs['Y'] = df['centroid-0'].values
        # save to h5ad
        adata.write_h5ad(Path(dir_output, f'{sample}.h5ad'))
    else:
        raise ValueError(f'File {file} not found.')

def plot_summary(sample:str, dir_input:str,dir_output:str, scale:bool = True, dim:int = 10, resolution:float = 0.5,
                 show_plot:bool = False,save_plot:bool = True):
    # generate output directory
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    # load h5ad file
    adata = ad.read_h5ad(Path(dir_input, f'{sample}.h5ad'))
    if scale:
        sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=dim)
    sc.pp.neighbors(adata, n_pcs=dim)
    sc.tl.louvain(adata, resolution=resolution)
    sc.tl.umap(adata, n_components=dim)

    df_plot = adata.to_df()
    features = df_plot.columns.tolist()
    # Add UMAP coordinates to metadata
    df_plot['UMAP_1'] = adata.obsm['X_umap'][:, 0]
    df_plot['UMAP_2'] = adata.obsm['X_umap'][:, 1]
    df_plot['X'] = adata.obs['X']
    df_plot['Y'] = adata.obs['Y']
    df_plot['louvain'] = adata.obs['louvain']
    # convert louvain to int type
    df_plot['louvain'] = df_plot['louvain'].astype(int)

    # plot umap louvain and spatial louvain
    cmap = plt.get_cmap('jet', len(df_plot['louvain'].unique()))
    cmap.set_under('gray')
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].scatter(df_plot['UMAP_1'], df_plot['UMAP_2'], c=df_plot['louvain'], cmap=cmap, s=1)
    axs[0].set_title('UMAP louvain', size=20, fontweight='bold')
    axs[0].axis('off')
    axs[1].scatter(df_plot['X'], df_plot['Y'], c=df_plot['louvain'], cmap=cmap, s=1)
    axs[1].set_title('Spatial louvain', size=20, fontweight='bold')
    axs[1].invert_yaxis()
    axs[1].set_aspect('equal', 'box')
    axs[1].axis('off')
    # add colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
    fig.colorbar(axs[0].collections[0], cax=cbar_ax, ticks=range(cmap.N), shrink=0.6)
    if save_plot:
        plt.savefig(Path(dir_output, f'{sample}_louvain.png'))
    if show_plot:
        plt.show()
    plt.close('all')

    beach_colscheme = ['#cdcdcd', '#edf8b1', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84']
    # generate colormap for beach_colscheme
    beach_cmap = LinearSegmentedColormap.from_list('beach_cmap', beach_colscheme)
    # define nrows and ncols
    n_features = len(features)
    n_cols = 8
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(40, 40))
    fig.tight_layout(pad=0.5)
    plt.subplots_adjust(wspace=0, hspace=0)
    for i, ax in enumerate(axs.flatten()):
        if i < len(features):
            pcm = ax.scatter(df_plot['UMAP_1'], df_plot['UMAP_2'], c=df_plot[features[i]], cmap=beach_cmap, s=1)
            ax.set_title(f'{features[i]}', size=30, fontweight='bold')
            ax.set_aspect('equal', 'box')
            ax.axis('off')
            clb = fig.colorbar(pcm, ax=ax, shrink=0.5)
            clb.ax.tick_params(labelsize=15)
        else:
            # empty axis
            ax.axis('off')

    if show_plot:
        plt.show()
    # save plot
    if save_plot:
        plt.savefig(Path(dir_output, f'{sample}_features_umap.png'))
    plt.close('all')
    # spatial feature plots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(40, 40))
    # do tight plot layout
    fig.tight_layout(pad=0.5)
    # tight layout between subplots
    plt.subplots_adjust(wspace=0, hspace=0)
    for i, ax in enumerate(axs.flatten()):
        if i < len(features):
            pcm = ax.scatter(df_plot['X'], df_plot['Y'], c=df_plot[features[i]], cmap=beach_cmap, s=1)
            ax.set_title(f'{features[i]}', size=30, fontweight='bold')
            # switch y axis
            ax.invert_yaxis()
            # equal x and y
            ax.set_aspect('equal', 'box')
            ax.axis('off')
            clb = fig.colorbar(pcm, ax=ax, shrink=0.5)
            clb.ax.tick_params(labelsize=15)
        else:
            # empty axis
            ax.axis('off')
    if save_plot:
        plt.savefig(Path(dir_output, f'{sample}_features_spatial.png'))
    if show_plot:
        plt.show()
    plt.close('all')
