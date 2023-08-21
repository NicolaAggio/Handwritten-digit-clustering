import os
from os import path

# Directories

def makedir(path_: str):
    """
    Creates a directory (if the directory does not exist)
    :param path_: path of directory to be logged
    """
    try:
        os.makedirs(path_)
        print(f"Created directory {path_} !")
    except OSError:
        pass

def get_root_dir() -> str:
    """
    :return: path to root directory
    """
    return str(path.abspath(path.join(__file__, "../")))

def get_dataset_dir() -> str:
    """
    :return: path to dataset directory
    """
    return path.join(get_root_dir(), "datasets")

# Plots

def plot_digit(pixels: np.array, save: bool = False,
               file_name: str = "digit"):
    """
    Plot a figure given a square matrix array, each cell represent a grey-scale pixel with intensity 0-1
    :param pixels: intensity of pixels
    :param save: true for storing the image
    :param file_name: name file if stored
    """

    fig, ax = plt.subplots(1)
    pixels = chunks(lst=pixels, n=SIZE)
    ax.imshow(pixels, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    if save:
        file = path.join(get_images_dir(), f"{file_name}.{IMG_EXT}")
        makedir(get_images_dir())
        print(f"Saving {file}..")
        plt.savefig(file)  

    plt.show()

def plot_mean_digit(X: pd.DataFrame, save: bool = False,
                    file_name: str = "mean_digit"):
    """
    Plots the average figure of a certain number of images
    :param X: set of images
    :param save: if true, image is stored
    :param file_name: name of file if stored
    """

    pixels = np.mean(X, axis=0)
    plot_digit(pixels=pixels, save=save, file_name=file_name)

def digits_histogram(labels: pd.DataFrame | np.ndarray,
                     save: bool = False, file_name: str = "plot"):
    """
    Plot distribution of labels in a dataset given its labels

    :param labels: collection with labels
    :param save: if true, the image is stored in the directory
    :param file_name: name of file if stored (including extension)
    """

    # type-check and casting
    if type(labels) == np.ndarray:
        labels = pd.DataFrame(labels)

    # digits count
    digits: Dict[str, int] = {
        k[0]: v for k, v in labels.value_counts().to_dict().items()
    }

    # plot
    fig, ax = plt.subplots(1)
    ax.bar(list(digits.keys()), digits.values(), edgecolor='black')
    ax.set_xticks(range(10))
    ax.set_title('Digits distribution')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Counts')

    if save:
        file = path.join(get_images_dir(), f"{file_name}.{IMG_EXT}")
        makedir(get_dataset_dir())
        print(f"Saving {file}..")
        plt.savefig(file)  

    plt.show()

def plot_cluster_frequencies_histo(frequencies: Dict[int, int], save: bool = False, file_name: str = 'frequencies'):
    """
    Plot clusters frequencies in a histogram
    :save: if to save the graph to images directory
    :file_name: name of stored file
    """
    fig, ax = plt.subplots(1)

    ax.bar(list(frequencies.keys()), frequencies.values(), edgecolor='black')

    # Title and axes
    ax.set_title('Clusters cardinality')
    ax.set_xlabel('Cluster dimension')
    ax.set_ylabel('Occurrences')

    if save:
        makedir(get_images_dir())
        out_file = path.join(get_images_dir(), f"{file_name}.{IMG_EXT}")
        print(f"Saving {out_file}..")
        plt.savefig(out_file)
    plt.show()