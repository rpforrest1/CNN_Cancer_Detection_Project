import pandas as pd
import os
import shutil
import pathlib
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_io as tfio


def prep_data(data_dir:str):
    """
    Preps data files for training.

    Args:
        data_dir: The main directory the data is in.
    
    Returns:
        df: DataFrame of the training labels.
        train_val_dir: The new directory of the training data. 
            Folders are populated based on the image's class.
    """

    # Read in labels and create a dataframe to handle file mapping
    df = pd.read_csv(f'{data_dir}/train_labels.csv')
    df['path'] = f'{data_dir}/train/' + df['id'] + '.tif'

    # Define the new filepaths for the training and validation images.
    # This setup allows the filepath the provide the label
    def create_new_path_string(row:pd.Series, set_type:str):
        """
        Creates the path string for the new image location.

        Args:
            row: A row in a DataFrame.
            set_type: The name for the dataset type (like train_val).
        
        Returns:
            The new path string
        """

        return row['path'].replace('/train/', f"/{set_type}_final/{row['label']}/")

    df['final_path'] = df.apply(create_new_path_string, axis=1, args=('train_val',))

    # Copy the images from the initial location, to their new locations.
    # No copying will be done if the file already exist.

    def copy_image(row:pd.Series):
        """
        Copies images from one location to another.

        Args:
            row: A row in a Dataframe. Should have a path column.
        """

        base_dir = '/'.join(row['final_path'].split('/')[:-1])
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        if not os.path.exists(row['final_path']):
            shutil.copyfile(row['path'], row['final_path'])

    _ = df.apply(copy_image, axis=1)

    train_val_dir = pathlib.Path(f'{data_dir}/train_val_final')


    return df, train_val_dir


def get_train_val_test_data(df:pd.DataFrame, use_half_data:bool=False):
    """
    Creates the training, validation, and test data splits.

    Args:
        df: The DataFrame to split
        use_half_data: A boolean where True means that the entire dataset
            is reduced by 50%.
    
    Returns:
        (train_df, val_df, test_df): The training, validation, and test datasets
    """

    if use_half_data:
        input_df, _ = train_test_split(df, test_size=0.5, stratify=df['label'], random_state=15)
    else:
        input_df = df

    train_df, test_val_df = train_test_split(input_df, test_size=0.3, stratify=input_df['label'], random_state=15)
    val_df, test_df = train_test_split(test_val_df, test_size=0.25, stratify=test_val_df['label'], random_state=15)

    print(f'Training set: 70%, Validation set: {0.3*0.75:.1%}, Test set: {0.3*0.25:.1%}')

    return train_df, val_df, test_df


def get_label(file_path):
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == ['0', '1']

    # Integer encode the label
    return tf.argmax(one_hot)


def decode_img(img):
    # Load tiff image
    img = tfio.experimental.image.decode_tiff(img)
    #Keep only 3 channels
    img = img[:,:,:3]
    # Normalize pixel values
    img = tf.cast(img, tf.float16)
    img = (img / 255.0)

    return img


def process_path(file_path):
    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    
    return img, label


def process_path_no_label(file_path):
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)

    return img


def configure_for_performance(ds, batch_size, shuffle=True):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.cache()

    #Turn off this shuffle so that images and labels could be re-mapped together
    if shuffle:
        ds = ds.shuffle(buffer_size=1000, seed=15) 
    
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def create_tensorflow_datasets(train_df:pd.DataFrame, val_df:pd.DataFrame, test_df:pd.DataFrame, batch_size:int):
    """
    Converts datasets into compatible tensorflow datasets.

    Args:
        train_df: The training dataset
        val_df: The validation dataset
        test_df: The test dataset
        batch_size: The batch size to use during training.

    Returns:
        train_ds_str, val_ds_str, test_ds_str: String versions 
            of the training, validation, and test datasets 
        train_ds, val_ds, test_ds: The final versions
            of the training, validation, and test datasets
    """

    train_ds_str = tf.data.Dataset.from_tensor_slices(train_df['final_path'].values)
    val_ds_str = tf.data.Dataset.from_tensor_slices(val_df['final_path'].values)
    test_ds_str = tf.data.Dataset.from_tensor_slices(test_df['final_path'].values)

    AUTOTUNE = tf.data.AUTOTUNE

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train_ds = configure_for_performance(train_ds_str.map(process_path, num_parallel_calls=AUTOTUNE),
                                         batch_size)
    val_ds = configure_for_performance(val_ds_str.map(process_path, num_parallel_calls=AUTOTUNE),
                                       batch_size)
    test_ds = configure_for_performance(test_ds_str.map(process_path, num_parallel_calls=AUTOTUNE),
                                        batch_size, shuffle=False)

    return train_ds_str, val_ds_str, test_ds_str, train_ds, val_ds, test_ds


def predict_on_kaggle_test_set(kaggle_test_dir:str, batch_size:int, model):
    """
    Creates a kaggle submission file.

    Args:
        kaggle_test_dir: The directory for the kaggle test data.
        batch_size: The batch size used during modeling
        model: The tensorflow model

    Returns:
        The final submission DataFrame
    """
    sub_test_dir = pathlib.Path(kaggle_test_dir)
    sub_test_size = len(list(sub_test_dir.glob('*.tif')))
    print('Submission test set size:', sub_test_size)

    sub_test_ds_str = tf.data.Dataset.list_files('./' + str(sub_test_dir/'*'), shuffle=False)

    AUTOTUNE = tf.data.AUTOTUNE
    sub_test_ds = configure_for_performance(sub_test_ds_str.map(process_path_no_label, num_parallel_calls=AUTOTUNE),
                                            batch_size, shuffle=False)

    x_sub = [x.decode().split('/')[-1].replace('.tif', '') for x in sub_test_ds_str.as_numpy_iterator()]
    y_sub_pred_flt = model.predict(sub_test_ds, verbose=1)
    y_sub_pred = y_sub_pred_flt.flatten()

    final_submission = pd.DataFrame({'id':x_sub, 'label':y_sub_pred})

    return final_submission