from utils.image_processing_utils import dataset_split_train_vlidation_test_notebook

if __name__ == '__main__':

    """
    it goes to src_path/image/*.png and add the to dst_path/image/img/
    """

    train_rate = 0.8
    val_rate = 0.2
    
    src_path = '/data/nips//EX3/all_train_512_png'
    dst_path = '/data/nips/EX3/ML_dataset'

    print('spliting the dataset')

    dataset_split_train_vlidation_test_notebook(
        src_path=src_path, dst_path=dst_path,
        train_rate=train_rate,
        val_rate=val_rate)
        
    print('done')
