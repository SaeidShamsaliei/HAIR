import fine_tune
import sys
import os
import json

if __name__ == '__main__':
    """
    Fine-Tunes the trained model
    """
    conf_exp = {
        'exp': 'train',
        'exp_no': 611,
        'trained_model_path': r'/data/nips/exp/exp_no591/2023-04-05_14:48:37.874381_deeplabv3_resnet50_freeze_0/model.hdf5',
        'model_name': 'deeplabv3_resnet50',
        'dropout': 0.1,
        # 'batch_size' : 12,
        'batch_size': 16,
        'image_name': 'image',
        'num_classes': 6,
        'label_name': 'label',
        'lr': 0.0001,
        'use_swa': False,
        'callback_patience': 10,
        'swa_start_epochs': 10,
        'use_albumentation': True,
        'img_per_epoch': 9353,
        'use_grayscale_only': False,
        'class_weight': True,
        'use_reduce_lr_platueau': True,
        'train_type_commnet': 'DeepLabV3+, same config as before, Training DeepLabV3+ with AlbumV2, changing the error of last layer',
    }

    train_data_folder_path = r"/data/nips/EX3/ML_dataset/train"
    val_data_folder_path = r"/data/nips/EX3/ML_dataset/val"

    run_path = f"/data/nips/exp/exp_no{conf_exp['exp_no']}"
    os.makedirs(run_path, exist_ok=True)

    repeat_experiment = 6
    repeat_begin = 0

    for exp_train_no in range(repeat_begin, repeat_experiment):
        if conf_exp['exp'] == 'train':
            print(
                '------------------------------------------------------------------------------------------')
            print(
                '-------------------------------------------train------------------------------------------')
            print(
                f'---------------------------------------------{exp_train_no}--------------------------------------------')
            print(
                '------------------------------------------------------------------------------------------')

            conf_exp['seed'] = exp_train_no

            fine_tune.finetune_from_dir(
                trained_model_path=conf_exp['trained_model_path'],
                train_data_folder_path=train_data_folder_path,
                val_data_folder_path=val_data_folder_path,
                model_name=conf_exp['model_name'],
                run_path=run_path,
                batch_size=conf_exp['batch_size'],
                train_image_directory_name=conf_exp['image_name'],
                train_label_directory_name=conf_exp['label_name'],
                val_image_directory_name=conf_exp['image_name'],
                val_label_directory_name=conf_exp['label_name'],
                lr=conf_exp['lr'], conf_exp=conf_exp, use_swa=conf_exp['use_swa'],
                callback_patience=conf_exp['callback_patience'],
                swa_start_epochs=conf_exp['swa_start_epochs'],
                use_albumentation=conf_exp['use_albumentation'],
                img_per_epoch=conf_exp['img_per_epoch'],
                use_grayscale_only=conf_exp['use_grayscale_only'],
                class_weight=conf_exp['class_weight'],
                use_reduce_lr_platueau=conf_exp['use_reduce_lr_platueau'],
                num_classes=conf_exp['num_classes'],
                seed_val=exp_train_no)

        elif conf_exp['exp'] == 'test':
            print(
                '-------------------------------------------test------------------------------------------')
            raise NotImplementedError('it is not implemented')
        else:
            raise ValueError("exp should be test or train")
    print(f'experiment {repeat_experiment} is done')
