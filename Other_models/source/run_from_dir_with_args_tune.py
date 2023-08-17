import pretrained_unet_tune
import sys
import os
import json

if __name__ == '__main__':
    """
    Runs the pretrained_unet with arguments using a data generator that flows from a directory. 
    """

    # important point, for validation, we just pass the tiff files
    # model_names = [deeplabv3_resnet50, deeplabv3, vgg16]
    conf_exp = {
        'exp' : 'tune',
        # for gaula 47 we use 47 prefix
        'exp_no' : 575,
        # 'model_name' : 'unet_resnet50',
        'model_name' : 'hrnet',
        'freeze' : '0',
        # for hrnet we need to do 12
        'batch_size' : 12,
        # 'batch_size' : 16,
        'image_name' : 'image',
        'label_name' : 'label',
        'num_classes': 7,
        'use_swa': False,
        'callback_patience': 10,
        'swa_start_epochs': 10,
        'use_albumentation': True,
        'img_per_epoch': 14900,
        'regularizer_flag': True,
        'use_grayscale_only': False,
        'adam_schedule': False,
        'class_weight': True,
        'use_reduce_lr_platueau': True,
        'hp_max_epochs': 100, 
        'hp_factor': 3, 
        'hp_hyperband_iterations': 1,
        'train_type_commnet': 'HP search for the DeepGlobe with hrnet, it searches the lr and dropout',
    }


    train_data_folder_path = r"/data/nips/deepglobe_grayscale/train"
    val_data_folder_path = r"/data/nips/deepglobe_grayscale/val"
    # train_data_folder_path = r"/data/arild_dataset/png_train"
    # val_data_folder_path = r"/data/arild_dataset/val"
    run_path = f"/data/nips/exp/exp_no{conf_exp['exp_no']}"
    os.makedirs(run_path, exist_ok=True)


    repeat_experiment = 1
    repeat_begin = 0

    for exp_train_no in range(repeat_begin, repeat_experiment):
        model_path = r"/workspace/runs/2021-05-29_15:50:27.537332_vgg16_freeze_2/model.hdf5"
        if conf_exp['exp'] == 'tune':
            print('------------------------------------------------------------------------------------------')
            print('-------------------------------------------tune------------------------------------------')
            print(f'---------------------------------------------{exp_train_no}--------------------------------------------')
            print('------------------------------------------------------------------------------------------')

            conf_exp['seed'] = exp_train_no

            pretrained_unet_tune.tune_from_dir_all(
                train_data_folder_path=train_data_folder_path, val_data_folder_path=val_data_folder_path, 
                model_name=conf_exp['model_name'], run_path=run_path, batch_size=conf_exp['batch_size'], 
                train_image_directory_name=conf_exp['image_name'], 
                train_label_directory_name=conf_exp['label_name'], 
                val_image_directory_name=conf_exp['image_name'], val_label_directory_name=conf_exp['label_name'], 
                num_classes =conf_exp['num_classes'], conf_exp=conf_exp, use_swa=conf_exp['use_swa'], 
                callback_patience=conf_exp['callback_patience'], swa_start_epochs=conf_exp['swa_start_epochs'],
                use_albumentation=conf_exp['use_albumentation'], img_per_epoch=conf_exp['img_per_epoch'],
                regularizer_flag=conf_exp['regularizer_flag'], use_grayscale_only=conf_exp['use_grayscale_only'],
                adam_schedule=conf_exp['adam_schedule'], class_weight=conf_exp['class_weight'],
                use_reduce_lr_platueau=conf_exp['use_reduce_lr_platueau'], seed_val=conf_exp['seed'],
                hp_max_epochs=conf_exp['hp_max_epochs'], hp_factor=conf_exp['hp_factor'], 
                hp_hyperband_iterations=conf_exp['hp_hyperband_iterations'])

        elif conf_exp['exp']  == 'test': 
            print('-------------------------------------------test------------------------------------------')
            pretrained_unet.evaluate_model_from_dir(model_path=model_path, val_data_folder_path=val_data_folder_path, freeze=freeze)
        else:
            raise ValueError("exp should be test or train")
    print(f'experiment {repeat_experiment} is done')