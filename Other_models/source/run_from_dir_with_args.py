import pretrained_unet
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
        'exp' : 'train',
        # for gaula 47 we use 47 prefix
        'exp_no' : 601,
        # 'model_name' : 'fpn_resnet50',
        'model_name' : 'deeplabv3_resnet50',
        'freeze' : '0',
        'dropout' : 0.1,
        # 'batch_size' : 16,
        'batch_size' : 16,
        'image_name' : 'image',
        'label_name' : 'label',
        'lr' : 0.0001,
        # 'lr' : 0.0005,
        'num_classes': 6,
        'use_swa': False,
        'callback_patience': 10,
        'swa_start_epochs': 10,
        'use_albumentation': True,
        'img_per_epoch': 9353,
        # 'img_per_epoch': 14900,

        # 'img_per_epoch': 21514,
        # 'img_per_epoch': 57648,
        # 'img_per_epoch': 38424,
        'regularizer_flag': True,
        'use_grayscale_only': False,
        'adam_schedule': False,
        'class_weight': True,
        'use_reduce_lr_platueau': True,
        # before it was at 10
        # 'train_type_commnet' : 'vgg with albom2 aaai paper first try with album v2, swa on 10 5 e-5, no class weight.',
        'train_type_commnet' : 'Training DeepLabV3+ without AlbumV2 and SWA',
    }

    
                 


    # train_data_folder_path = r"/data/nips/EX2/trian"
    # val_data_folder_path = r"/data/nips/EX2/val"
    train_data_folder_path = r"/data/nips/EX3/ML_dataset/train"
    val_data_folder_path = r"/data/nips/EX3/ML_dataset/val"
    # train_data_folder_path = r"/data/arild_dataset/png_train"
    # val_data_folder_path = r"/data/arild_dataset/val"
    run_path = f"/data/nips/exp/exp_no{conf_exp['exp_no']}"
    os.makedirs(run_path, exist_ok=True)


    # # write the config to file (in a hacky way)
    # file_name = 'config.txt'
    # with open(os.path.join(run_path, file_name), 'w') as convert_file:
    #     convert_file.write(json.dumps(conf_exp))

    repeat_experiment = 8
    repeat_begin = 0

    for exp_train_no in range(repeat_begin, repeat_experiment):
        model_path = r""
        if conf_exp['exp'] == 'train':
            print('------------------------------------------------------------------------------------------')
            print('-------------------------------------------train------------------------------------------')
            print(f'---------------------------------------------{exp_train_no}--------------------------------------------')
            print('------------------------------------------------------------------------------------------')

            conf_exp['seed'] = exp_train_no

            pretrained_unet.run_from_dir(
                train_data_folder_path=train_data_folder_path, val_data_folder_path=val_data_folder_path, 
                model_name=conf_exp['model_name'], freeze=conf_exp['freeze'], run_path=run_path, 
                dropout=conf_exp['dropout'], batch_size=conf_exp['batch_size'], 
                train_image_directory_name=conf_exp['image_name'], train_label_directory_name=conf_exp['label_name'], 
                val_image_directory_name=conf_exp['image_name'], val_label_directory_name=conf_exp['label_name'], 
                lr=conf_exp['lr'], num_classes =conf_exp['num_classes'], conf_exp=conf_exp, use_swa=conf_exp['use_swa'], 
                callback_patience=conf_exp['callback_patience'], swa_start_epochs=conf_exp['swa_start_epochs'],
                use_albumentation=conf_exp['use_albumentation'], img_per_epoch=conf_exp['img_per_epoch'],
                regularizer_flag=conf_exp['regularizer_flag'], use_grayscale_only=conf_exp['use_grayscale_only'],
                adam_schedule=conf_exp['adam_schedule'], class_weight=conf_exp['class_weight'],
                use_reduce_lr_platueau=conf_exp['use_reduce_lr_platueau'], seed_val=exp_train_no)

        elif conf_exp['exp']  == 'test': 
            print('-------------------------------------------test------------------------------------------')
            pretrained_unet.evaluate_model_from_dir(model_path=model_path, val_data_folder_path=val_data_folder_path, freeze=freeze)
        else:
            raise ValueError("exp should be test or train")
    print(f'experiment {repeat_experiment} is done')