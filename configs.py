class argHandler(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    _descriptions = {'help, --h, -h': 'show this super helpful message and exit'}

    def setDefaults(self):
        self.define('train_csv', './data/all_data_pathology_latest.csv',
                    'path to training csv containing the images names and the labels')
        self.define('test_csv', './data/testing_set_pathology.csv',
                    'path to testing csv containing the images names and the labels')
        self.define('image_directory', './data/images_rana_cropped_224',
                    'path to folder containing the patient folders which containg the images')
        self.define('visual_model_name', 'EfficientNetB0',
                    'select from (VGG16, VGG19, DenseNet121, DenseNet169, DenseNet201, Xception, ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2, InceptionV3, InceptionResNetV2, NASNetMobile, NASNetLarge, MobileNet, MobileNetV2, EfficientNetB0 to EfficientNetB7). Note that the classifier layer is removed by default.')
        self.define('use_chexnet_weights', True,
                    'use pre-trained chexnet weights. Note only works with DenseNet121. The classifier layer is removed by default as with other models')
        self.define('chexnet_weights_path', 'pretrained_models/chexnet_densenet121_weights.h5', 'chexnet weights path')
        self.define('image_target_size', (224, 224, 3), 'the target size to resize the image')
        self.define('num_epochs', 100, 'maximum number of epochs')
        self.define('csv_label_columns', ['Pathology Classification'], 'the name of the label columns in the csv')
        self.define('classes', ['Normal', 'Benign', 'Malignant'],
                    'the names of the output classes')

        self.define('multi_label_classification', False,
                    'determines if this is a multi classification problem or not. It affects the loss function')

        self.define('classifier_layer_sizes', [0.9],
                    'a list describing the hidden layers of the classifier. Example [10,0.4,5] will create a hidden layer with size 10 then dropout wth drop prob 0.4, then hidden layer with size 5. If empty it will connect to output nodes directly.')
        self.define('conv_layers_to_train', -1,
                    'the number of layers that should be trained in the visual model counting from the end. -1 means train all and 0 means freezing the visual model')
        self.define('use_imagenet_weights', True, 'initialize the visual model with pretrained weights on imagenet')
        self.define('pop_conv_layers', 1,
                    'number of layers to be popped from the visual model. Note that the imagenet classifier is removed by default so you should not take them into considaration')
        self.define('final_layer_pooling', 'avg', 'the pooling to be used as a final layer to the visual model')
        self.define('load_model_path', './saved_models/pathology_all_efficientb0_2_224/best_model.hdf5',
                    'a path containing the checkpoints. If provided with load_model_name the system will continue the training from that point or use it in testing.')
        self.define('save_model_path', './saved_models/pathology_all_efficientb0_2_224',
                    'where to save the checkpoints. The path will be created if it does not exist. The system saves every epoch by default')
        self.define('save_best_model_only', True,
                    'Only save the best weights on validation loss')
        self.define('learning_rate', 1e-4, 'The optimizer learning rate')
        self.define('learning_rate_decay_factor', 0.5,
                    'Learning rate decay factor when validation loss stops decreasing')
        self.define('optimizer_type', 'Adam', 'Choose from (Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam)')
        self.define('gpu_percentage', 0.95, 'gpu utilization. If 0 it will use the cpu')
        self.define('batch_size', 16, 'batch size for training and testing')
        self.define('multilabel_threshold', 0.5,
                    'The threshold from which to detect a class. Only used with multi label classification.')
        self.define('generator_workers', 4, 'The number of cpu workers generating batches.')
        self.define('generator_queue_length', 6, 'The maximum number of batches in the queue to be trained on.')
        self.define('minimum_learning_rate', 1e-7, 'The minimum possible learning rate when decaying')
        self.define('reduce_lr_patience', 3,
                    'The number of epochs to reduce the learning rate when validation loss is not decreasing')
        self.define('show_model_summary', True, 'A flag to show or hide the model summary')
        self.define('positive_weights_multiply', 1.0,
                    'Controls the class_weight ratio between 0 and 1. Higher value means higher weighting of positive samples. Only works if use_class_balancing is set to true')
        self.define('use_class_balancing', True,
                    'If set to true it will automatically balance the classes by settings class weights')
        self.define('cnn_downscaling_factor', 0,
                    'Controls the cnn layers responsible for downscaling the input image. if input image is 512x512 and downscaling factor is set to 2 then the downscaling cnn will output image with size 128x128. Note it is a learnable net and if set to 0 it will skip it')
        self.define('cnn_downscaling_filters', 64, 'Number of filters in the downscaling model')
    def define(self, argName, default, description):
        self[argName] = default
        self._descriptions[argName] = description

    def help(self):
        print('Arguments:')
        spacing = max([len(i) for i in self._descriptions.keys()]) + 2
        for item in self._descriptions:
            currentSpacing = spacing - len(item)
            print('  --' + item + (' ' * currentSpacing) + self._descriptions[item])
        print('')
        exit()
