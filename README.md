# doodle_classifier

NOTE 1: Visit https://quickdraw.withgoogle.com to know more about quickdraw (and please don't compare my classifier with theirs).

NOTE 2: You may find it confusing if you're not familiar with deep learning or neural networks.

NOTE 3: There are only 5 dataset files to keep the size small because they are approximately 90 mb each. 

NOTE 4: Extract the zip files in the quickdraw_data/numpy_bitmap folder (and not in a subfolder) before running the program.

NOTE 5: You have to delete the 'doodle_classifier.h5' file in model folder and train the model again everytime you increase or decrease the number of datasets.

NOTE 6: If you don't get enough accuracy, try increasing the number of epochs or the total number of training data.

<h1> Data </h2>

This is a simple doodle classifier written in python which recognizes your doodles (drawings).
For simplicity's sake, there are only 5 datasets -- Apple, Cat, Computer, Eye, Headphones.

But you can easily download the datasets (in the form of numpy bitmaps, .npy files) for other categories and 
then train the model again.

To download the datasets for other categories, visit https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap?pli=1

Put the downloaded datasets in quickdraw_data/numpy_bitmap folder in .npy file format.

<h3> Modules required : </h3>

    os              - for getting filenames of the datasets
    
    numpy           - for loading data and working with arrays
    
    keras           - for creating and training the classifier
    
    openCV (cv2)    - for working with images (drawing, displaying etc.)
    
    sklearn         - for splitting the training and testing data and encoding the labels

<h3> Some parameters : </h3>
    
    path                - path of the directory in which the datasets are present
    
    filenames           - names of the dataset files
    
    label_map           - dictionary to map integer labels with their respective string labels
    
    train_features      - final features to train the model on
    
    train_labels        - final labels for training data
    
    test_features       - final features for evaluation of the model
    
    test_labels         - final labels for test data
    
    slice_train         - number of 
    
    MODEL_PATH          - path where the model will be saved
    
    n_classes           - number of classes (or categories) of the doodles. This will be automatically assigned as you put new datasets
    
    width, height       - width and height of the final images for training, testing and prediction
    
    depth               - number of channels in the images, here the images are grayscale, so depth is 1
    
    K.image_data_format - this is the format of the input data for the model. There are two formats, 'channels_first' and 'channels_last'.
                          The input data will be shaped according to these formats.
                          'channels_first' corresponds to (num_samples, depth, width, height)
                          'channels_last' corresponds to (num_samples, width, height, depth)

<h2> Model </h2>

The program will try to load the model, if it doesn't exist, a new model will be created and trained.

To train the model from scratch, just delete the .h5 model file in the model folder.

<h2> OpenCV </h2>

There are 2 windows, one for drawing and another for displaying the prediction made on that drawing.
The cv2 (OpenCV) module is responsible for the drawing and displaying of the images.

