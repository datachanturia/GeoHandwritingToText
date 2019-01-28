
# 1 Symbol processing

    In "./Images" folder we had handwritten texts in different formats
        We adjusted and ran "./Data-Processing/symbol_processing.py", which iterated
            over given images in "./Images/old" and extracted single characters
            from these texts
        After getting single images we manually sorted these characters and got
            "./Images/chars" folder with more than 3000 characters 300 on each
        Next step was changing image format and we manually reformated all character
            images into 25x25 format and saved them in same location
            
# 2 Data Processing
    
     For CNN we needed dataset, dataset in organized structure
        So we ran "./Data-Processing/data_processing.py" and all these character
            images were turned into dataset which had circa 3000 rows [elements]
            and 626 columns - 525 for each pixel from 25x25 image and 1 for
            labels - Georgian handwritten characters - which was saved in
            "./Data/train.csv"
        After getting dataset we divided it into 2 parts - training and test sets,
            training set was 90% of the data, chosen randomly from whole dataset
            and the test set was 10% of the data, which was left after randomly
            choosing training data. We saved these datasets in "./Data" folder:
            "./Data/ntrain", "./Data/ntest"
            
# 3 CNN and data training

    [ Note: in this section you need "numpy", "tqdm", "matplotlib" libraries ]
    - We had dataset and CNN ready and it was time to train our model
    - After running "python3 train_cnn.py model.pkl" for 1-2 hours, we got our model ready
        and saved it in "./Data" folder as "./Data/model.pkl"

# 4 Testing the Model

    - So we ran "python3 measure_performance.py" and we got our result - circa 96% accuracy.
    
# Materials Used in Project:
    https://www.coursera.org/learn/machine-learning/home/welcome
    https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
    https://www.kdnuggets.com/2018/04/building-convolutional-neural-network-numpy-scratch.html
    https://gluon.mxnet.io/chapter04_convolutional-neural-networks/cnn-scratch.html
    https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1
    https://towardsdatascience.com/build-your-own-convolution-neural-network-in-5-mins-4217c2cf964f