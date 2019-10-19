# Final_ML
Final_ML contains the code of interface and the model training process.

models.py contains the AlexNet class

test.py contains methods:

-create_model() saves the model in json format for reusability

-test() trains the model and saves the final weights in model.h5

-give_label(X) predicts the label of a given input X by loading 
pretrained weights

data.py contains methods:

-prep_data() takes references from train.csv
and uploaded images from folder train. Then it resizes it to 224*224*3
format and balances the first 6000 images. 

kaggle.py contains methods:

-kaggle_test() takes references from sample_submission.csv
(given by kaggle) and uploaded images from folder static

-generate_kaggle_csv() applied model prediction on each test sample and 
and saves the data set with labeled predictions on predictions.csv

main.py contains methods:

-analyzeFrame(frame) call method give_label

-homePage() renders the page of web application,
while requiesting Post method, calls analyzeFrame 
for a given input picture

-allowed_file() checks whether the uploaded file is in 
picture allowed formats


