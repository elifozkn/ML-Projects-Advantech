from create_training_data import create_training_data
from preprocess import preprocess
from connect_to_database import get_data

'''
this script is used by PowerAutomate to trigger data update and preprocessing. 
'''
labeled_data = create_training_data()
preprocessed_data_train = preprocess(labeled_data) # this generates training
preprocessed_data_predict = preprocess(labeled_data,pred=True) #to_be_predicted data 


