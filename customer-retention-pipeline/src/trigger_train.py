from train import train

''' 
this script is used by PowerAutomate to trigger the training process, with the latest updated training data. 
The model information is then saved to the database. 
'''
train()
print("train success")