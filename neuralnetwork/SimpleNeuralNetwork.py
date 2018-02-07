from dataset.titanic_dataset import load_titanic_dataset
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('D:/Project workspace/Titanic_Project/dataset/train.csv')

'''Split the data into training set and validation set'''
dataset_x, dataset_t = load_titanic_dataset(train_data)
x_train, x_test, t_train, t_test = train_test_split(dataset_x, dataset_t, test_size = 0.2, random_state = 42)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)


''' 6 inputs for each person'''
x = tf.placeholder(tf.float32, shape = [None, 6])
''' 2 outputs for each person'''
t = tf.placeholder(tf.float32, shape = [None, 2])


w1 = tf.Variable(tf.random_normal([6, 10]), name = 'w1')
b1 = tf.Variable(tf.zeros([10]), name = 'b1')

w2 = tf.Variable(tf.random_normal([10, 20]), name = 'w2')
b2 = tf.Variable(tf.zeros([20]), name = 'b2')

w3 = tf.Variable(tf.random_normal([20, 2]), name = 'w3')
b3 = tf.Variable(tf.zeros([2]), name = 'b3')

y = tf.nn.softmax(
    tf.matmul(
    tf.matmul(
    tf.matmul(x
    , w1) + b1
    , w2) + b2
    , w3) + b3)

cross_entropy = -tf.reduce_sum(t * tf.log(y + 1e-10), reduction_indices = 1)
cost = tf.reduce_mean(cross_entropy)

train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

'''Save the parameters'''
sav = tf.train.Saver(max_to_keep = 3)


''' If ckpt file exists, load the parameters from the latest version'''
ckpt = tf.train.get_checkpoint_state('D:/Project workspace/Titanic_Project/neuralnetwork/')

if ckpt and ckpt.model_checkpoint_path:
    with tf.Session() as sess2:
        tf.global_variables_initializer().run()        
        
        print('Restore the latest parameters......')
        sav.restore(sess2, ckpt.model_checkpoint_path)
        
        for epoch in range(20):
            total_loss = 0.0
            
            for idx in range(len(x_train)):
                feed = { x : [x_train[idx]], t : [t_train[idx]]}
                
                _, loss = sess2.run([train_op, cost], feed_dict = feed)
                total_loss += loss
    
            print('Epoch: %4d, total loss = %.9f' % (epoch + 1, total_loss))
        print('Futher training is completed!')
        
        predict  = sess2.run( y , feed_dict = {x : x_test} )
        correct  = np.equal(np.argmax(predict, 1), np.argmax(t_test, 1))
        accuracy = np.mean(correct.astype(np.float32))
        
        print('Accuracy on validation set: %.9f' % accuracy)
        
        print('Saving parameters......')
        save_path = sav.save(sess2, 'D:/Project workspace/Titanic_Project/neuralnetwork/params.ckpt')

else:
    with tf.Session() as sess1:
        
        tf.global_variables_initializer().run()
        
        for epoch in range(20):
            total_loss = 0.0
            
            for idx in range(len(x_train)):
                feed = { x : [x_train[idx]], t : [t_train[idx]]}
                
                _, loss = sess1.run([train_op, cost], feed_dict = feed)
                total_loss += loss
    
            print('Epoch: %4d, total loss = %.9f' % (epoch + 1, total_loss))
        print('Training complete!')
        
        predict  = sess1.run( y , feed_dict = {x : x_test} )
        correct  = np.equal(np.argmax(predict, 1), np.argmax(t_test, 1))
        accuracy = np.mean(correct.astype(np.float32))
        
        print('Accuracy on validation set: %.9f' % accuracy)
        
        print('Saving parameters......')
        save_path = sav.save(sess1, 'D:/Project workspace/Titanic_Project/neuralnetwork/params.ckpt')
        
        
        
'''Testing stage and save the predictions as csv file'''
test_data  = pd.read_csv('D:/Project workspace/Titanic_Project/dataset/test.csv')
dataset_x, dataset_t = load_titanic_dataset(test_data)

with tf.Session() as sess3:
    tf.global_variables_initializer().run()  
    
    sav.restore(sess3, ckpt.model_checkpoint_path)
    
    predict  = np.argmax(sess3.run( y , feed_dict = {x : dataset_x} ), 1)
    
    submission = pd.DataFrame({
        'PassengerId' : dataset_t,
        'Survived': predict
        })
    
    submission.to_csv('titanic-submission.csv', index = False)