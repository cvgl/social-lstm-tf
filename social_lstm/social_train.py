import tensorflow as tf
import argparse
import os
import time
import pickle
import numpy as np
#import ipdb

from social_model import SocialModel
from social_utils import SocialDataLoader
from grid import getSequenceGridMask

# CHK_DIR = '/cvgl2/u/junweiy/Jackrabbot/social-lstm-checkpoints/'
CHK_DIR = '/cvgl2/u/junweiy/Jackrabbot/test-checkpoints/'

def main():
    parser = argparse.ArgumentParser()
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # TODO: (improve) Number of layers not used. Only a single layer implemented
    # Number of layers parameter
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    # Model currently not used. Only LSTM implemented
    # Type of recurrent unit parameter
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=10,
                        help='minibatch size')
    # Length of sequence to be considered parameter
    parser.add_argument('--seq_length', type=int, default=20,
                        help='RNN sequence length')
    # Length of sequence to be considered parameter
    parser.add_argument('--obs_length', type=int, default=8,
                        help='Observed length of frames in a sequence')
    # Length of sequence to be considered parameter
    parser.add_argument('--pred_length', type=int, default=8,
                        help='Predicted length of frames in a sequence')

    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=200,
                        help='save frequency')
    # TODO: (resolve) Clipping gradients for now. No idea whether we should
    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    # Dropout not implemented.
    # Dropout probability parameter
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    # Size of neighborhood to be considered parameter
    parser.add_argument('--neighborhood_size', type=float, default=32,
                        help='Neighborhood size to be considered for social grid')
    # Size of the social grid parameter
    parser.add_argument('--grid_size', type=int, default=4,
                        help='Grid size of the social grid')
    # Maximum number of pedestrians to be considered
    parser.add_argument('--maxNumPeds', type=int, default=60,
                        help='Maximum Number of Pedestrians')

    parser.add_argument('--dataset_path', type=str, default='./../data/',
                        help='Path training data')
    parser.add_argument('--visible',type=str,
                        required=False, default=None, help='GPU to run on')
    parser.add_argument('--mode', type=str, default='social', 
                        help='social, occupancy, naive')
    parser.add_argument('--model_path', type=str)


    args = parser.parse_args()
    train(args)


def make_save_path(args):
    import datetime
    folder_name = args.mode
    now = datetime.datetime.now()
    timestamp = now.strftime("%m_%d_%H_%M")
    save_path = os.path.join(CHK_DIR, folder_name, timestamp)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    return save_path

def train(args):
    if args.visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible

    if args.model_path:
        save_path = args.model_path
    else:
        save_path = make_save_path(args)
    
    dataset_path = args.dataset_path
    log_path = os.path.join(save_path, 'log')
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    
    # Create the SocialDataLoader object
    data_loader = SocialDataLoader(args.batch_size, args.seq_length,
            args.maxNumPeds, dataset_path, forcePreProcess=True)
    print data_loader.num_batches
    # print data_loader.next_batch()

    with open(os.path.join(save_path, 'social_config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Create a SocialModel object with the arguments
    model = SocialModel(args)
    all_loss = []
    # Initialize a TensorFlow session
    with tf.Session() as sess:
        # Get the checkpoint state for the model
        ckpt = tf.train.get_checkpoint_state(save_path)

        if ckpt:
            # Restore the model at the checkpoint
            print ('loading model: ', ckpt.model_checkpoint_path)
            
            # Initialize a saver that saves all the variables in the graph
            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(log_path, sess.graph)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print 'initializing variables....'
            
            # Initialize all variables in the graph
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            summary_writer = tf.summary.FileWriter(log_path, sess.graph)

        # For each epoch
        for e in range(args.num_epochs):
            # Assign the learning rate value for this epoch
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            # Reset the data pointers in the data_loader
            data_loader.reset_batch_pointer()

            # For each batch
            for b in range(data_loader.num_batches):
                # Tic
                start = time.time()

                # Get the source, target and dataset data for the next batch
                # s_batch, t_batch are input and target data which are lists containing numpy arrays of size seq_length x maxNumPeds x 3
                # d is the list of dataset indices from which each batch is generated (used to differentiate between datasets)
                s_batch, t_batch, d = data_loader.next_batch()

                # variable to store the loss for this batch
                loss_batch = 0
                counter = 0

                # For each sequence in the batch
                for seq_num in range(data_loader.batch_size):
                    # s_seq, t_seq and d_batch contains the source, target and dataset index data for
                    # seq_length long consecutive frames in the dataset
                    # s_seq, t_seq would be numpy arrays of size seq_length x maxNumPeds x 3
                    # d_batch would be a scalar identifying the dataset from which this sequence is extracted
                    s_seq, t_seq, d_seq = s_batch[seq_num], t_batch[seq_num], d[seq_num]
                    
                    '''
                    if d_seq == 0 and datasets[0] == 0:
                        dataset_data = [640, 480]
                    else:
                        dataset_data = [720, 576]
                    '''
                    print 'Processing frame sequence ' + str(seq_num) + '.....................'
                    for starting_frame_index in range(args.seq_length - args.obs_length - args.pred_length):
                        sub_s_seq = s_seq[starting_frame_index:starting_frame_index + args.obs_length + args.pred_length, :, :]
                        sub_t_seq = t_seq[starting_frame_index:starting_frame_index + args.obs_length + args.pred_length, :, :]

                        grid_batch = getSequenceGridMask(sub_s_seq, [0, 0], args.neighborhood_size, args.grid_size)
                        
                        # for frame_index in range(args.seq_length):
                        #     print s_seq[frame_index, 0:10, :]

                        # Feed the source, target data
                        feed = {model.input_data: sub_s_seq, model.target_data: sub_t_seq, model.grid_data: grid_batch}

                        train_loss, train_counter, _ = sess.run([model.cost, model.counter, model.train_op], feed)

                        loss_batch += train_loss

                        counter += train_counter

                end = time.time()
                loss_batch = loss_batch / counter
                all_loss.append(loss_batch)
                print(
                    "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                    .format(
                        e * data_loader.num_batches + b,
                        args.num_epochs * data_loader.num_batches,
                        e,
                        loss_batch, end - start))

                # Save the model if the current epoch and batch number match the frequency
                if (e * data_loader.num_batches + b) % args.save_every == 0 and ((e * data_loader.num_batches + b) > 0):
                    checkpoint_path = os.path.join(save_path, 'social_model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))
                    np.savetxt(os.path.join(log_path, 'loss.txt'), np.asarray(all_loss))

if __name__ == '__main__':
    main()
