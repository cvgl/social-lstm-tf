#!/usr/bin/env python

import rospy
import sys
import os
import argparse
import time

import numpy as np
import tensorflow as tf
import pickle

from social_utils import SocialDataLoader
from social_model import SocialModel
from grid import getSequenceGridMask

from spencer_tracking_msgs.msg import TrackedPersons
from people_msgs.msg import PeoplePrediction
from people_msgs.msg import People
from people_msgs.msg import Person
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

CHK_DIR = '/cvgl2/u/junweiy/Jackrabbot/social-lstm-checkpoints/social/08_28_00_45'
# CHK_DIR = '/home/patrick/jr_catkin_ws/src/social-lstm-tf/social_lstm/checkpoints/social/08_28_00_45/'
# CHK_DIR = '/cvgl2/u/junweiy/Jackrabbot/social-lstm-obs-4-checkpoints/social/09_01_19_56/'

class Social_Lstm_Prediction():
    def __init__(self):
        self.node_name = 'social_lstm'
        rospy.init_node(self.node_name)
        rospy.on_shutdown(self.cleanup)

        self.obs_length = 4
        self.pred_length = 8
        self.frame_interval = 6
        self.max_pedestrians = 60
        self.dimensions = [0, 0]
        self.fps = 15
        self.frame_interval_index = 0
        self.time_resolution = float(self.frame_interval * self.obs_length / self.fps)

        # Define the path for the config file for saved args
        with open(os.path.join(CHK_DIR, 'social_config.pkl'), 'rb') as f:
            self.saved_args = pickle.load(f)
            
        rospy.loginfo("Creating the model.  This can take about 10 minutes...")

        # Create a SocialModel object with the saved_args and infer set to true
        self.social_lstm_model = SocialModel(self.saved_args, True)
        
        rospy.loginfo("Model created.")

        # Initialize a TensorFlow session
        self.sess = tf.InteractiveSession()

        # Initialize a saver
        saver = tf.train.Saver()

        # Get the checkpoint state for the model
        ckpt = tf.train.get_checkpoint_state(CHK_DIR)
        print ('loading model: ', ckpt.model_checkpoint_path)

        # Restore the model at the checkpoint
        saver.restore(self.sess, ckpt.model_checkpoint_path)

        # Dict of person_id -> [row_index in obs_seq]
        self.id_index_dict = {}
        self.vacant_rows = range(self.max_pedestrians)
        self.frame_num = 0
        self.obs_sequence = np.zeros((self.obs_length * self.frame_interval + self.frame_interval/2, self.max_pedestrians, 3))

        self.tracked_persons_sub = rospy.Subscriber("tracked_persons", TrackedPersons, self.predict, queue_size=1)
        self.pedestrian_prediction_pub = rospy.Publisher("predicted_persons", PeoplePrediction, queue_size=1)
        self.prediction_marker_pub = rospy.Publisher("predicted_persons_marker_array", MarkerArray, queue_size=1)
        
        # self.prev_frames = []
        # for i in range(self.prev_length):
        #     self.prev_frames.append({})

        rospy.loginfo("Waiting for tracked persons...")
        rospy.loginfo("Ready.")

    def predict(self, tracked_persons):
        print "********************** Processing new frame ******************************"
        start_time = time.time()

        # Initialize the markers array
        prediction_markers = MarkerArray()

        # Initialize the people predictions message
        people_predictions = PeoplePrediction()

        tracks = tracked_persons.tracks
        track_ids = [track.track_id for track in tracks]

        print "Number of people being tracked: ", len(tracks)
        print 'track ids in current frame is ', track_ids

        self.frame_num += 1
        # self.frame_interval_index += 1
        self.obs_sequence = np.delete(self.obs_sequence, 0, axis=0)
        
        existing_track_ids = self.obs_sequence[:, :, 0]
        print 'existing track ids: ', existing_track_ids.shape, existing_track_ids
        for track_id in self.id_index_dict.keys():
            if track_id not in existing_track_ids:
                self.vacant_rows.append(self.id_index_dict[track_id])
                del self.id_index_dict[track_id]


        curr_frame = np.zeros((1, self.max_pedestrians, 3))
        for track in tracks:
            track_id = track.track_id
            if track_id in self.id_index_dict:
                row_index = self.id_index_dict[track_id]
            else:
                row_index = self.vacant_rows[0]
                print 'vacant row is: ', self.vacant_rows
                del self.vacant_rows[0]
                self.id_index_dict[track_id] = row_index
            print 'row_index is: ', row_index
            curr_frame[0, row_index, :] = [track_id, track.pose.pose.position.x, track.pose.pose.position.y]

        self.obs_sequence = np.concatenate((self.obs_sequence, curr_frame), axis=0)

        if len(tracks) == 0 or self.frame_num < self.obs_sequence.shape[0]: # or self.frame_interval_index < self.frame_interval:
            self.pedestrian_prediction_pub.publish(people_predictions)
            self.prediction_marker_pub.publish(prediction_markers)
            return

        print "This is a predicting step............................"

        interpolated_obs_sequence = np.zeros((self.obs_length, self.max_pedestrians, 3))
        # Generate interpolated obs_sequence
        for step in range(self.obs_length, 0, -1):
            end_step = step * self.frame_interval + self.frame_interval/2 - 1
            start_step = end_step - self.frame_interval
            curr_seq = self.obs_sequence[start_step: end_step + 1, :, :]

            mean_seq_cords = np.mean(curr_seq[:, :, 1:], axis=0)
            all_zeros_rows = np.where(~mean_seq_cords.any(axis=1))[0]
            
            non_zeros_rows = np.where(mean_seq_cords.any(axis=1))[0]
            print non_zeros_rows
            nonzeros = np.nonzero(curr_seq[:, :, 0])
            print np.unique(nonzeros[1])
            if step < self.obs_length:
                mean_seq_cords[all_zeros_rows, :] = interpolated_obs_sequence[step, all_zeros_rows, 1:]
                interpolated_obs_sequence[step - 1, all_zeros_rows, 0] = interpolated_obs_sequence[step, all_zeros_rows, 0]

            interpolated_obs_sequence[step - 1, :, 1:] = mean_seq_cords
            interpolated_obs_sequence[step - 1, nonzeros[1], 0] = curr_seq[nonzeros[0], nonzeros[1], 0]

            # for seq_frame_index in range(self.frame_interval + 1):
            #     for pedIndex in non_zeros_rows:
            #         if curr_seq[seq_frame_index, pedIndex, 0] != 0:
            #             self.interpolated_obs_sequence[step - 1, pedIndex, 0] = curr_seq[seq_frame_index, pedIndex, 0]

        x_batch = np.concatenate((interpolated_obs_sequence, np.zeros((self.pred_length, self.max_pedestrians, 3))), axis=0)
        grid_batch = getSequenceGridMask(x_batch, self.dimensions, self.saved_args.neighborhood_size, self.saved_args.grid_size)

        complete_traj = self.social_lstm_model.sample(self.sess, interpolated_obs_sequence, x_batch, grid_batch, self.dimensions, self.pred_length)

        for frame_index in range(self.pred_length):
            people_one_time_step = People()
            people_one_time_step.header.stamp = tracked_persons.header.stamp + rospy.Duration(frame_index * self.time_resolution)
            people_one_time_step.header.frame_id = tracked_persons.header.frame_id
            
            predicted_frame_index = frame_index + self.obs_length
            
            for person_index in range(self.max_pedestrians):
                track_id = complete_traj[predicted_frame_index, person_index, 0]
                
                if track_id not in track_ids:
                   continue
               
                x_coord = complete_traj[predicted_frame_index, person_index, 1]
                y_coord = complete_traj[predicted_frame_index, person_index, 2]

                person_one_time_step = Person()
                person_one_time_step.name = str(track_id)

                point = Point()
                point.x = x_coord
                point.y = y_coord
                person_one_time_step.position = point
                people_one_time_step.people.append(person_one_time_step)
                
                prediction_marker = Marker()
                prediction_marker.type = Marker.SPHERE
                prediction_marker.action = Marker.MODIFY
                prediction_marker.ns = "predictor"
                prediction_marker.lifetime = rospy.Duration(0.1)
                prediction_marker.pose.orientation.w = 1
                prediction_marker.color.r = 0
                prediction_marker.color.g = 0
                prediction_marker.color.b = 0.5
                prediction_marker.scale.x = 0.2
                prediction_marker.scale.y = 0.2
                prediction_marker.scale.z = 0.2
                
                prediction_marker.header.stamp = tracked_persons.header.stamp
                prediction_marker.header.frame_id = tracked_persons.header.frame_id
                prediction_marker.id = int(frame_index + person_index * self.pred_length)
                prediction_marker.pose.position.x = person_one_time_step.position.x
                prediction_marker.pose.position.y = person_one_time_step.position.y
                #prediction_marker.color.a = 1 - (frame_index * 1.0 / (self.pred_length * 1.0))
                prediction_marker.color.a = 1.0
                prediction_markers.markers.append(prediction_marker)

            people_predictions.predicted_people.append(people_one_time_step)
     
        print people_predictions 

        self.frame_interval_index = 0
        print 'time spent for frame: ', time.time() - start_time
        self.pedestrian_prediction_pub.publish(people_predictions)
        self.prediction_marker_pub.publish(prediction_markers)

    '''
    def __interp_helper(self, y):
        """Helper to handle indices and logical indices of NaNs.

        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
              to convert logical indices of NaNs to 'equivalent' indices
        Example:
            >>> # linear interpolation of NaNs
            >>> nans, x= nan_helper(y)
            >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        """
print
        return np.isnan(y), lambda z: z.nonzero()[0]


    def __interpolate_1d_array(self, before_interpolated):
        nparray = np.array(before_interpolated)
        
        index = -1
        for i in range(self.prev_length):
            if np.isnan(nparray[i]) == False:
                index = i
                break

        for i in range(index):
            nparray[i] = nparray[index]

        nans, x= self.__interp_helper(nparray)
        nparray[nans]= np.interp(x(nans), x(~nans), nparray[~nans])
        return nparray


    def __generate_input(self, tracks):
        num_tracks = len(tracks)
        whole_array = []
        for i in range(num_tracks):
            track = tracks[i]
            print track_id = track.track_id
            
            history_positions_x = []
            history_positions_y = []
            for index in range(self.prev_length):
                history_positions_x.append(float('nan'))
                history_positions_y.append(float('nan'))
                if track_id in self.prev_frames[index]:
                    history_positions_x[index] = self.prev_frames[index][track_id][0]
                    history_positions_y[index] = self.prev_frames[index][track_id][1]

            print history_positions_x
            print history_positions_y
            
            history_positions_x = self.__interpolate_1d_array(history_positions_x)
            history_positions_y = self.__interpolate_1d_array(history_positions_y)
            tracks_array = np.zeros((self.obs_length, 3))
            tracks_array[:, 0] = track_id
            tracks_array[:, 1] = np.array(history_positions_x)[4:]
            tracks_array[:, 2] = np.array(history_positions_y)[4:]
            tracks_array = np.expand_dims(tracks_array, 1)

            print tracks_array

            if i == 0:
                whole_array = tracks_array
            else:
                whole_array = np.append(whole_array, tracks_array, axis=1)

        res_input = np.zeros((self.obs_length + self.prev_length, self.max_pedestrians, 3))
        res_input[:self.obs_length, :num_tracks, :] = whole_array
        return res_input
    '''


    def cleanup(self):
        print "Shutting down social lstm node"


def main(args):
    try:
        Social_Lstm_Prediction()
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down social lstm node."

if __name__ == '__main__':
    main(sys.argv)
