import rospy
import sys
import os
import argparse

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

class Social_Lstm_Prediction():
    def __init__(self):
        self.node_name = 'social_lstm'

        rospy.init_node(self.node_name)
        rospy.on_shutdown(self.cleanup)

        self.obs_length = 8
        self.pred_length = 8
        self.max_pedestrians = 60
        self.dimensions = [0, 0]
        self.time_resolution = 0.5

        # Define the path for the config file for saved args
        with open(os.path.join(CHK_DIR, 'social_config.pkl'), 'rb') as f:
            self.saved_args = pickle.load(f)

        # Create a SocialModel object with the saved_args and infer set to true
        self.social_lstm_model = SocialModel(self.saved_args, True)
        
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
        self.obs_sequence = np.zeros((self.obs_length, self.max_pedestrians, 3))

        self.tracked_persons_sub = rospy.Subscriber("/tracked_persons", TrackedPersons, self.predict)
        self.pedestrian_prediction_pub = rospy.Publisher("/predicted_persons", PeoplePrediction, queue_size=1)
        self.prediction_marker_pub = rospy.Publisher("/predicted_persons_marker_array", MarkerArray, queue_size=1)
        
        # Initialize the marker for people prediction
        self.prediction_marker = Marker()
        self.prediction_marker.type = Marker.SPHERE
        self.prediction_marker.action = Marker.MODIFY
        self.prediction_marker.ns = "people_predictions"
        self.prediction_marker.pose.orientation.w = 1
        self.prediction_marker.color.r = 0
        self.prediction_marker.color.g = 0
        self.prediction_marker.color.b = 0.5
        self.prediction_marker.scale.x = 0.2
        self.prediction_marker.scale.y = 0.2
        self.prediction_marker.scale.z = 0.2

        # self.prev_frames = []
        # for i in range(self.prev_length):
        #     self.prev_frames.append({})

        rospy.loginfo("Waiting for tracked persons...")
        rospy.wait_for_message("/predicted_persons", PeoplePrediction)
        rospy.loginfo("Ready.")

    def predict(self, tracked_persons):
        self.frame_num += 1
        tracks = tracked_persons.tracks
        # print tracks

        if len(tracks) == 0:
            return

        print 'before delete: ', self.obs_sequence.shape
        self.obs_sequence = np.delete(self.obs_sequence, 0, axis=0)
        print 'after delete: ', self.obs_sequence.shape
        if self.frame_num >= self.obs_length:
            existing_track_ids = self.obs_sequence[:, :, 0]
            for track_id in self.id_index_dict.keys():
                if track_id not in existing_track_ids:
                    self.vacant_rows.append(self.id_index_dict[track_id])
                    del self.id_index_dict[track_id]


        curr_frame = np.zeros((1, self.max_pedestrians, 3))
        for track in tracks:
            # print track
            # print track.pose.pose.position.x
            track_id = track.track_id
            if track_id in self.id_index_dict:
                row_index = self.id_index_dict[track_id]
            else:
                row_index = self.vacant_rows[0]
                del self.vacant_rows[0]
                self.id_index_dict[track_id] = row_index
            curr_frame[0, row_index, :] = [track_id, track.pose.pose.position.x, track.pose.pose.position.y]

        self.obs_sequence = np.concatenate((self.obs_sequence, curr_frame), axis=0)

        if self.frame_num < self.obs_length:
            return

        print self.obs_sequence.shape
        x_batch = np.concatenate((self.obs_sequence, np.zeros((self.pred_length, self.max_pedestrians, 3))), axis=0)
        grid_batch = getSequenceGridMask(x_batch, self.dimensions, self.saved_args.neighborhood_size, self.saved_args.grid_size)

        print "********************** PREDICT NEW TRAJECTORY ******************************"
        complete_traj = self.social_lstm_model.sample(self.sess, self.obs_sequence, x_batch, grid_batch, self.dimensions, self.pred_length)
        
        # Initialize the markers array
        prediction_markers = MarkerArray()

        # Publish them
        people_predictions = PeoplePrediction()
        for frame_index in range(self.pred_length):
            people = People()
            people.header.stamp = tracked_persons.header.stamp + rospy.Duration(frame_index * self.time_resolution);
            people.header.frame_id = tracked_persons.header.frame_id
            
            predicted_frame_index = frame_index + self.obs_length
            for person_index in range(self.max_pedestrians):
                track_id = complete_traj[predicted_frame_index, person_index, 0]
                x_coord = complete_traj[predicted_frame_index, person_index, 1]
                y_coord = complete_traj[predicted_frame_index, person_index, 2]
                if track_id == 0:
                    continue

                person = Person()
                person.name = str(track_id)

                point = Point()
                point.x = x_coord
                point.y = y_coord
                person.position = point
                people.people.append(person)
                
                self.prediction_marker.header.frame_id = tracked_persons.header.frame_id
                self.prediction_marker.header.stamp = tracked_persons.header.stamp
                self.prediction_marker.id = int(track_id);
                self.prediction_marker.pose.position.x = person.position.x
                self.prediction_marker.pose.position.y = person.position.y
                # self.prediction_marker.color.a = 1 - (frame_index * 1.0 / (self.pred_length * 1.0))
                self.prediction_marker.color.a = 1.0
                prediction_markers.markers.append(self.prediction_marker)

            people_predictions.predicted_people.append(people)
     
        print people_predictions 

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
            track_id = track.track_id
            
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
