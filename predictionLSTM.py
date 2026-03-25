import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import tensorflow as tf
from tensorflow.keras.models import load_model
tf.keras.config.enable_unsafe_deserialization()

# Load your trained model
predictor = load_model('Pose_Predictor_LSTM_30_25_3speeds.keras', custom_objects={"mse": tf.keras.losses.MeanSquaredError()})

# Constants
NUM_FRAMES = 31
NUM_JOINTS = 15
DIMENSIONS = 3
JOINT_NAMES = ['CLAV', 'C7', 'RSHO', 'LSHO', 'LAEL', 'RAEL',
               'LWPS', 'RWPS', 'L3', 'LHIP', 'RHIP', 'LKNE', 'RKNE', 'LHEE', 'RHEE']


class PosePredictor(Node):
    def __init__(self):
        super().__init__('pose_predictor')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'joint_array',
            self.listener_callback,
            10
        )
        self.preds=[]
        self.curr=[]
        self.buff_count=0
        self.curr_pub = self.create_publisher(MarkerArray, 'current_pose_markers', 10)
        self.pred_pub = self.create_publisher(MarkerArray, 'predicted_pose_markers', 10)
        
    def listener_callback(self, msg):
        data = np.array(msg.data).reshape((NUM_FRAMES, NUM_JOINTS, DIMENSIONS))
        self.predict_pose(data)

    def predict_pose(self, inp_set):
        
        correct=True
        for x in inp_set:
            for pt in x:
                if pt[2]==0:
                    correct=False
        if correct==False:
            print('❌')
        elif correct==True:
            print('✅')
            
        torso_scale = np.linalg.norm(inp_set[0][1]-inp_set[0][8])/10
        
        # outp_torso_lens = [np.linalg.norm(p[1]-p[8]) for p in outp_set[1:]]
        inp_poses = [p-p[8] for p in inp_set[1:]]
        # outp_poses = [p-p[8] for p in outp_set[1:]]
        past_disp = inp_set[1:,8,:]-inp_set[:-1,8,:]
        # future_disp = outp_set[1:,8,:]-outp_set[:-1,8,:]
        past_disp = np.expand_dims(past_disp,axis=1)
        # future_disp = np.expand_dims(future_disp,axis=1)
        past = np.concatenate([np.delete(inp_poses,8,axis=1)/torso_scale, past_disp], axis=1)
        past = np.expand_dims(past,axis=0)
        outp = predictor.predict(past)
        # print("past life", past)
        # print("future",outp[0])
        outp=outp[0]
        # print(outp.shape)
        # outp_L3 = np.sum(outp[0][:,14,:],axis=0)
        outp_L3 = np.sum(outp[:10,14,:],axis=0)
        outp_L3+=inp_set[-1,8,:]
        # outp_L3 = np.expand_dims(outp_L3,axis=1)
        # print(outp_L3.shape)
        # f_pose = np.delete(outp[0][-1],14,axis=0)
        f_pose = np.delete(outp[-10],14,axis=0)
        
        # print(f_pose.shape)
        f_pose*=torso_scale
        # outp_L3 = np.delete(outp_L3,0,axis=1)
        # print(outp_L3.shape)
        # print(f_pose.shape)
        # f_pose = np.sum([f_pose,outp_L3],axis=1)
        f_pose = f_pose+outp_L3
        f_pose = np.insert(f_pose,8,outp_L3,axis=0)
        outp_final = f_pose
        
        
        
        
        # outp_inc = np.sum(outp[0], axis=0)  # shape: (15, 3), in mmutp
        # outp_final = outp_inc + p_curr      # Predicted positions, in mm
        
        self.preds.append(outp_final)
        self.curr.append(inp_set[-1,:,:])
        self.buff_count+=1
        if(self.buff_count>=90):
            np.save('live_predictions.npy',np.array(self.preds))
            np.save('live_capture.npy',np.array(self.curr))
            self.buff_count=0
        

        # Publish both current and predicted poses
        curr_markers = self.publish_markers(inp_set[-1], 0, (0.0, 1.0, 0.0))   # Green
        pred_markers = self.publish_markers(outp_final, 100, (1.0, 0.0, 0.0))  # Red

        # Add bone links
        curr_links = self.publish_links(inp_set[-1], 200, (0.0, 1.0, 0.0))
        pred_links = self.publish_links(outp_final, 201, (1.0, 0.0, 0.0))

        curr_markers.markers.append(curr_links)
        pred_markers.markers.append(pred_links)

        self.curr_pub.publish(curr_markers)
        self.pred_pub.publish(pred_markers)

        # print("✅ Published current and predicted poses to RViz2.")

    def to_point(self, xyz):
        # Camera → ROS transform
        x_ros = float(xyz[0])         # forward (Z_cam)
        y_ros = float(xyz[1])        # left = -right (X_cam)
        z_ros = float(xyz[2])        # up = -down (Y_cam)

        pt = Point()
        pt.x = x_ros / 1000.0
        pt.y = y_ros / 1000.0
        pt.z = z_ros / 1000.0
        return pt

    def publish_markers(self, joints, marker_id_offset, color_rgb, frame_id="base"):
        marker_array = MarkerArray()
        for i, (name, (x, y, z)) in enumerate(zip(JOINT_NAMES, joints)):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.id = i + marker_id_offset
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = marker.scale.y = marker.scale.z = 0.05

            if (x, y, z) == (0, 0, 0):  # Occluded or missing
                marker.color.a = 0.2  # Dim
                marker.color.r = marker.color.g = marker.color.b = 0.5
            else:
                marker.color.a = 1.0
                marker.color.r, marker.color.g, marker.color.b = color_rgb

            marker.pose.position.x = float(x) / 1000.0
            marker.pose.position.y = float(y) / 1000.0
            marker.pose.position.z = float(z) / 1000.0
            marker_array.markers.append(marker)
        return marker_array

    def publish_links(self, joints, marker_id, color_rgb, frame_id="base"):
        connections = self.get_joint_connections()
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.02
        marker.color.a = 1.0
        marker.color.r, marker.color.g, marker.color.b = color_rgb

        name_to_idx = {name: i for i, name in enumerate(JOINT_NAMES)}
        for j1, j2 in connections:
            if j1 in name_to_idx and j2 in name_to_idx:
                p1 = joints[name_to_idx[j1]]
                p2 = joints[name_to_idx[j2]]

                # ✅ Use allclose instead of == for NumPy arrays
                if np.allclose(p1, 0) or np.allclose(p2, 0):
                    continue

                marker.points.append(self.to_point(p1))
                marker.points.append(self.to_point(p2))
        return marker


    def get_joint_connections(self):
        return [
            ('CLAV', 'C7'), ('C7', 'LSHO'), ('C7', 'RSHO'),
            ('LSHO', 'LAEL'), ('LAEL', 'LWPS'),
            ('RSHO', 'RAEL'), ('RAEL', 'RWPS'),
            ('C7', 'L3'), ('L3', 'LHIP'), ('L3', 'RHIP'),
            ('LHIP', 'LKNE'), ('LKNE', 'LHEE'),
            ('RHIP', 'RKNE'), ('RKNE', 'RHEE')
        ]

def main(args=None):
    rclpy.init(args=args)
    node = PosePredictor()
    # print("Subscribed to /joint_array and publishing to RViz2...")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
