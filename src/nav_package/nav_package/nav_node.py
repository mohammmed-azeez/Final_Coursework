#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import cv2
import numpy as np
import math
import time

from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import Image, LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge, CvBridgeError


class FieldExplorerNode(Node):
    def __init__(self):
        super().__init__('field_explorer_node')

        self.declare_parameter('img_topic', '/camera_depth/image_raw')
        img_topic = self.get_parameter('img_topic').get_parameter_value().string_value
        
        # QoS for sensor data
        self.qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # QoS for map/localization data
        self.qos_map = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        # Publishers
        self.vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.vis_pub_objects = self.create_publisher(Marker, '/detected_objects', 10)
        self.vis_pub_path = self.create_publisher(MarkerArray, '/waypoint_markers', 10)

        # Nav2 action client
        self.nav_action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Subscribers
        self.sub_cam = self.create_subscription(
            Image, img_topic, self.process_camera_feed, self.qos_sensor
        )
        self.sub_lidar = self.create_subscription(
            LaserScan, '/scan', self.process_lidar, self.qos_sensor
        )
        self.sub_amcl = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.update_pose, self.qos_map
        )

        self.cv_bridge = CvBridge()
        self.curr_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.pose_received = False
        
        # Navigation tracking
        self.nav_handle = None
        self.is_navigating = False
        self.nav_start_ts = 0.0
        self.last_progress_val = 0.0
        self.stall_timestamp = 0.0
        self.abort_cause = "NONE"

        # Detection state
        self.active_target = None 
        self.obstacle_dist = None
        self.identified_items = []
        self.marker_idx = 100

        # Behavior mode
        self.operation_mode = "SEARCH"
        self.maneuver_tick = 0
        
        # Waypoints
        self.pending_targets = self._generate_search_grid()
        self.skipped_targets = [] 
        self.current_goal = None

        self.min_blob_size = 600.0
        
        # HSV color ranges for detection
        self.colors = {
            'red': [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([160, 100, 100]), np.array([180, 255, 255]))
            ],
            'green': [
                (np.array([35, 20, 20]), np.array([85, 255, 255]))
            ]
        }

        self.get_logger().info("Explorer Node Initialized.")
        self.control_timer = self.create_timer(0.1, self.run_behavior_loop)

    # generates grid waypoints for exploration
    def _generate_search_grid(self):
        pts = []
        x_range = np.arange(-1.6, 5.7, 2.0)
        y_range = np.arange(-3.7, 4.0, 2.0)
        for x in x_range:
            for y in y_range:
                pts.append((float(x), float(y)))
        return pts

    # updates robot position from AMCL
    def update_pose(self, data: PoseWithCovarianceStamped):
        p = data.pose.pose
        self.curr_pose['x'] = p.position.x
        self.curr_pose['y'] = p.position.y
        
        # quaternion to yaw conversion
        q = p.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.curr_pose['theta'] = math.atan2(siny, cosy)
        self.pose_received = True

    # processes lidar to get front obstacle distance
    def process_lidar(self, data: LaserScan):
        if not data.ranges:
            self.obstacle_dist = None
            return
        
        front_cone = data.ranges[0:25] + data.ranges[-25:]
        clean_vals = [x for x in front_cone if not math.isinf(x) and x > 0.1]
        
        if clean_vals:
            self.obstacle_dist = min(clean_vals)
        else:
            self.obstacle_dist = None

    # processes camera for color detection
    def process_camera_feed(self, data: Image):
        try:
            frame = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError:
            return

        h, w, _ = frame.shape
        
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # create masks for red and green
        mask_r = np.zeros((h, w), dtype='uint8')
        for (lower, upper) in self.colors['red']:
            mask_r = cv2.bitwise_or(mask_r, cv2.inRange(hsv_frame, lower, upper))

        mask_g = np.zeros((h, w), dtype='uint8')
        for (lower, upper) in self.colors['green']:
            mask_g = cv2.bitwise_or(mask_g, cv2.inRange(hsv_frame, lower, upper))

        # morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel)
        mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_OPEN, kernel)

        obj_red = self._analyze_contours(mask_r, w)
        obj_grn = self._analyze_contours(mask_g, w)

        potential_targets = []
        if obj_red:
            potential_targets.append({'tag': 'hydrant', 'data': obj_red})
        if obj_grn:
            potential_targets.append({'tag': 'trash_can', 'data': obj_grn})

        cv2.imshow("Robot Vision", frame)
        cv2.waitKey(1)

        if not potential_targets:
            self.active_target = None
            return

        # pick largest object
        winner = max(potential_targets, key=lambda x: x['data']['size'])
        self.active_target = {
            'type': winner['tag'],
            'err_x': winner['data']['err_x'],
            'size': winner['data']['size']
        }

    # finds largest contour and calculates centroid
    def _analyze_contours(self, mask, img_width):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        biggest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(biggest)
        
        if area < self.min_blob_size:
            return None
        
        M = cv2.moments(biggest)
        if M["m00"] == 0: return None
        cx = int(M["m10"] / M["m00"])
        
        center_x = img_width / 2.0
        err_x = (cx - center_x) / center_x
        
        return {'cx': cx, 'err_x': err_x, 'size': area}

    # sends navigation goal to Nav2
    def dispatch_goal(self, coords):
        if not self.nav_action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("Nav2 server offline.")
            return False

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = coords[0]
        goal.pose.pose.position.y = coords[1]
        goal.pose.pose.orientation.w = 1.0

        self.get_logger().info(f"Navigating to: {coords}")
        
        self.abort_cause = "NONE"
        self.nav_start_ts = time.time()
        self.stall_timestamp = time.time()
        self.last_progress_val = 999.0

        future = self.nav_action_client.send_goal_async(
            goal, feedback_callback=self._nav_feedback_handler
        )
        future.add_done_callback(self._goal_accepted_handler)
        return True

    # checks navigation progress and detects stalls
    def _nav_feedback_handler(self, msg):
        remain = msg.feedback.distance_remaining
        now = time.time()

        if (now - self.nav_start_ts) < 3.0:
            return

        if remain < 0.35 and self.is_navigating and self.abort_cause == "NONE":
            self.get_logger().info("Target within tolerance.")
            self.abort_cause = "ARRIVED"
            self.halt_navigation()
            return

        delta_dist = self.last_progress_val - remain
        if delta_dist < 0.05:
            if (now - self.stall_timestamp) > 15.0:
                self.get_logger().warn("Robot is stuck.")
                self.abort_cause = "BLOCKED"
                self.halt_navigation()
        else:
            self.stall_timestamp = now
            self.last_progress_val = remain

    def _goal_accepted_handler(self, future):
        handle = future.result()
        if not handle or not handle.accepted:
            self.is_navigating = False
            return
        
        self.nav_handle = handle
        self.is_navigating = True
        res_future = handle.get_result_async()
        res_future.add_done_callback(self._nav_finished_handler)

    # handles navigation result
    def _nav_finished_handler(self, future):
        status = future.result().status
        self.is_navigating = False
        self.nav_handle = None
        
        if status == GoalStatus.STATUS_SUCCEEDED:
            if self.current_goal in self.pending_targets:
                self.pending_targets.remove(self.current_goal)
            self.skipped_targets.clear()
            
        elif status == GoalStatus.STATUS_CANCELED:
            if self.abort_cause == "ARRIVED":
                if self.current_goal in self.pending_targets:
                    self.pending_targets.remove(self.current_goal)
                self.skipped_targets.clear()
            elif self.abort_cause == "BLOCKED":
                if self.current_goal: self.skipped_targets.append(self.current_goal)
            else:
                pass
                
        elif status == GoalStatus.STATUS_ABORTED:
             if self.current_goal: self.skipped_targets.append(self.current_goal)

        self.current_goal = None
        if self.operation_mode == "SEARCH":
            self.operation_mode = "IDLE"

    # stops robot and cancels navigation
    def halt_navigation(self):
        if self.nav_handle and self.is_navigating:
            self.nav_handle.cancel_goal_async()
        self.is_navigating = False
        self.vel_publisher.publish(Twist())

    # picks closest unvisited waypoint
    def _select_next_waypoint(self):
        if not self.pending_targets: return None
        
        valid_opts = [t for t in self.pending_targets if t not in self.skipped_targets]
        
        if not valid_opts:
            self.get_logger().info("All paths blocked. Retrying skipped targets.")
            self.skipped_targets.clear()
            valid_opts = self.pending_targets

        best_pt = None
        min_d = float('inf')
        
        rx, ry = self.curr_pose['x'], self.curr_pose['y']
        
        for pt in valid_opts:
            d = math.hypot(pt[0] - rx, pt[1] - ry)
            if d < min_d:
                min_d = d
                best_pt = pt
        
        return best_pt

    # estimates object position on map
    def _calculate_obj_location(self, detection, lidar_dist):
        if not self.pose_received: return None, None
        
        dist = lidar_dist if lidar_dist else 0.8
        radius_offset = 0.2
        final_dist = dist + radius_offset
        
        fov_h = 1.08
        angle_offset = -detection['err_x'] * (fov_h / 2.0)
        
        global_yaw = self.curr_pose['theta'] + angle_offset
        
        obj_x = self.curr_pose['x'] + final_dist * math.cos(global_yaw)
        obj_y = self.curr_pose['y'] + final_dist * math.sin(global_yaw)
        
        return obj_x, obj_y

    # checks if object was already found
    def _is_new_object(self, obj_type, x, y):
        for item in self.identified_items:
            if item['type'] == obj_type:
                dist = math.hypot(item['x'] - x, item['y'] - y)
                if dist < 1.0:
                    return False
        return True

    # saves object and publishes marker
    def _register_object(self, obj_type, x, y):
        self.identified_items.append({'type': obj_type, 'x': x, 'y': y})
        self.get_logger().info(f"Registered {obj_type} at ({x:.2f}, {y:.2f})")
        
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "found_items"
        m.id = self.marker_idx
        self.marker_idx += 1
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.2
        m.scale.x = 0.2
        m.scale.y = 0.2
        m.scale.z = 0.4
        m.color.a = 1.0
        
        m.color.r = 0.0
        m.color.g = 0.0
        m.color.b = 0.0

        if obj_type == 'hydrant':
            m.color.r = 1.0
        else:
            m.color.g = 1.0
            
        self.vis_pub_objects.publish(m)

    # main control loop - state machine
    def run_behavior_loop(self):
        self._publish_path_markers()
        
        det = self.active_target
        front_dist = self.obstacle_dist
        
        # interrupt search if object spotted
        if det and self.operation_mode == "SEARCH":
            if front_dist and self.pose_received:
                est_x, est_y = self._calculate_obj_location(det, front_dist)
                if self._is_new_object(det['type'], est_x, est_y):
                    self.get_logger().info(f"Visual contact: {det['type']}. Engaging.")
                    self.abort_cause = "INTERRUPT"
                    self.halt_navigation()
                    self.operation_mode = "ALIGN"
                    return
        
        move_cmd = Twist()
        
        if self.operation_mode == "IDLE":
            if not self.pose_received: return
            
            if self.pending_targets:
                nxt = self._select_next_waypoint()
                if nxt:
                    self.current_goal = nxt
                    if self.dispatch_goal(nxt):
                        self.operation_mode = "SEARCH"
            else:
                self.get_logger().info("Mission Complete.")
                self.destroy_node()

        elif self.operation_mode == "SEARCH":
            pass

        elif self.operation_mode == "ALIGN":
            if not det:
                self.operation_mode = "IDLE"
            else:
                if abs(det['err_x']) < 0.1:
                    self.operation_mode = "TRACK"
                else:
                    move_cmd.angular.z = -0.5 * det['err_x']
        
        elif self.operation_mode == "TRACK":
            if not det:
                self.operation_mode = "IDLE"
            else:
                move_cmd.angular.z = -0.4 * det['err_x']
                move_cmd.linear.x = 0.15
                
                safe_zone = 0.7
                if front_dist and front_dist < safe_zone:
                    est_x, est_y = self._calculate_obj_location(det, front_dist)
                    if self._is_new_object(det['type'], est_x, est_y):
                        self._register_object(det['type'], est_x, est_y)
                    
                    self.operation_mode = "RETREAT"
                    self.maneuver_tick = 20
        
        elif self.operation_mode == "RETREAT":
            if self.maneuver_tick > 0:
                move_cmd.linear.x = -0.15
                self.maneuver_tick -= 1
            else:
                self.operation_mode = "ROTATE"
                self.maneuver_tick = 35
        
        elif self.operation_mode == "ROTATE":
            if self.maneuver_tick > 0:
                move_cmd.angular.z = 0.6
                self.maneuver_tick -= 1
            else:
                self.operation_mode = "IDLE"

        self.vel_publisher.publish(move_cmd)

    # shows remaining waypoints in rviz
    def _publish_path_markers(self):
        arr = MarkerArray()
        d_mk = Marker()
        d_mk.action = Marker.DELETEALL
        arr.markers.append(d_mk)
        
        for i, pt in enumerate(self.pending_targets):
            m = Marker()
            m.header.frame_id = 'map'
            m.ns = 'waypoints'
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = pt[0]
            m.pose.position.y = pt[1]
            m.scale.x = 0.1; m.scale.y = 0.1; m.scale.z = 0.1
            m.color.b = 1.0; m.color.a = 0.6
            arr.markers.append(m)
        self.vis_pub_path.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = FieldExplorerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
