# COMP4034 Robotics - Final Coursework
## Group 30

### Team Members
- Mohammed Azeez
- Muhammed Mehboob

## Project Overview
This project implements an autonomous field exploration and object detection system for the TurtleBot3 robot. The robot autonomously navigates through a search grid pattern, detects fire hydrants (red) and trash cans (green) using computer vision, and marks their locations on the map. The system uses ROS2 Navigation Stack (Nav2) for path planning and AMCL for localization.

### Project Structure
```
.
├── src/
│   ├── nav_package/
│   │   └── nav_package/
│   │       └── nav_node.py      # Main exploration and detection node
│   └── maps/
│       ├── arena_map.yaml       # Arena environment map
│       ├── cwk_map.yaml         # Coursework map
│       └── my_map.yaml          # Custom map
├── demonstration.mp4            # Video demonstration
└── README.md
```

### Features
1. Autonomous Grid Exploration:
   - Waypoint-based search pattern covering the environment
   - Nearest-neighbor waypoint selection for efficient coverage
   - Stall detection and recovery mechanism
   - Dynamic goal cancellation when objects are detected

2. Object Detection:
   - Real-time color detection using HSV color space
   - Fire hydrant (red) and trash can (green) identification
   - Contour analysis with noise filtering
   - Object position estimation using LiDAR and robot pose

3. Behavior State Machine:
   - IDLE: Selecting next waypoint
   - SEARCH: Navigating while scanning for objects
   - ALIGN: Centering detected object in camera view
   - TRACK: Approaching the object
   - RETREAT: Backing away after registration
   - ROTATE: Turning to resume exploration

4. Visualization:
   - RViz markers for detected objects (cylinders)
   - Waypoint visualization for remaining search points
   - Live camera feed display

### Team Contributions

#### Mohammed Azeez
- Implemented the `FieldExplorerNode` class
- Developed the state machine behavior logic
- Integrated Nav2 action client for navigation
- Implemented object tracking and approach behavior
- Handled LiDAR-based distance sensing

#### Muhammed Mehboob
- Developed HSV color detection pipeline
- Implemented contour analysis and filtering
- Created object position estimation algorithm
- Handled AMCL localization integration
- Implemented RViz marker visualization

### Technical Implementation

#### Navigation
- **Nav2 Action Client**: Sends goals to the navigation stack for autonomous path planning
- **AMCL Localization**: Receives robot pose updates for position tracking
- **Stall Detection**: Monitors navigation progress and handles stuck situations

#### Computer Vision
- **HSV Filtering**: Robust color detection independent of lighting variations
- **Morphological Operations**: Removes noise from detection masks
- **Centroid Calculation**: Determines object position in camera frame

#### Object Localization
- **Coordinate Transformation**: Converts camera detections to map coordinates
- **Duplicate Detection**: Prevents registering the same object multiple times
- **LiDAR Integration**: Uses front-cone distance for accurate positioning



### Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Search Grid | -1.6 to 5.7 (X), -3.7 to 4.0 (Y) | Area coverage range |
| Grid Spacing | 2.0 meters | Distance between waypoints |
| Red HSV Range | [0-10, 100-255, 100-255] | Fire hydrant detection |
| Green HSV Range | [35-85, 20-255, 20-255] | Trash can detection |
| Min Contour Size | 600 pixels | Noise filtering threshold |
| Arrival Tolerance | 0.35 meters | Goal reached threshold |
| Safe Zone Distance | 0.7 meters | Object approach distance |
| Control Loop Rate | 10 Hz | Main loop frequency |

### Design Decisions

1. **State Machine Architecture**:
   - Clear separation of behaviors for predictable robot actions
   - Easy to debug and extend with new states

2. **Nearest-Neighbor Selection**:
   - Reduces travel time between waypoints
   - Efficient coverage of the search area

3. **Dual Color Detection**:
   - Red range split into two to handle HSV wraparound
   - Wide green range for varying lighting conditions

4. **LiDAR-Based Positioning**:
   - More accurate than depth camera for distance estimation
   - Front-cone filtering for relevant obstacle detection

### Testing
The system was tested in Gazebo simulation:
- Verified complete grid coverage
- Confirmed detection of both red and green objects
- Validated duplicate prevention mechanism
- Tested stall recovery behavior
- Checked smooth state transitions

### Video Demonstration
A video demonstration of the system is included as `demonstration.mp4` showing:
- Autonomous exploration behavior
- Object detection and approach
- Marker placement in RViz
- Mission completion

### Future Improvements
1. Multiple object type detection (beyond red/green)
2. Dynamic waypoint generation based on map size
3. Improved object approach using visual servoing
4. Adaptive color thresholds for different environments
5. Multi-robot coordination for faster coverage

### References
- ROS2 Navigation2 Documentation
- OpenCV Python Tutorials
- TurtleBot3 Documentation
- AMCL Localization Package
