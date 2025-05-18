import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib import colors
import matplotlib.patches as mpatches
import math
from typing import List, Tuple, Dict, Any, Optional, Union
import csv
import os
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional, Union



def prim_traj_gen(waypoints: List[Tuple[float, float, float]], 
            time_window: Tuple[float, float], 
            samples_per_segment: int = 15) -> List[Tuple[float, float, float, float]]:
    """
    Generate time-sequenced trajectory from waypoints using a time window.
    
    Args:
        waypoints: List of 3D waypoints (x, y, z)
        time_window: Tuple of (start_time, end_time) for entire mission
        samples_per_segment: Number of points to generate per trajectory segment
        
    Returns:
        List of (x, y, z, t) tuples representing the trajectory
    """
    if len(waypoints) < 2:
        raise ValueError("At least two waypoints required")
    
    t_start, t_end = time_window
    num_segments = len(waypoints) - 1
    time_per_segment = (t_end - t_start) / num_segments
    
    trajectory = []
    for i in range(num_segments):
        wp_start = waypoints[i]
        wp_end = waypoints[i+1]
        
        # Generate timestamps for this segment
        seg_start_time = t_start + i * time_per_segment
        seg_end_time = t_start + (i+1) * time_per_segment
        times = np.linspace(seg_start_time, seg_end_time, samples_per_segment)
        
        # Linear interpolation
        for t in times:
            alpha = (t - seg_start_time) / (seg_end_time - seg_start_time)
            x = wp_start[0] + alpha * (wp_end[0] - wp_start[0])
            y = wp_start[1] + alpha * (wp_end[1] - wp_start[1])
            z = wp_start[2] + alpha * (wp_end[2] - wp_start[2])
            trajectory.append((x, y, z, t))
    
    return trajectory

def traj_gen(waypoints: List[Tuple[float, float, float]], 
             time_steps: List[float]) -> List[Tuple[float, float, float, float]]:
    """
    Generate a time-sequenced trajectory using linear interpolation between waypoints.
    
    Args:
        waypoints: List of 3D waypoints (x, y, z)
        time_steps: List of timestamps corresponding to each waypoint
        
    Returns:
        A list of (x, y, z, t) tuples representing the trajectory
        
    Raises:
        ValueError: If lengths of waypoints and time_steps don't match or if fewer than 2 waypoints
    """
    if len(waypoints) != len(time_steps):
        raise ValueError("Lengths of waypoints and time_steps must match")
    
    if len(waypoints) < 2:
        raise ValueError("At least 2 waypoints are required to generate a trajectory")
    
    # Convert to numpy arrays for easier manipulation
    waypoints_array = np.array(waypoints)  # shape: (n, 3)
    time_steps_array = np.array(time_steps)  # shape: (n,)
    
    # Create full trajectory with small time increments for smoother simulation
    # Use the smallest reasonable time increment (0.1 sec)
    t_start = time_steps_array[0]
    t_end = time_steps_array[-1]
    t_increment = 0.1  # 0.1 second increment for smooth animation
    t_full = np.arange(t_start, t_end + t_increment, t_increment)
    
    # Initialize arrays for the full trajectory
    x_full = np.zeros_like(t_full)
    y_full = np.zeros_like(t_full)
    z_full = np.zeros_like(t_full)
    
    # For each time point, find the corresponding position by linear interpolation
    for i, t in enumerate(t_full):
        # Find the segment where this time point falls
        if t <= time_steps_array[0]:
            # Before first waypoint, use first waypoint
            x_full[i], y_full[i], z_full[i] = waypoints_array[0]
        elif t >= time_steps_array[-1]:
            # After last waypoint, use last waypoint
            x_full[i], y_full[i], z_full[i] = waypoints_array[-1]
        else:
            # Find the two waypoints to interpolate between
            idx = np.searchsorted(time_steps_array, t) - 1
            t1, t2 = time_steps_array[idx], time_steps_array[idx + 1]
            w1, w2 = waypoints_array[idx], waypoints_array[idx + 1]
            
            # Linear interpolation formula: p = p1 + (p2-p1)*(t-t1)/(t2-t1)
            ratio = (t - t1) / (t2 - t1)
            x_full[i] = w1[0] + (w2[0] - w1[0]) * ratio
            y_full[i] = w1[1] + (w2[1] - w1[1]) * ratio
            z_full[i] = w1[2] + (w2[2] - w1[2]) * ratio
    
    # Combine into a list of tuples (x, y, z, t)
    trajectory = [(x_full[i], y_full[i], z_full[i], t_full[i]) for i in range(len(t_full))]
    return trajectory

def generate_primary_trajectory(waypoints: List[Tuple[float, float, float]], 
                               time_window: Tuple[float, float]) -> List[Tuple[float, float, float, float]]:
    """
    Generate primary drone trajectory with automatic time distribution
    
    Args:
        waypoints: List of 3D waypoints
        time_window: Mission time window (start, end)
        
    Example:
        generate_primary_trajectory(
            [(0,0,0), (10,10,10), (20,20,20)],
            (0, 20)  # 20 second mission
        )
    """
    return prim_traj_gen(waypoints, time_window)


def generate_secondary_trajectories(secondary_data: List[Dict[str, Any]]) -> Dict[str, List[Tuple[float, float, float, float]]]:
    """
    Generate trajectories for all secondary drones.
    
    Args:
        secondary_data: List of dictionaries, each containing:
            - 'id': drone identifier
            - 'waypoints': list of (x, y, z) tuples
            - 'time_steps': list of timestamps
            
    Returns:
        Dictionary mapping drone IDs to their trajectories
    """
    secondary_trajectories = {}
    
    for drone_data in secondary_data:
        drone_id = drone_data['id']
        waypoints = drone_data['waypoints']
        time_steps = drone_data['time_steps']
        
        try:
            trajectory = traj_gen(waypoints, time_steps)
            secondary_trajectories[drone_id] = trajectory
        except ValueError as e:
            print(f"Error generating trajectory for secondary drone {drone_id}: {e}")
            # Skip this drone
            continue
            
    return secondary_trajectories

def get_position_at_time(trajectory: List[Tuple[float, float, float, float]], 
                        time: float) -> Optional[Tuple[float, float, float]]:
    """
    Get the position (x, y, z) of a drone at a specific time from its trajectory.
    
    Args:
        trajectory: List of (x, y, z, t) tuples
        time: The time at which to determine the position
        
    Returns:
        (x, y, z) tuple representing the position, or None if time is out of range
    """
    # Check if time is out of range
    if time < trajectory[0][3] or time > trajectory[-1][3]:
        return None
    
    # Find the closest time points in the trajectory
    for i in range(len(trajectory) - 1):
        t1, t2 = trajectory[i][3], trajectory[i+1][3]
        
        if t1 <= time <= t2:
            # Found the right segment, interpolate
            if t1 == t2:  # Avoid division by zero
                return (trajectory[i][0], trajectory[i][1], trajectory[i][2])
            
            ratio = (time - t1) / (t2 - t1)
            x = trajectory[i][0] + ratio * (trajectory[i+1][0] - trajectory[i][0])
            y = trajectory[i][1] + ratio * (trajectory[i+1][1] - trajectory[i][1])
            z = trajectory[i][2] + ratio * (trajectory[i+1][2] - trajectory[i][2])
            
            return (x, y, z)
    
    # This shouldn't happen if the time checks at the beginning are correct
    return None

def calculate_distance(pos1: Tuple[float, float, float], 
                       pos2: Tuple[float, float, float]) -> float:
    """
    Calculate Euclidean distance between two 3D points.
    
    Args:
        pos1: First position (x1, y1, z1)
        pos2: Second position (x2, y2, z2)
        
    Returns:
        Euclidean distance
    """
    return math.sqrt((pos1[0] - pos2[0])**2 + 
                     (pos1[1] - pos2[1])**2 + 
                     (pos1[2] - pos2[2])**2)

def check_intersection(primary_trajectory: List[Tuple[float, float, float, float]],
                      secondary_trajectory: List[Tuple[float, float, float, float]]) -> List[Dict[str, Any]]:
    """
    Check for intersections between primary and secondary time intervals.
    
    Args:
        primary_trajectory: Primary drone's trajectory as list of (x, y, z, t) tuples
        secondary_trajectory: Secondary drone's trajectory as list of (x, y, z, t) tuples
        
    Returns:
        List of dictionaries with intersection details:
            - 'position': (x, y, z) tuple of time intersection region
            - 'time': time at this position
    """
    # Extract time ranges
    primary_start, primary_end = primary_trajectory[0][3], primary_trajectory[-1][3]
    secondary_start, secondary_end = secondary_trajectory[0][3], secondary_trajectory[-1][3]
    
    # Find overlapping time range
    start_time = max(primary_start, secondary_start)
    end_time = min(primary_end, secondary_end)
    
    if start_time >= end_time:
        # No temporal overlap
        return []
    
    # Create time samples within overlapping range
    # Using smaller step for higher precision
    time_step = 0.1  # 0.1 second
    check_times = np.arange(start_time, end_time + time_step, time_step)
    
    overlap = []
    
    for t in check_times:
        primary_pos = get_position_at_time(primary_trajectory, t)
        secondary_pos = get_position_at_time(secondary_trajectory, t)
        
        if primary_pos is None or secondary_pos is None:
            continue
        
        # We'll collect possible positions at this stage
        # Actual conflict determination happens later with buffer distance
        overlap.append({
            'time': t,
            'primary_position': primary_pos,
            'secondary_position': secondary_pos
        })
    
    return overlap

def determine_conflict(primary_trajectory: List[Tuple[float, float, float, float]],
                      secondary_trajectory: List[Tuple[float, float, float, float]],
                      time_overlap_points: List[Dict[str, Any]],
                      buffer_distance: float,
                      secondary_id: str) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Determine if there's a conflict between primary and secondary trajectories.
    
    Args:
        primary_trajectory: Primary drone's trajectory
        secondary_trajectory: Secondary drone's trajectory
        time_overlap_points: List of potential intersection points from check_intersection
        buffer_distance: Minimum safe distance between drones
        secondary_id: ID of the secondary drone
        
    Returns:
        Tuple of (conflict_exists, conflict_details) where:
            - conflict_exists: Boolean indicating if conflict exists
            - conflict_details: List of dictionaries with conflict details
    """
    conflicts = []
    
    for point in time_overlap_points:
        primary_pos = point['primary_position']
        secondary_pos = point['secondary_position']
        time = point['time']
        
        distance = calculate_distance(primary_pos, secondary_pos)
        
        if distance < buffer_distance:
            conflicts.append({
                'time': time,
                'primary_position': primary_pos,
                'secondary_position': secondary_pos,
                'distance': distance,
                'secondary_id': secondary_id
            })
    
    return len(conflicts) > 0, conflicts


class UAVDeconflictionVisualizer:
    def __init__(self, primary_trajectory, secondary_trajectories, 
                buffer_distance=1.0, sim_speed=1.0):
        """
        Initialize the UAV deconfliction system visualizer.
        
        Args:
            primary_trajectory: The trajectory of the primary UAV
            secondary_trajectories: Dictionary mapping secondary UAV IDs to their trajectories
            buffer_distance: Minimum safe distance between UAVs
            sim_speed: Simulation speed multiplier
        """
        self.primary_trajectory = primary_trajectory
        self.secondary_trajectories = secondary_trajectories
        self.buffer_distance = buffer_distance
        self.sim_speed = sim_speed
        
        # Calculate time bounds for the simulation
        self.t_start = primary_trajectory[0][3]
        self.t_end = primary_trajectory[-1][3]
        
        for traj in secondary_trajectories.values():
            self.t_start = min(self.t_start, traj[0][3])
            self.t_end = max(self.t_end, traj[-1][3])
        
        # Find spatial bounds
        self.x_min, self.x_max = float('inf'), float('-inf')
        self.y_min, self.y_max = float('inf'), float('-inf')
        self.z_min, self.z_max = float('inf'), float('-inf')
        
        # Include primary trajectory in bounds calculation
        for x, y, z, _ in primary_trajectory:
            self.x_min = min(self.x_min, x)
            self.x_max = max(self.x_max, x)
            self.y_min = min(self.y_min, y)
            self.y_max = max(self.y_max, y)
            self.z_min = min(self.z_min, z)
            self.z_max = max(self.z_max, z)
        
        # Include secondary trajectories in bounds calculation
        for traj in secondary_trajectories.values():
            for x, y, z, _ in traj:
                self.x_min = min(self.x_min, x)
                self.x_max = max(self.x_max, x)
                self.y_min = min(self.y_min, y)
                self.y_max = max(self.y_max, y)
                self.z_min = min(self.z_min, z)
                self.z_max = max(self.z_max, z)
        
        # Add a margin to bounds
        margin = max(
            0.1 * (self.x_max - self.x_min),
            0.1 * (self.y_max - self.y_min),
            0.1 * (self.z_max - self.z_min),
            2.0  # Minimum margin
        )
        self.x_min -= margin
        self.x_max += margin
        self.y_min -= margin
        self.y_max += margin
        self.z_min -= margin
        self.z_max += margin
        
        # Detect conflicts before starting animation
        self.conflicts = {}
        self.all_conflict_details = []
        
        for secondary_id, secondary_traj in self.secondary_trajectories.items():
            intersections = check_intersection(primary_trajectory, secondary_traj)
            has_conflict, conflict_details = determine_conflict(
                primary_trajectory, secondary_traj, 
                intersections, buffer_distance, secondary_id
            )
            
            if has_conflict:
                self.conflicts[secondary_id] = conflict_details
                self.all_conflict_details.extend(conflict_details)
        
        # Sort conflicts by time for display
        self.all_conflict_details.sort(key=lambda x: x['time'])
        
        # Setup figure and 3D axis
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initialize visualization elements
        self.primary_marker = None
        self.primary_trail = None
        self.secondary_markers = {}
        self.secondary_trails = {}
        self.conflict_markers = []
        self.text_display = None
        
        # Initialize colors for secondary drones
        color_options = list(colors.TABLEAU_COLORS)
        self.secondary_colors = {}
        for i, drone_id in enumerate(self.secondary_trajectories.keys()):
            self.secondary_colors[drone_id] = color_options[i % len(color_options)]
        
        # Current simulation time
        self.current_time = self.t_start
        
    def setup_plot(self):
        """Set up the initial state of the 3D plot."""
        self.ax.set_xlim([self.x_min, self.x_max])
        self.ax.set_ylim([self.y_min, self.y_max])
        self.ax.set_zlim([self.z_min, self.z_max])
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('UAV Deconfliction System')
        
        # Initialize primary drone marker and trail
        primary_pos = get_position_at_time(self.primary_trajectory, self.t_start)
        if primary_pos:
            self.primary_marker, = self.ax.plot([primary_pos[0]], [primary_pos[1]], [primary_pos[2]], 
                                           'ro', markersize=10, label='Primary UAV')
            self.primary_trail, = self.ax.plot([], [], [], 'r-', linewidth=2)
        
        # Initialize secondary drone markers and trails
        for drone_id, trajectory in self.secondary_trajectories.items():
            # Find first valid position (at or after t_start)
            first_valid_time = max(trajectory[0][3], self.t_start)
            drone_pos = get_position_at_time(trajectory, first_valid_time)  # <-- Use first valid time
            
            if drone_pos:
                color = self.secondary_colors[drone_id]
                marker, = self.ax.plot([drone_pos[0]], [drone_pos[1]], [drone_pos[2]], 
                                    'o', color=color, markersize=8, 
                                    label=f'Secondary UAV {drone_id}')
                trail, = self.ax.plot([], [], [], '-', color=color, linewidth=1)
                self.secondary_markers[drone_id] = marker
                self.secondary_trails[drone_id] = trail
        
        # Text display for conflict information
        self.text_display = self.ax.text2D(
            0.02,  # X position (2% from left)
            0.85,  # Lowered Y position (15% from top)
            '',
            transform=self.ax.transAxes,
            backgroundcolor='white',
            verticalalignment='top',  # Anchor text to top
            wrap=True  # Enable text wrapping
        )
        
        # Increase figure margins
        self.fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)

        
        # Add legend
        legend_handles = []
        if self.primary_marker:
            legend_handles.append(self.primary_marker)
        for drone_id, marker in self.secondary_markers.items():
            legend_handles.append(marker)

        # Add legend once with all handles
        self.ax.legend(handles=legend_handles, loc='upper right')
        
        # Add timestamp display
        self.time_display = self.ax.text2D(0.02, 0.02, '', transform=self.ax.transAxes)
        
        return [self.primary_marker, self.primary_trail] + \
               list(self.secondary_markers.values()) + \
               list(self.secondary_trails.values()) + \
               [self.text_display, self.time_display]
    
    def update(self, frame):
        """
        Update function for the animation.
        
        Args:
            frame: Frame number (unused, we use self.current_time instead)
            
        Returns:
            List of artists that were modified
        """
        # Update current time based on simulation speed
        self.current_time = self.t_start + (frame * 0.1 * self.sim_speed)
        
        # Update time display
        self.time_display.set_text(f'Time: {self.current_time:.1f}s')
        
        # Update primary drone position and trail
        primary_history = [get_position_at_time(self.primary_trajectory, 
                                              self.t_start + (i * 0.1 * self.sim_speed)) 
                         for i in range(frame + 1)]
        primary_history = [p for p in primary_history if p is not None]
        
        if primary_history:
            # Update position
            current_pos = primary_history[-1]
            self.primary_marker.set_data([current_pos[0]], [current_pos[1]])
            self.primary_marker.set_3d_properties([current_pos[2]])
            
            # Update trail
            x_trail = [p[0] for p in primary_history]
            y_trail = [p[1] for p in primary_history]
            z_trail = [p[2] for p in primary_history]
            self.primary_trail.set_data(x_trail, y_trail)
            self.primary_trail.set_3d_properties(z_trail)
        
        # Update secondary drone positions and trails
        for drone_id, trajectory in self.secondary_trajectories.items():
            secondary_history = [get_position_at_time(trajectory, 
                                                   self.t_start + (i * 0.1 * self.sim_speed)) 
                              for i in range(frame + 1)]
            secondary_history = [p for p in secondary_history if p is not None]
            
            if secondary_history and drone_id in self.secondary_markers:
                # Update position
                current_pos = secondary_history[-1]
                self.secondary_markers[drone_id].set_data([current_pos[0]], [current_pos[1]])
                self.secondary_markers[drone_id].set_3d_properties([current_pos[2]])
                
                # Update trail
                x_trail = [p[0] for p in secondary_history]
                y_trail = [p[1] for p in secondary_history]
                z_trail = [p[2] for p in secondary_history]
                self.secondary_trails[drone_id].set_data(x_trail, y_trail)
                self.secondary_trails[drone_id].set_3d_properties(z_trail)
        
        # Check for active conflicts at current time
        active_conflicts = []
        for conflict in self.all_conflict_details:
            if abs(conflict['time'] - self.current_time) < 0.15:  # Within 0.15s window
                active_conflicts.append(conflict)
        
        # Update conflict markers
        for marker in self.conflict_markers:
            marker.remove()
        self.conflict_markers = []
        
        for conflict in active_conflicts:
            # Add a sphere to mark the conflict location
            conflict_pos = conflict['primary_position']
            conflict_marker = self.ax.plot([conflict_pos[0]], [conflict_pos[1]], [conflict_pos[2]], 
                                      'yo', markersize=20, alpha=0.5)[0]
            self.conflict_markers.append(conflict_marker)
            
            # Change color of involved drones to indicate conflict
            secondary_id = conflict['secondary_id']
            if secondary_id in self.secondary_markers:
                self.secondary_markers[secondary_id].set_color('yellow')
                self.primary_marker.set_color('yellow')
            
        # Reset drone colors if not in conflict
        if not active_conflicts:
            self.primary_marker.set_color('red')
            for drone_id, marker in self.secondary_markers.items():
                marker.set_color(self.secondary_colors[drone_id])
        
        # Update text display with conflict information
        if active_conflicts:
            conflict_text = "CONFLICT DETECTED!\n"
            for i, conflict in enumerate(active_conflicts):
                # Truncate long location values
                loc = conflict['primary_position']
                loc_str = f"({loc[0]:.1f}, {loc[1]:.1f}, {loc[2]:.1f})"
                
                conflict_text += (
                    f"Conflict {i+1}: UAV {conflict['secondary_id']}\n"
                    f"Time: {conflict['time']:.1f}s | Dist: {conflict['distance']:.1f}m\n"
                    f"Loc: {loc_str}\n"
                )
                
            self.text_display.set_text(conflict_text)
            # Auto-adjust font size based on number of conflicts
            self.text_display.set_fontsize(8 if len(active_conflicts) > 2 else 10)
        else:
            # If there are any conflicts in the simulation but not active now
            if self.all_conflict_details:
                next_conflicts = [c for c in self.all_conflict_details if c['time'] > self.current_time]
                if next_conflicts:
                    next_conflict = next_conflicts[0]
                    time_to_conflict = next_conflict['time'] - self.current_time
                    self.text_display.set_text(f"Next conflict in {time_to_conflict:.1f}s with Secondary UAV {next_conflict['secondary_id']}")
                    self.text_display.set_backgroundcolor('white')
                else:
                    past_conflicts = [c for c in self.all_conflict_details if c['time'] <= self.current_time]
                    if past_conflicts:
                        self.text_display.set_text(f"All {len(self.all_conflict_details)} conflicts passed")
                        self.text_display.set_backgroundcolor('white')
            else:
                self.text_display.set_text("No conflicts detected in simulation")
                self.text_display.set_backgroundcolor('white')
        
        return [self.primary_marker, self.primary_trail] + \
               list(self.secondary_markers.values()) + \
               list(self.secondary_trails.values()) + \
               self.conflict_markers + \
               [self.text_display, self.time_display]
    
    def animate(self):
        """Create and run the animation."""
        num_frames = int((self.t_end - self.t_start) / (0.1 * self.sim_speed)) + 1
        
        ani = FuncAnimation(
            self.fig, self.update, frames=num_frames,
            init_func=self.setup_plot, blit=True, interval=100,
            repeat=True
        )
        
        plt.tight_layout()
        plt.show()
        
        return ani



def save_collision_data(collisions: List[Dict[str, Any]], 
                       file_path: str, 
                       file_format: str ,
                       primary_id: str = "Primary") -> bool:
    """
    Save collision data to a file in the specified format.
    
    Args:
        collisions: List of collision dictionaries containing collision details
        file_path: Path to save the output file
        file_format: Format to save the file (TEXT or CSV)
        primary_id: ID to use for the primary drone
        
    Returns:
        True if successful, False otherwise
        
    Each collision dictionary should contain:
        - 'time': Time of collision
        - 'primary_position': (x, y, z) position of primary drone
        - 'secondary_position': (x, y, z) position of secondary drone
        - 'distance': Distance between drones at collision
        - 'secondary_id': ID of the secondary drone involved
    """
    try:
        if file_format == "text":
            with open(file_path, 'w') as f:
                if not collisions:
                    f.write("No collisions detected in the simulation.\n")
                else:
                    f.write(f"UAV Deconfliction System - Collision Report\n")
                    f.write(f"Total collisions detected: {len(collisions)}\n")
                    f.write(f"{'-' * 60}\n\n")
                    
                    for i, collision in enumerate(collisions, 1):
                        f.write(f"Collision #{i}:\n")
                        f.write(f"  Time: {collision['time']:.2f}s\n")
                        f.write(f"  Drones involved: {primary_id} and {collision['secondary_id']}\n")
                        f.write(f"  Distance: {collision['distance']:.2f}m\n")
                        f.write(f"  Location (Primary): ({collision['primary_position'][0]:.2f}, "
                                f"{collision['primary_position'][1]:.2f}, "
                                f"{collision['primary_position'][2]:.2f})\n")
                        f.write(f"  Location (Secondary): ({collision['secondary_position'][0]:.2f}, "
                                f"{collision['secondary_position'][1]:.2f}, "
                                f"{collision['secondary_position'][2]:.2f})\n")
                        f.write(f"\n")
                        
                print(f"Collision data saved to text file: {file_path}")
                
        elif file_format == "csv":
            with open(file_path, 'w', newline='') as f:
                csv_writer = csv.writer(f)
                
                # Write headers
                headers = [
                    "Collision_ID", "Timestamp", 
                    "Primary_Drone_ID", "Secondary_Drone_ID",
                    "Distance", 
                    "Primary_Location_X", "Primary_Location_Y", "Primary_Location_Z",
                    "Secondary_Location_X", "Secondary_Location_Y", "Secondary_Location_Z"
                ]
                csv_writer.writerow(headers)
                
                if not collisions:
                    csv_writer.writerow(["No collisions detected"] + [""] * (len(headers) - 1))
                else:
                    for i, collision in enumerate(collisions, 1):
                        row = [
                            i,  # Collision_ID
                            f"{collision['time']:.2f}",  # Timestamp
                            primary_id,  # Primary_Drone_ID
                            collision['secondary_id'],  # Secondary_Drone_ID
                            f"{collision['distance']:.2f}",  # Distance
                            f"{collision['primary_position'][0]:.2f}",  # Primary_Location_X
                            f"{collision['primary_position'][1]:.2f}",  # Primary_Location_Y
                            f"{collision['primary_position'][2]:.2f}",  # Primary_Location_Z
                            f"{collision['secondary_position'][0]:.2f}",  # Secondary_Location_X
                            f"{collision['secondary_position'][1]:.2f}",  # Secondary_Location_Y
                            f"{collision['secondary_position'][2]:.2f}"   # Secondary_Location_Z
                        ]
                        csv_writer.writerow(row)
                
                print(f"Collision data saved to CSV file: {file_path}")
        
        return True
    
    except Exception as e:
        print(f"Error saving collision data: {e}")
        return False