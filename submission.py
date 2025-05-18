import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib import colors
import matplotlib.patches as mpatches
import math
from typing import List, Tuple, Dict, Any, Optional, Union
import os
from datetime import datetime
from UAVDeconflictionVisualizer import UAVDeconflictionVisualizer, generate_primary_trajectory, generate_secondary_trajectories, \
    check_intersection, determine_conflict, save_collision_data


def automated_test_suite():
    """Execute all test cases and save results with individual files"""
    # Create test cases with expected conflict counts
    test_cases = [
    {
        "name": "Test Case 1 - Clear Mission",
        "primary": {
            "waypoints": [(0,0,10), (20,0,10)],
            "time_window": (0, 20)
        },
        "secondary": [
            {"id": "A", "waypoints": [(0,20,15), (20,20,15)], "time_steps": [0, 20]},
            {"id": "B", "waypoints": [(10,5,8), (10,15,8)], "time_steps": [5, 15]},
            {"id": "C", "waypoints": [(5,0,20), (5,20,20)], "time_steps": [10, 18]}
        ],
        "expected_conflicts": 1
    },
    {
        "name": "Test Case 2 - Direct Midair Collision",
        "primary": {
            "waypoints": [(0,0,10), (20,20,10)],
            "time_window": (0, 20)
        },
        "secondary": [
            {"id": "A", "waypoints": [(20,20,10), (0,0,10)], "time_steps": [0, 20]},
            {"id": "B", "waypoints": [(10,0,10), (10,20,10)], "time_steps": [5, 15]},
            {"id": "C", "waypoints": [(0,20,10), (20,0,10)], "time_steps": [0, 20]}
        ],
        "expected_conflicts": 0
    },
    {
        "name": "Test Case 3 - Buffer Zone Boundary Case",
        "primary": {
            "waypoints": [(0,0,0), (10,0,0)],
            "time_window": (0, 10)
        },
        "secondary": [
            {"id": "A", "waypoints": [(0,2.0,0), (10,2.0,0)], "time_steps": [0, 10]},
            {"id": "B", "waypoints": [(5,1.9,0), (5,10.9,0)], "time_steps": [5, 10]},
            {"id": "C", "waypoints": [(3,2.1,0), (7,2.1,0)], "time_steps": [3, 7]}
        ],
        "expected_conflicts": 0
    },
    {
        "name": "Test Case 4 - Temporal Separation",
        "primary": {
            "waypoints": [(0,0,0), (10,10,0)],
            "time_window": (7,14)
        },
        "secondary": [
            {"id": "A", "waypoints": [(0,0,0), (10,10,0)], "time_steps": [15, 20]},
            {"id": "B", "waypoints": [(0,0,0), (10,10,0)], "time_steps": [4, 6]},
            {"id": "C", "waypoints": [(0,0,0), (10,10,0)], "time_steps": [0, 5]}
        ],
        "expected_conflicts": 1
    },
    {
        "name": "Test Case 5 - Vertical Stack Conflict",
        "primary": {
            "waypoints": [(5,5,0), (5,5,20)],
            "time_window": (0, 10)
        },
        "secondary": [
            {"id": "A", "waypoints": [(5,5,10), (5,5,10)], "time_steps": [4, 6]},
            {"id": "B", "waypoints": [(5,5,5), (5,5,15)], "time_steps": [0, 10]},
            {"id": "C", "waypoints": [(4,4,8), (6,6,12)], "time_steps": [0, 10]}
        ],
        "expected_conflicts": 0
    },
    {
        "name": "Test Case 6 - Parallel Path Proximity",
        "primary": {
            "waypoints": [(0,0,0), (20,0,0)],
            "time_window": (0, 20)
        },
        "secondary": [
            {"id": "A", "waypoints": [(0,1.5,0), (20,1.5,0)], "time_steps": [0, 20]},
            {"id": "B", "waypoints": [(0,3.0,0), (20,3.0,0)], "time_steps": [0, 20]},
            {"id": "C", "waypoints": [(10,-1.0,0), (10,2.0,0)], "time_steps": [5, 15]}
        ],
        "expected_conflicts": 0
    },
    {
         "name": "Test Case 7 - Late-Starting Drones",
         "primary": {
             "waypoints": [(0,0,0), (10,10,0)],
             "time_window": (5, 15)
         },
         "secondary": [
             {"id": "A", "waypoints": [(5,5,0), (5,5,0)], "time_steps": [6, 14]},
             {"id": "B", "waypoints": [(0,10,0), (10,0,0)], "time_steps": [7, 13]},
             {"id": "C", "waypoints": [(2,2,0), (8,8,0)], "time_steps": [4, 16]}
         ],
         "expected_conflicts": 0
     },
     {
         "name": "Test Case 8 - Complex 3D Maneuvers",
         "primary": {
             "waypoints": [(0,0,5), (10,10,15), (20,0,0)],
             "time_window": (0, 30)
         },
         "secondary": [
             {"id": "A", "waypoints": [(0,0,0), (10,10,15), (20,0,25)], "time_steps": [0,15, 30]},
             {"id": "B", "waypoints": [(5,5,20), (5,5,0)], "time_steps": [10, 20]},
             {"id": "C", "waypoints": [(2,3,7), (8,9,11), (12,5,15)], "time_steps": [0, 10, 20]}  # Example with multiple waypoints
         ],
         "expected_conflicts": 0  # Assuming conflicts occur
     },
     {
         "name": "Test Case 9 - Zero-Length Segments",
         "primary": {
             "waypoints": [(5,5,5), (5,5,5)],
             "time_window": (0, 10)
         },
         "secondary": [
             {"id": "A", "waypoints": [(5,5,5), (5,5,5)], "time_steps": [0, 10]},
             {"id": "B", "waypoints": [(4,4,4), (6,6,6)], "time_steps": [0, 10]},
             {"id": "C", "waypoints": [(5,5,3), (5,5,7)], "time_steps": [0, 10]}
         ],
         "expected_conflicts": 0
     },
     {
         "name": "Test Case 10 - High-Density Traffic",
         "primary": {
             "waypoints": [(0,0,0), (10,10,10)],
             "time_window": (0, 10)
         },
         "secondary": [
             {"id": "A", "waypoints": [(10,10,10), (0,0,0)], "time_steps": [0, 10]},
             {"id": "B", "waypoints": [(5, 5, 5), (8, 8, 8), (5, 5, 5)], "time_steps": [0, 5, 10]},  # Example: Orbital path
             {"id": "C", "waypoints": [(1,1,1), (2,2,2), (2.5, 2.5, 2.5), (3.5, 3.5, 3.5), (4,4,4)], "time_steps": [0, 2, 4, 6, 8]}  # Example: Random walk
         ],
         "expected_conflicts": 0  # Assuming conflicts occur
     }
    ]

    # Create log directory
    os.makedirs("log/tests", exist_ok=True)
    
    test_results = []
    
    for idx, test in enumerate(test_cases, 1):
        print(f"\n{'='*40}\nRunning {test['name']}\n{'='*40}")
        
        try:
            # Generate trajectories
            primary_traj = generate_primary_trajectory(
                test["primary"]["waypoints"],
                test["primary"]["time_window"]
            )
            
            secondary_trajs = generate_secondary_trajectories(test["secondary"])
            
            # Check conflicts
            all_conflicts = []
            for sec_id, sec_traj in secondary_trajs.items():
                intersections = check_intersection(primary_traj, sec_traj)
                has_conflict, conflicts = determine_conflict(
                    primary_traj, sec_traj,
                    intersections, 2.0, sec_id
                )
                if has_conflict:
                    all_conflicts.extend(conflicts)
            
            # Save results with unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"log/tests/test_{idx:02d}_{timestamp}.csv"
            
            if all_conflicts:
                all_conflicts.sort(key=lambda x: x['time'])
                save_collision_data(all_conflicts, filename, "csv")
                conflict_count = len(all_conflicts)
            else:
                conflict_count = 0
                save_collision_data([], filename, "csv")

            if conflict_count ==0:
                conflict_bool = 1
            else:
                conflict_bool = 0
            
            
            # Verify results
            passed = conflict_bool == test["expected_conflicts"]
            test_results.append({
                "test_name": test["name"],
                "status": "PASS" if passed else "FAIL",
                "detected_conflicts": conflict_bool,
                "expected_conflicts": test["expected_conflicts"],
                "log_file": filename
            })
            
        except Exception as e:
            test_results.append({
                "test_name": test["name"],
                "status": "ERROR",
                "message": str(e),
                "log_file": ""
            })

    # Generate summary report
    print("\n\nTest Execution Summary:")
    print(f"{'Test Name':<40} | {'Status':<6} | {'Detected':<8} | {'Expected':<8} | {'Log File'}")
    print("-"*90)
    for result in test_results:
        if result["status"] == "ERROR":
            print(f"{result['test_name']:<40} | {result['status']:<6} | {'N/A':<8} | {'N/A':<8} | ERROR: {result['message']}")
        else:
            print(f"{result['test_name']:<40} | {result['status']:<6} | {result['detected_conflicts']:<8} | {result['expected_conflicts']:<8} | {result['log_file']}")

    return test_results


# Test case
def test_run():
    """ Example run """
    # Define primary drone waypoints and times
    primary_waypoints = [
        (0, 0, 10),      # Starting position
        (20, 0, 10)     # Ending position
    ]
    primary_times = [0, 20]  # Time in seconds 
    
    # Generate primary trajectory
    primary_trajectory = generate_primary_trajectory(primary_waypoints, primary_times)
    
    # Define secondary drones
    secondary_data = [
        {
            'id': 'A',
            'waypoints': [(0, 20, 15), (20, 20, 15)],
            'time_steps': [0, 20]
        },
        {
            'id': 'B',
            'waypoints': [(10, 5, 8), (10, 15, 8)],
            'time_steps': [5, 15]
        },
        {
            'id': 'C',
            'waypoints': [(5,0, 20), (5,20,20 )],
            'time_steps': [10, 18]
        }
    ]
    
    # Generate secondary trajectories
    secondary_trajectories = generate_secondary_trajectories(secondary_data)
    
    # Check for conflicts
    print("Checking for conflicts...")
    conflicts_exist = False
    all_conflicts = []
    for secondary_id, secondary_traj in secondary_trajectories.items():
        print(f"Checking drone {secondary_id}...")
        intersections = check_intersection(primary_trajectory, secondary_traj)
        has_conflict, conflict_details = determine_conflict(
            primary_trajectory, secondary_traj, 
            intersections, 2.0, secondary_id
        )
        
        if has_conflict:
            conflicts_exist = True
            all_conflicts.extend(conflict_details)

    
    if not conflicts_exist:
        print("No conflicts detected in the simulation")
    
    if conflicts_exist:
        print("Conflicts detected in the simulation") 
        all_conflicts.sort(key=lambda x: x['time'])
        save_collision_data(all_conflicts, "log/collision_data_1.csv","csv") 

    # Visualize the trajectories and conflicts
    visualizer = UAVDeconflictionVisualizer(
        primary_trajectory, secondary_trajectories, 
        buffer_distance=2.0, sim_speed=1.0
    )
    
    # Run the animation
    animation = visualizer.animate()
    return animation

if __name__ == "__main__":
    #test_run()
    automated_test_suite()

