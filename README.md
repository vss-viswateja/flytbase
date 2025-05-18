## UAV Strategic Deconfliction System

This project addresses the challenge of UAV strategic deconfliction in shared airspace. The implemented system serves as a final authority to verify the safety of a drone's planned mission, ensuring it avoids conflicts with other drones' trajectories. The solution incorporates spatial and temporal checks, providing a crucial component for safe drone operations by analyzing waypoint missions and simulated flight schedules.

### Prerequisites

* **Python 3.x** is required.
* The following Python libraries are necessary:
    * numpy
    * matplotlib
    * mpl\_toolkits.mplot3d
    * math
    * typing
    * csv
    * os
    * datetime
    * enum

    You can install these using pip:
    ```bash
    pip install numpy matplotlib
    ```

### Getting the Code

1.  Clone the repository from GitHub:
    ```bash
    git clone <your_repository_url>
    ```
    (Replace `<your_repository_url>` with the actual URL of your repository.)

2.  Navigate to the project directory:
    ```bash
    cd <your_repository_name>
    ```
    (Replace `<your_repository_name>` with the name of the cloned repository.)

### File Structure

The codebase is organized as follows:

* `UAVDeconflictionVisualizer.py`: Contains the core logic for trajectory generation, conflict checking, distance calculation, visualization, and saving collision data.
* `submission.py`: This file contains the main execution logic, including the `automated_test_suite()` and `test_run()` example functions. It imports necessary components from `UAVDeconflictionVisualizer.py`.
* `log/`: This directory is created during execution to store test results and collision data.
    * `log/tests/`: Contains detailed conflict data for each automated test case (CSV files).
    * `log/collision_data.csv`: Stores collision data when running the example simulation.

### Execution

The primary script for running the simulations is `submission.py`.

#### Running Automated Tests

By default, `submission.py` is configured to execute the automated test suite.

1.  Open your terminal.
2.  Navigate to the project directory.
3.  Run the following command:
    ```bash
    python submission.py
    ```
    This will execute the `automated_test_suite()` function, run through predefined scenarios, print results to the console, and save detailed conflict data in the `log/tests/` directory.

#### Running the Example Simulation

You can also run a specific example simulation scenario defined in the `test_run()` function.

1.  Open the `submission.py` file in a text editor.
2.  Locate the `if __name__ == "__main__":` block at the end of the file.
3.  **Comment out or remove** the line `automated_test_suite()`.
4.  **Uncomment** the line `test_run()`.
5.  Save the file.
6.  Run the script from your terminal:
    ```bash
    python submission.py
    ```
    This will execute the `test_run()` function, generate trajectories, check for conflicts, save collision data to `log/collision_data.csv` (if conflicts are found), and launch an interactive 3D visualization.

### Customizing Scenarios

To define your own deconfliction scenarios, modify the data structures within the `test_run()` function in `submission.py`. This includes:

* `primary_waypoints`: Define the waypoints for the primary drone's mission.
* `primary_times`: Specify the time windows for reaching each primary waypoint.
* `secondary_data`: Define data for secondary drones, including their waypoints and corresponding time steps.

Ensure that the format of these data structures matches the structure used in the existing `test_run()` function and adheres to the requirements outlined in the project documentation.
