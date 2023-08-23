"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog

import matplotlib.pyplot as plt

class PosePlotting:

    def plot_distances_over_time(procs):
        n_procs = len(procs)
        fig, axes = plt.subplots(1, n_procs, figsize=(7*n_procs, 6), sharey=True)
        for idx, proc in enumerate(procs):
            if proc.contains_circles:
                for i in range(len(proc.circles)):
                    axes[idx].plot(proc.snout_circle_distances[i], label=f'Circle {i+1}')
                axes[idx].set_title(f'Processor {idx+1}')
                axes[idx].set_xlabel('Frame')
                axes[idx].legend()
        fig.suptitle('Distance from Snout to Each Circle Over Time')
        plt.ylabel('Distance (pixels)')
        plt.show()

    def plot_orientations_over_time(procs):
        n_procs = len(procs)
        fig, axes = plt.subplots(1, n_procs, figsize=(7*n_procs, 6), sharey=True)
        for idx, proc in enumerate(procs):
            if proc.contains_circles:
              for i in range(len(proc.circles)):
                  axes[idx].plot(proc.orientations[i], label=f'Circle {i+1}')
              axes[idx].set_title(f'Processor {idx+1}')
              axes[idx].set_xlabel('Frame')
              axes[idx].legend()
        fig.suptitle('Orientation of Mouse with Respect to Each Circle Over Time')
        plt.ylabel('Orientation (degrees)')
        plt.show()

    def plot_time_spent_idle_vs_moving(procs):
        n_procs = len(procs)
        fig, axes = plt.subplots(1, n_procs, figsize=(7 * n_procs, 7))
        for idx, proc in enumerate(procs):
            axes[idx].pie([proc.percentage_idle, proc.percentage_moving], labels=['Idle', 'Moving'], autopct='%1.1f%%')
            axes[idx].set_title(f'Processor {idx+1}')
        fig.suptitle('Percentage of Time Spent Idle vs. Moving')
        plt.show()

    def plot_time_spent_in_each_quadrant(procs):
        n_procs = len(procs)
        fig, axes = plt.subplots(1, n_procs, figsize=(7 * n_procs, 7))
        for idx, proc in enumerate(procs):
            axes[idx].pie(proc.percentage_in_quadrants, labels=['Quadrant 1', 'Quadrant 2', 'Quadrant 3', 'Quadrant 4'], autopct='%1.1f%%')
            axes[idx].set_title(f'Processor {idx+1}')
        fig.suptitle('Percentage of Time Spent in Each Quadrant')
        plt.show()


    def plot_turns_and_straight_periods(procs):
        n_procs = len(procs)
        fig, axes = plt.subplots(1, n_procs, figsize=(10 * n_procs, 6))
        for idx, proc in enumerate(procs):
            axes[idx].bar(['Left Turns', 'Right Turns'], [proc.left_turns.sum(), proc.right_turns.sum()])
            axes[idx].set_title(f'Processor {idx+1}')
            axes[idx].set_xlabel('Behavior')
            axes[idx].set_ylabel('Count')
        fig.suptitle('Number of Left Turns, Right Turns, and Periods of Going Straight')
        plt.show()

    def plot_approaches_to_objects(procs):
        n_procs = len(procs)
        fig, axes = plt.subplots(1, n_procs, figsize=(7 * n_procs, 4))
        for idx, proc in enumerate(procs):
            if proc.contains_circles:
                axes[idx].bar([f"Circle At {f}" for f in list(procs[idx].circles)], proc.approach_to_objects)
                axes[idx].set_title(f'Processor {idx+1}')
                axes[idx].set_xlabel('Object')
                axes[idx].set_ylabel('Count')
        fig.suptitle('Number of Approaches to Each Object')
        plt.show()

    def plot_time_spent_near_each_circle(procs):
      n_procs = len(procs)
      fig, axes = plt.subplots(1, n_procs, figsize=(7 * n_procs, 7))
      for idx, proc in enumerate(procs):
          if proc.contains_circles:
              axes[idx].pie(proc.percentage_near_points, labels=[f"Circle At {f}" for f in list(procs[idx].circles)], autopct='%1.1f%%')
              axes[idx].set_title(f'Processor {idx+1}')
      fig.suptitle('Percentage of Time Spent near Each Circle')
      plt.show()


    def plot_time_near_object_pie(procs):
        fig, axs = plt.subplots(1, len(procs), figsize=(7 * len(procs), 7))
        if len(procs) == 1:
            axs = [axs]
        for i, proc in enumerate(procs):
            if proc.contains_circles:
                axs[i].pie(proc.time_near_object, labels=[f"Circle At {f}" for f in list(procs[i].circles)], autopct='%1.1f%%')
                axs[i].set_title(f'Time Spent Near Each Object for Processor {i+1}')
        plt.show()

    def plot_visits_to_object_pie(procs):
        fig, axs = plt.subplots(1, len(procs), figsize=(7 * len(procs), 7))
        if len(procs) == 1:
            axs = [axs]
        for i, proc in enumerate(procs):
            if proc.contains_circles:
                axs[i].pie(proc.visits_to_object, labels=[f"Circle At {f}" for f in list(procs[i].circles)], autopct='%1.1f%%')
                axs[i].set_title(f'Number of Visits to Each Object for Processor {i+1}')
        plt.show()

    def plot_time_near_object_bar(procs):
        fig, axs = plt.subplots(1, len(procs), figsize=(7 * len(procs), 7), sharey=True)
        if len(procs) == 1:
            axs = [axs]
        for i, proc in enumerate(procs):
            if proc.contains_circles:
                axs[i].bar([f"Circle At {f}" for f in list(procs[i].circles)], proc.time_near_object, alpha=0.5)
                axs[i].set_title(f'Time Spent Near Each Object for Data {i+1}')
                axs[i].set_xlabel('Object')
                axs[i].set_ylabel('Time (fraction of total)')
        plt.tight_layout()
        plt.show()

    def plot_visits_to_object_bar(procs):
        fig, axs = plt.subplots(1, len(procs), figsize=(7 * len(procs), 7), sharey=True)
        if len(procs) == 1:
            axs = [axs]
        for i, proc in enumerate(procs):
            if proc.contains_circles:
                axs[i].bar([f"Circle At {f}" for f in list(procs[i].circles)], proc.visits_to_object, alpha=0.5)
                axs[i].set_title(f'Number of Visits to Each Object for Data {i+1}')
                axs[i].set_xlabel('Object')
                axs[i].set_ylabel('Number of Visits')
        plt.tight_layout()
        plt.show()

    def plot_paths(procs):
        plt.figure(figsize=(10, 10))
        for proc in procs:
            plt.plot(proc.paths[:, 0], proc.paths[:, 1], label=f'Data {procs.index(proc)+1}')
        plt.title('Paths Taken')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()


    plotting_functions = {
    'plot_distances_over_time': plot_distances_over_time,
    'plot_orientations_over_time': plot_orientations_over_time,
    'plot_time_spent_idle_vs_moving': plot_time_spent_idle_vs_moving,
    'plot_time_spent_in_each_quadrant': plot_time_spent_in_each_quadrant,
    'plot_turns_and_straight_periods': plot_turns_and_straight_periods,
    'plot_approaches_to_objects': plot_approaches_to_objects,
    'plot_time_spent_near_each_circle':plot_time_spent_near_each_circle,
    'plot_time_near_object_pie':plot_time_near_object_pie,
    'plot_time_near_object_bar':plot_time_near_object_bar,
    'plot_visits_to_object_pie':plot_visits_to_object_pie,
    'plot_visits_to_object_bar':plot_visits_to_object_bar,
    'plot_paths':plot_paths
}

class PoseEstimationProcessor:
    def __init__(self, data, circles=None, idle_speed_threshold=5, frame_interval=15, turn_threshold=30, straight_threshold=10, approach_frames=100, frame_size=(640, 480), box_sizes=(25, 25), use_mask =False):

        self.data = data.iloc[2:].reset_index(drop=True).astype(float)
        self.circles = circles
        print(type(circles))
        self.contains_circles = True if type(circles)==np.ndarray else False
        print(self.contains_circles)
        self.idle_speed_threshold = idle_speed_threshold
        self.frame_interval = frame_interval
        self.turn_threshold = turn_threshold
        self.straight_threshold = straight_threshold
        self.approach_frames = approach_frames
        self.frame_size = frame_size
        self.box_sizes = box_sizes
        self.use_mask = use_mask

        self._process()

    def _euclidean_distance(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2, axis=-1))

    def _calculate_angle(self, p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p1
        angle = np.arctan2(v2[:,1], v2[:,0]) - np.arctan2(v1[:,1], v1[:,0])
        return np.degrees(angle)

    def _process(self):
        # Extract the snout and neck coordinates
        snout_coords = self.data[['DLC_resnet50_m-orientationJul10shuffle1_540000', 'DLC_resnet50_m-orientationJul10shuffle1_540000.1']].values
        neck_coords = self.data[['DLC_resnet50_m-orientationJul10shuffle1_540000.3', 'DLC_resnet50_m-orientationJul10shuffle1_540000.4']].values


        # Calculate the distance moved in each interval
        distances_moved = self._euclidean_distance(snout_coords[self.frame_interval:], snout_coords[:-self.frame_interval])

        # Determine when the mouse is idle vs. moving
        is_idle = distances_moved < self.idle_speed_threshold

        # Calculate the percentage of time spent idle vs. moving
        self.percentage_idle = np.mean(is_idle) * 100
        self.percentage_moving = 100 - self.percentage_idle

        # Determine the quadrant of each coordinate
        x_mid, y_mid = np.array(self.frame_size) // 2
        quadrants = np.empty_like(snout_coords[:, 0])
        quadrants[(snout_coords[:, 0] <= x_mid) & (snout_coords[:, 1] <= y_mid)] = 1  # Q1
        quadrants[(snout_coords[:, 0] > x_mid) & (snout_coords[:, 1] <= y_mid)] = 2  # Q2
        quadrants[(snout_coords[:, 0] <= x_mid) & (snout_coords[:, 1] > y_mid)] = 3  # Q3
        quadrants[(snout_coords[:, 0] > x_mid) & (snout_coords[:, 1] > y_mid)] = 4  # Q4


        # Calculate the time spent in each quadrant
        time_in_quadrants = np.array([np.sum(quadrants == i) for i in range(1, 5)])

        # Convert to percentages
        self.percentage_in_quadrants = (time_in_quadrants / len(quadrants)) * 100


        # Store the paths taken
        self.paths = self.data[['DLC_resnet50_m-orientationJul10shuffle1_540000', 'DLC_resnet50_m-orientationJul10shuffle1_540000.1']].values

        if self.contains_circles:

            # if not len(self.box_sizes) == 1:
            #   self.box_sizes = [self.box_sizes for b in range(len(self.circles))]
            # elif len(self.box_sizes) == len(self.circles):
            #   self.box_sizes = self.box_sizes
            # else:
            #   assert "Box_Sizes Entered INCORRECTLY"
            # self.box_sizes = self.box_sizes


            # Calculate the distance from the snout and neck to each circle

            self.snout_circle_distances = np.array([self._euclidean_distance(snout_coords, circle) for circle in self.circles])
            self.neck_circle_distances = np.array([self._euclidean_distance(neck_coords, circle) for circle in self.circles])

            # Calculate the orientation of the mouse with respect to each circle
            self.orientations = np.array([self._calculate_angle(snout_coords, neck_coords, circle) for circle in self.circles])

            # Identify approaches to objects
            self.approach_to_objects = np.count_nonzero(np.diff(self.snout_circle_distances, n=self.approach_frames, axis=1) < 0, axis=1)

            # Calculate the time spent near each object
            is_near = self.snout_circle_distances < 2 * 30  # assuming a radius of 20
            self.time_near_object = np.sum(is_near, axis=1) / self.data.shape[0]

            # Calculate the number of visits to each object
            is_visiting = np.concatenate([np.zeros((is_near.shape[0], 1)), is_near[:, 1:] & ~is_near[:, :-1]], axis=1)
            self.visits_to_object = np.sum(is_visiting, axis=1)

            #create areas of interest around the points to see if a mouse is spending more time around one point than the other

            def rescale_coordinates(snout_coords, mask_shape):
                x_scale = mask_shape[0] / 640
                y_scale = mask_shape[1] / 480
                snout_coords[:, 0] *= x_scale
                snout_coords[:, 1] *= y_scale
                return snout_coords

            def get_overlapping_points(snout_coords, mask):
                snout_coords = rescale_coordinates(snout_coords, mask.shape)
                overlapping_points = []
                for i in range(snout_coords.shape[0]):
                    x, y = int(snout_coords[i, 0]), int(snout_coords[i, 1])
                    if x < mask.shape[0] and y < mask.shape[1] and mask[x, y] > 0:
                        overlapping_points.append((x, y))
                return np.array(overlapping_points)

            def time_spent_in_mask(snout_coords, mask):
                overlapping_points = get_overlapping_points(snout_coords, mask)
                time_spent = len(overlapping_points)
                return time_spent

            if self.use_mask:
                circles = np.load("/content/shapes.npy")
                circles = circles.reshape(2, 640, 480)

                total_time_spent = [0, 0]
                for i, circle in enumerate(circles):
                    mask = circle == 1
                    total_time_spent[i] += time_spent_in_mask(snout_coords, mask)

                time_near_points = np.array(total_time_spent)
                self.percentage_near_points = (time_near_points / np.sum(time_near_points)) * 100


            else:
                aoi = np.zeros(shape=snout_coords[:, 0].shape)
                print(self.box_sizes)
                for i, circle in enumerate(self.circles):
                    x, y = tuple(circle)
                    horizontal_mask = (snout_coords[:, 0] <= x+self.box_sizes[i, 0]) & (snout_coords[:, 0] >= x-self.box_sizes[i, 0])
                    vertical_mask = (snout_coords[:, 1] <= y+self.box_sizes[i, 1]) & (snout_coords[:, 1] >= y-self.box_sizes[i, 1])
                    aoi[horizontal_mask & vertical_mask] = i + 1

                time_near_points = np.array([np.sum(aoi == i) for i in range(1, len(self.circles)+1)])
                self.percentage_near_points = (time_near_points / len(aoi)) * 100

            # Calculate the change in orientation between frames
            orientation_changes = np.diff(self.orientations, axis=1)

            # Identify left and right turns
            self.left_turns = np.count_nonzero(orientation_changes > self.turn_threshold, axis=1)
            self.right_turns = np.count_nonzero(orientation_changes < -self.turn_threshold, axis=1)


def compare_processors(processors, method=None):
    if not method:
      for key in PosePlotting.plotting_functions:
          PosePlotting.plotting_functions[key](processors)
    else:
          PosePlotting.plotting_functions[method](processors)





def main():
    # Ask for multiple CSV files
    inputs = filedialog.askopenfiles()

    # # Process each file
    # processors = [PoseEstimationProcessor(pd.read_csv(input), np.array([[231, 181], [414, 177]])) for input in inputs]

    # # Compare the results
    # compare_processors(processors)

    # p1 = PoseEstimationProcessor(pd.read_csv(inputs[0]))
    p2 = PoseEstimationProcessor(pd.read_csv(inputs[1]), np.array([[234, 183], [411, 183]]), box_sizes=np.array([[50, 40], [50, 40]]), use_mask = False)
    p3 = PoseEstimationProcessor(pd.read_csv(inputs[2]), np.array([[234, 183], [411, 183]]), box_sizes=np.array([[50, 50], [50, 40]]), use_mask = False)

    # Process each file
    processors = [p2, p3]

    # Compare the results
    compare_processors(processors, method='plot_time_spent_near_each_circle')
    compare_processors(processors, method='plot_time_near_object_pie')


if __name__ == '__main__':
    main()

"""
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from visualize import calculate_angle

def load_data(data_path):

    data = pd.read_csv(data_path)
    # shapes = np.load(shapes_path)
    return data

def extract_coordinates(data):
    snout_x = data.iloc[2:, 1].astype(float)
    snout_y = data.iloc[2:, 2].astype(float)
    base_x = data.iloc[2:, 4].astype(float)
    base_y = data.iloc[2:, 5].astype(float)
    return snout_x, snout_y, base_x, base_y

def calculate_exploration_noangle(snout_x, snout_y, shapes, fps):
    num_objects = len(shapes)
    exploration_counts = np.zeros(num_objects, dtype=np.int32)

    for sx, sy, in zip(snout_x, snout_y,):
        sx, sy = min(639, int(sx)), min(479, int(sy))
        
        # Check if the mouse is facing the object and is near the object for all objects in a list comprehension
        exploring_objects = [(shapes[i, sy, sx] == 1) for i in range(len(exploration_counts))]

        exploration_counts += exploring_objects

    exploration_times = exploration_counts / fps
    return exploration_times

def calculate_exploration(snout_x, snout_y, neck_x, neck_y, objs, shapes, fps):
    num_objects = len(objs)
    exploration_counts = np.zeros(num_objects, dtype=np.int32)

    # Store object_points as a list of arrays
    object_points = [np.array(np.where(obj == 1)) for obj in objs]

    for sx, sy, nx, ny in zip(snout_x, snout_y, neck_x, neck_y):
        sx, sy = min(639, int(sx)), min(479, int(sy))
        
        # Compute distances for all objects in a list comprehension
        all_distances = [np.sqrt((points[0]-sy)**2 + (points[1]-sx)**2) for points in object_points]

        # Compute nearest_points for all objects in a list comprehension
        nearest_points = [points.T[np.argmin(distances)] for points, distances in zip(object_points, all_distances)]

        # Compute angles for all objects in a list comprehension
        angles = [calculate_angle(np.array([nx, ny]), np.array([sx, sy]), point[::-1]) for point in nearest_points]

        # Check if the mouse is facing the object and is near the object for all objects in a list comprehension
        exploring_objects = [(angle % 90 <= 22.5 and shapes[i, sy, sx] == 1) for i, angle in enumerate(angles)]

        exploration_counts += exploring_objects

    exploration_times = exploration_counts / fps
    return exploration_times

def calculate_entries_exits(snout_x, snout_y, shapes):
    num_objects = shapes.shape[0]
    entries = [0] * num_objects
    exits = [0] * num_objects
    in_object = [False] * num_objects

    for x, y in zip(snout_x, snout_y):
        x, y = int(x), int(y)

        y = 479 if y > 479 else y
        x = 639 if x > 639 else x

        for i in range(num_objects):
            if shapes[i, y, x] == 1:
                if not in_object[i]:
                    entries[i] += 1
                    in_object[i] = True
            else:
                if in_object[i]:
                    exits[i] += 1
                    in_object[i] = False

    return entries, exits


# def plot_exploration(object1_exploration, object2_exploration, labels = ('Object 1', 'Object 2'), colors = ['gold', 'lightcoral']):
#     sizes = [object1_exploration, object2_exploration]
#     explode = (0.1, 0)

#     plt.figure(figsize=(6,6))
#     plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
#     plt.axis('equal')
#     plt.show()

# def plot_entries_exits(entries, exits, labels= ['Object 1', 'Object 2']):
#     # Create subplots
#     fig, axs = plt.subplots(1, 2, figsize=(12, 6))

#     # Create a bar plot for entries
#     axs[0].bar(labels, entries, color=['blue', 'orange'])
#     axs[0].set_title('Number of Entries')
#     axs[0].set_xlabel('Object')
#     axs[0].set_ylabel('Number of Entries')

#     # Create a bar plot for exits
#     axs[1].bar(labels, exits, color=['blue', 'orange'])
#     axs[1].set_title('Number of Exits')
#     axs[1].set_xlabel('Object')
#     axs[1].set_ylabel('Number of Exits')

#     # Display the plots
#     plt.tight_layout()
#     plt.show()

# def plot_interactive_charts(object1_exploration, object2_exploration,  labels = ['Object 1', 'Object 2']):
   
#     values = [object1_exploration, object2_exploration]

#     # Create a pie chart
#     pie_chart = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
#     pie_chart.update_layout(title_text="Exploration Time - Pie Chart")
#     pie_chart.show()

#     # Create a bar plot
#     bar_plot = go.Figure(data=[go.Bar(x=labels, y=values)])
#     bar_plot.update_layout(title_text="Exploration Time - Bar Plot", xaxis_title="Object", yaxis_title="Exploration Time")
#     bar_plot.show()

# def individual_process():
#     data_path = filedialog.askopenfile()
#     shapes_path = 'shapes.npy'

#     data, shapes = load_data(data_path, shapes_path)
#     snout_x, snout_y = extract_coordinates(data)

#     object1_exploration, object2_exploration = calculate_exploration(snout_x, snout_y, shapes)
#     object1_entries, object1_exits, object2_entries, object2_exits = calculate_entries_exits(snout_x, snout_y, shapes)

#     entries = [object1_entries, object2_entries]
#     exits = [object1_exits, object2_exits]

#     print(f"Object 1 Exploration: {object1_exploration} Seconds, Object 2 Exploration: {object2_exploration} Seconds")
#     print(f"Object 1 Entries: {object1_entries}, Object 1 Exits: {object1_exits}")
#     print(f"Object 2 Entries: {object2_entries}, Object 2 Exits: {object2_exits}")

#     plot_exploration(object1_exploration, object2_exploration)
#     plot_entries_exits(entries, exits)

def combo_process():
    pass


if __name__ == "__main__":
    individual_process()
