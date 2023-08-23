import cv2
import shutil
import os
from tkinter import filedialog
from PoseEstimationProcessor import *
from visualize import create_mask, create_viz, create_viz_from_points, create_mask_from_points
import json
import re

user_custom = {}

def ask_for_masks():
    que = input('Do you already have json polygons (masks) made - y or n: ') 
    already_created = que == 'y'
    if not already_created:
        return create_masks_from_video()
    else:
        return use_existing_mask()

def create_masks_from_video():

    print("Enter the video that you want to use:")

    video_input = filedialog.askopenfile().name
    video_name_ext = os.path.basename(video_input)
    video_name = os.path.splitext(video_name_ext)[0]
    video_path = f'video-input/{video_name_ext}'

    shutil.copy2(video_input, video_path)
    os.system(f'video-toimg "{video_path}"')

    print("Your video was loaded successfully into frames")
    json_name = input("What do you want to name the output json file: ")

    print("Once the gui loads, click the open-dir button and \n navigate to the created frame directory and select it")
    print("Then, use the gui to create polygons around the areas of interest. Once completed, hit save")

    json_target = f"json-target/{json_name}.json"
    os.system(f'labelme -o {json_target}')

    return json_name, json_target

def create_points_from_video(video):

    # Define the callback function for mouse events
    circles = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f'Clicked point: ({x}, {y})')
            circles.append((x,y))
        
    

    # Load the video

    cap = cv2.VideoCapture(video)

    # Check if the video is opened successfully
    if not cap.isOpened():
        print('Error: Could not open the video')
        exit()

    # Set up the window and mouse callback
    cv2.namedWindow('Video')
    cv2.setMouseCallback('Video', click_event)

    # Loop through the video frames
    while True:
        ret, frame = cap.read()

        # Break the loop if the video has ended
        if not ret:
            break

        # Draw the circle

        for circle in circles:
            circle_x, circle_y = circle
            cv2.circle(frame, (circle_x, circle_y), 5, (0, 255, 0), -1)

        cv2.imshow('Video', frame)

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release the video and close the window
    cap.release()
    cv2.destroyAllWindows()

    return circles

def use_existing_mask():

    print("Pick the mask file that you want to use")

    json_name = filedialog.askopenfile().name
    json_name = os.path.splitext(os.path.basename(json_name))[0]
    json_target = f"json-target/{json_name}.json"

    return json_name, json_target

def mask_exists(csv_loc):
    for mask, values in user_custom['mask_locs'].items():
        # print(values)
        if csv_loc in values:
            return True
            
    return False

def process_csv_file(csv_loc, masks, objs, noangle=False):

    print("Selected CSV file as " + csv_loc)
    print("Loading data and extracting the coordinates")

    data = load_data(csv_loc)
    snout_x, snout_y, base_x, base_y = extract_coordinates(data)
    if noangle:
        exploration = calculate_exploration_noangle(snout_x, snout_y, masks, fps=15)
    else:
        exploration = calculate_exploration(snout_x, snout_y, base_x, base_y, objs, masks, fps=15)
    entries, exits = calculate_entries_exits(snout_x, snout_y, masks)

    out = []

    for i in range(len(masks)):
        out.extend([exploration[i], entries[i], exits[i]])

    return out

def process_csv_files(masks, objs, count=1):
    # Define the columns of the DataFrame
    objectslen = len(masks)
    columns = []

    for i in range(1, objectslen+1):
        columns.extend([f"object{i}_exploration", f"object{i}_entries", f"object{i}_exits"])

    # Create an empty DataFrame
    df = pd.DataFrame(columns=columns)

    if count == 1:
        print("Prompting users to select a CSV file for the Object Detection Analysis")
        print("This file should be in deep lab cut CSV export format")

        csv_loc = filedialog.askopenfilename()

        assert not mask_exists(csv_loc), "Mask does not exist for " + csv_loc

        # Append the result of process_csv_file as a row to the DataFrame
        df.loc[csv_loc] = process_csv_file(csv_loc, masks, objs)

    else:
        print("Prompting users to select a CSV file for the Object Detection Analysis")
        print("This file should be in deep lab cut CSV export format")

        csv_locs = filedialog.askopenfilenames()

        for loc in csv_locs:
            assert mask_exists(loc), "Mask does not exist for " + loc
            # Append the result of process_csv_file as a row to the DataFrame
            df.loc[loc] = process_csv_file(loc, masks, objs)

    return df
        
def load_json(json_target, json_filename ='fileinfo.json'):
    global user_custom
    with open(json_filename, 'r') as f:
            data = json.load(f)
    user_custom = data

def append_masks_to_json(file_paths, json_target, json_filename='fileinfo.json'):
    global user_custom
    if os.path.exists(json_filename):
        with open(json_filename, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    if 'mask_locs' not in data:
        data['mask_locs'] = {}

    if json_target not in data['mask_locs']:
        data['mask_locs'][json_target] = file_paths
    else: 
        data['mask_locs'][json_target].extend(file_paths)

    user_custom = data

    with open(json_filename, 'w') as f:
        json.dump(data, f, indent=4)

def main(mask_ext):

    load_json('json')

    print("We will begin the Object Detection Analysis Pipeline!")

    json_name, json_target = use_existing_mask()

    masks = create_mask(name=f"{json_name}-mask", input_json=json_target, pixel_ext = mask_ext)
    objs = create_mask(name=f"{json_name}-mask", input_json=json_target, pixel_ext = 0)


    df = process_csv_files(masks, objs, count=2)

    # loop through rows of DataFrame
    for i, row in df.iterrows():
        print(f"File: {i}")
        print(f"Object 1 Exploration: {row['object1_exploration']} Seconds, Object 2 Exploration: {row['object2_exploration']} Seconds")
        print(f"Object 1 Entries: {row['object1_entries']}, Object 1 Exits: {row['object1_exits']}")
        print(f"Object 2 Entries: {row['object2_entries']}, Object 2 Exits: {row['object2_exits']}")
        print("\n")

        # plot_exploration(row['object1_exploration'], row['object2_exploration'])
        # plot_entries_exits([row['object1_entries'], row['object2_entries']], [row['object1_exits'], row['object2_exits']])

def NOR(name, mask_ext):
    global user_custom
    load_json(json_target='fileinfo.json')

    print("What are all mask names to use: ")
    mask_names = [os.path.basename(f) for f in filedialog.askopenfilenames()]

    df = pd.DataFrame()

    for mask in mask_names:
        for mask_loc, files in user_custom['mask_locs'].items():
            if mask in mask_loc:
                for file in files:
                    # Extract the test number from the file path
                    test_number = int(re.search('Test (\d+)', os.path.basename(file)).group(1))
                    print(f"file: {file} + test_number: {test_number}")

                    # Look up the animal type and animal letter using the test number
                    animal_type = None
                    animal_not_present = False

                    pattern = r"c(\d+)"

                 # Use the re.search function to search for the pattern in the string
                    match = re.search(pattern, file)

                    if match:
                        # If a match is found, the group() function returns the matched string
                        cohort_index = int(match.group(1))
                    else:
                        print("No match found")
                        cohort_index=1

                    # for cohort, animals in user_custom['mice_in_cohorts'].items():
                        # print(treatment, animals)
                        # if str(c) in cohort:

                    for at, tests in user_custom['mice_in_cohorts'][f"c{cohort_index}"].items():
                        print(at, tests)
                        if tests == -1:
                            animal_not_present == True

                        elif test_number in tests:
                            animal_type = at
                            break
                    
                    if animal_type is None:
                        print(f"Warning: Test number '{test_number}' not found in mice_in_cohorts data.")
                        continue

                    masks = create_mask(name=f"{mask}-mask", input_json=mask_loc, pixel_ext = mask_ext)
                    objs = create_mask(name=f"{mask}-mask", input_json=mask_loc, pixel_ext = 0)

                    object1_exploration, object1_entries, object1_exits, object2_exploration, object2_entries, object2_exits = tuple(process_csv_file(file, masks, objs))
                    # print(results)

                    if test_number <  user_custom['Threshold'][f"c{cohort_index}"][0]:
                        obj1 = "leftold"
                        obj2 = "rightold"
                    else:
                        obj1 = "novel"
                        obj2 = "old"
                    
                    # Create a temporary DataFrame to hold the results
                    cohortstring = f"c{cohort_index}"
                    if not animal_not_present:
                        # results = object1_exploration, object2_exploration, object1_entries, object1_exits, object2_entries, object2_exits
                        if test_number > user_custom['Threshold'][cohortstring][1]:
                            results = object2_exploration, object1_exploration, object2_entries, object2_exits, object1_entries, object1_exits
                            temp_df = pd.DataFrame([results], columns=[f'{obj1}_exploration', f'{obj2}_exploration', f'{obj1}_entries', f'{obj1}_exits', f'{obj2}_entries', f'{obj2}_exits'])
                        else:
                            results = object1_exploration, object2_exploration, object1_entries, object1_exits, object2_entries, object2_exits
                            temp_df = pd.DataFrame([results], columns=[f'{obj1}_exploration', f'{obj2}_exploration', f'{obj1}_entries', f'{obj1}_exits', f'{obj2}_entries', f'{obj2}_exits'])

                        if test_number <  user_custom['Threshold'][cohortstring][0]:
                            temp_df['left_discrim'] = (temp_df['leftold_exploration']) / (temp_df['leftold_exploration'] + temp_df['rightold_exploration'])
                            temp_df['next_iden'] = f"c{cohort_index}-{test_number + user_custom['Between'][cohortstring]}"
                        else:
                            temp_df['discrimination_index'] = (temp_df['novel_exploration'] - temp_df['old_exploration']) / (temp_df['novel_exploration'] + temp_df['old_exploration'])

                        temp_df['mask_loc'] = mask_loc
                        temp_df['animal_type'] = animal_type
                        temp_df['identifer'] = f"c{cohort_index}-{test_number}"
                        # temp_df['animal_letter'] = animal_letter
                        temp_df['file'] = file

                        # Append the results to the main DataFrame
                        df = pd.concat([df, temp_df])

    df.to_csv(f'{name}.csv')

def NOR_points(name, mask_ext):
    global user_custom
    load_json(json_target='fileinfo.json')

    print("What are all mask names to use: ")
    mask_names = [os.path.basename(f) for f in filedialog.askopenfilenames()]

    print("Select video")
    video = filedialog.askopenfilename()
    # video = 'C:/Users/EphysLaptop/Documents/Hussaini Lab Initial DLC Object Testing/2022-02-07 - NOR - COHORT 2/Test 45.mp4'
    print(video)
    
    points = create_points_from_video(video)

    df = pd.DataFrame()

    for mask in mask_names:
        for mask_loc, files in user_custom['mask_locs'].items():
            if mask in mask_loc:
                masks = create_mask_from_points('test', points, pixel_ext=mask_ext)
                for file in files:
                    # Extract the test number from the file path
                    test_number = int(re.search('Test (\d+)', os.path.basename(file)).group(1))
                    print(f"file: {file} + test_number: {test_number}")

                    # Look up the animal type and animal letter using the test number
                    animal_type = None
                    animal_not_present = False

                    pattern = r"c(\d+)"

                 # Use the re.search function to search for the pattern in the string
                    match = re.search(pattern, file)

                    if match:
                        # If a match is found, the group() function returns the matched string
                        cohort_index = int(match.group(1))
                    else:
                        print("No match found")
                        cohort_index=1

                    # for cohort, animals in user_custom['mice_in_cohorts'].items():
                        # print(treatment, animals)
                        # if str(c) in cohort:

                    for at, tests in user_custom['mice_in_cohorts'][f"c{cohort_index}"].items():
                        print(at, tests)
                        if tests == -1:
                            animal_not_present == True

                        elif test_number in tests:
                            animal_type = at
                            break
                    
                    if animal_type is None:
                        print(f"Warning: Test number '{test_number}' not found in mice_in_cohorts data.")
                        continue

                    object1_exploration, object1_entries, object1_exits, object2_exploration, object2_entries, object2_exits = tuple(process_csv_file(file, masks, objs=None, noangle=True))
                    # print(results)

                    if test_number <  user_custom['Threshold'][f"c{cohort_index}"][0]:
                        obj1 = "leftold"
                        obj2 = "rightold"
                    else:
                        obj1 = "novel"
                        obj2 = "old"
                    
                    # Create a temporary DataFrame to hold the results
                    cohortstring = f"c{cohort_index}"
                    if not animal_not_present:
                        # results = object1_exploration, object2_exploration, object1_entries, object1_exits, object2_entries, object2_exits
                        if test_number > user_custom['Threshold'][cohortstring][1]:
                            results = object2_exploration, object1_exploration, object2_entries, object2_exits, object1_entries, object1_exits
                            temp_df = pd.DataFrame([results], columns=[f'{obj1}_exploration', f'{obj2}_exploration', f'{obj1}_entries', f'{obj1}_exits', f'{obj2}_entries', f'{obj2}_exits'])
                        else:
                            results = object1_exploration, object2_exploration, object1_entries, object1_exits, object2_entries, object2_exits
                            temp_df = pd.DataFrame([results], columns=[f'{obj1}_exploration', f'{obj2}_exploration', f'{obj1}_entries', f'{obj1}_exits', f'{obj2}_entries', f'{obj2}_exits'])

                        if test_number <  user_custom['Threshold'][cohortstring][0]:
                            temp_df['left_discrim'] = (temp_df['leftold_exploration']) / (temp_df['leftold_exploration'] + temp_df['rightold_exploration'])
                            temp_df['next_iden'] = f"c{cohort_index}-{test_number + user_custom['Between'][cohortstring]}"
                        else:
                            temp_df['discrimination_index'] = (temp_df['novel_exploration'] - temp_df['old_exploration']) / (temp_df['novel_exploration'] + temp_df['old_exploration'])

                        temp_df['mask_loc'] = mask_loc
                        temp_df['animal_type'] = animal_type
                        temp_df['identifer'] = f"c{cohort_index}-{test_number}"
                        # temp_df['animal_letter'] = animal_letter
                        temp_df['file'] = file

                        # Append the results to the main DataFrame
                        df = pd.concat([df, temp_df])

    df.to_csv(f'{name}.csv')

def SLR(name, mask_ext):
    global user_custom
    load_json(json_target='fileinfo.json')

    print("What are all mask names to use: ")
    mask_names = [os.path.basename(f) for f in filedialog.askopenfilenames()]

    df = pd.DataFrame()

    for mask in mask_names:
        for mask_loc, files in user_custom['mask_locs'].items():
            if mask in mask_loc:
                masks = create_mask(name=f"{mask}-mask", input_json=mask_loc, pixel_ext = mask_ext)
                objs = create_mask(name=f"{mask}-mask", input_json=mask_loc, pixel_ext = 0)
                for file in files:
                    # Extract the test number from the file path
                    test_number = int(re.search('Test (\d+)', os.path.basename(file)).group(1))
                    print(f"file: {file} + test_number: {test_number}")

                    # Look up the animal type and animal letter using the test number
                    animal_type = None
                    animal_not_present = False

                    pattern = r"c(\d+)"

                 # Use the re.search function to search for the pattern in the string
                    match = re.search(pattern, file)

                    if match:
                        # If a match is found, the group() function returns the matched string
                        cohort_index = int(match.group(1))
                    else:
                        print("No match found")
                        cohort_index=1

                    # for cohort, animals in user_custom['mice_in_cohorts'].items():
                        # print(treatment, animals)
                        # if str(c) in cohort:

                    for at, tests in user_custom['mice_in_cohorts'][f"c{cohort_index}"].items():
                        print(at, tests)
                        if tests == -1:
                            animal_not_present == True

                        elif test_number in tests:
                            animal_type = at
                            break
                    
                    if animal_type is None:
                        print(f"Warning: Test number '{test_number}' not found in mice_in_cohorts data.")
                        continue

                    row = process_csv_file(file, masks, objs)

                    columns = []

                    for i in range(1, len(objs)+1):
                        columns.extend([f"object{i}_exploration", f"object{i}_entries", f"object{i}_exits"])

                    if not animal_not_present:
                        print(row, columns)
                        temp_df = pd.DataFrame(data=[row], columns=columns)
                        temp_df['mask_loc'] = mask_loc
                        temp_df['animal_type'] = animal_type
                        temp_df['identifer'] = f"c{cohort_index}-{test_number}"
                        # temp_df['animal_letter'] = animal_letter
                        temp_df['file'] = file

                        # Append the results to the main DataFrame
                        df = pd.concat([df, temp_df])

    df.to_csv(f'{name}.csv')

def SLR_points(name, mask_ext):
    global user_custom
    load_json(json_target='fileinfo.json')

    print("What are all mask names to use: ")
    mask_names = [os.path.basename(f) for f in filedialog.askopenfilenames()]

    df = pd.DataFrame()

    print("Select video")
    video = filedialog.askopenfilename()
    # video = 'C:/Users/EphysLaptop/Documents/Hussaini Lab Initial DLC Object Testing/2022-02-07 - NOR - COHORT 2/Test 45.mp4'
    print(video)
    
    points = create_points_from_video(video)

    for mask in mask_names:
        for mask_loc, files in user_custom['mask_locs'].items():
            if mask in mask_loc:
                masks = create_mask_from_points(name=f"{mask}-mask", points=points, pixel_ext = mask_ext)
                for file in files:
                    # Extract the test number from the file path
                    test_number = int(re.search('Test (\d+)', os.path.basename(file)).group(1))
                    print(f"file: {file} + test_number: {test_number}")

                    # Look up the animal type and animal letter using the test number
                    animal_type = None
                    animal_not_present = False

                    pattern = r"c(\d+)"

                 # Use the re.search function to search for the pattern in the string
                    match = re.search(pattern, file)

                    if match:
                        # If a match is found, the group() function returns the matched string
                        cohort_index = int(match.group(1))
                    else:
                        print("No match found")
                        cohort_index=1

                    # for cohort, animals in user_custom['mice_in_cohorts'].items():
                        # print(treatment, animals)
                        # if str(c) in cohort:

                    for at, tests in user_custom['mice_in_cohorts'][f"c{cohort_index}"].items():
                        print(at, tests)
                        if tests == -1:
                            animal_not_present == True

                        elif test_number in tests:
                            animal_type = at
                            break
                    
                    if animal_type is None:
                        print(f"Warning: Test number '{test_number}' not found in mice_in_cohorts data.")
                        continue

                    row = process_csv_file(file, masks, None, noangle=True)

                    columns = []

                    for i in range(1, len(masks)+1):
                        columns.extend([f"object{i}_exploration", f"object{i}_entries", f"object{i}_exits"])

                    if not animal_not_present:
                        print(row, columns)
                        temp_df = pd.DataFrame(data=[row], columns=columns)
                        temp_df['mask_loc'] = mask_loc
                        temp_df['animal_type'] = animal_type
                        temp_df['identifer'] = f"c{cohort_index}-{test_number}"
                        # temp_df['animal_letter'] = animal_letter
                        temp_df['file'] = file

                        # Append the results to the main DataFrame
                        df = pd.concat([df, temp_df])

    df.to_csv(f'{name}.csv')

def vid(mask_ext):
    print("Select video")
    video = filedialog.askopenfilename()
    # video = 'C:/Users/EphysLaptop/Documents/Hussaini Lab Initial DLC Object Testing/2022-02-07 - NOR - COHORT 2/Test 45.mp4'
    print(video)
    

    print("Select corresponding CSV")
    csv_loc = filedialog.askopenfilename()
    # csv_loc = 'C:/Users/EphysLaptop/Documents/Hussaini Lab Initial DLC Object Testing/analysis-pipeline/3p-analysis/c2/Test 45DLC_resnet50_m-orientationJul10shuffle1_645000.csv'
    print(csv_loc)

    load_json('json')
    usable = None

    for mask, values in user_custom['mask_locs'].items():
        if csv_loc in values:
            usable = mask
            break

    print(csv_loc, usable)

    masks = create_mask(name=f"{usable}-mask", input_json=usable, pixel_ext = mask_ext)
    objs = create_mask(name=f"{usable}-mask", input_json=usable, pixel_ext = 0)



    data = load_data(csv_loc)
    snout_x, snout_y, base_x, base_y = extract_coordinates(data)

    create_viz(video, snout_x, snout_y, base_x, base_y, objs, masks, 15)

def vid_circle(mask_ext):
    print("Select video")
    video = filedialog.askopenfilename()
    # video = 'C:/Users/EphysLaptop/Documents/Hussaini Lab Initial DLC Object Testing/2022-02-07 - NOR - COHORT 2/Test 45.mp4'
    print(video)
    

    print("Select corresponding CSV")
    csv_loc = filedialog.askopenfilename()
    # csv_loc = 'C:/Users/EphysLaptop/Documents/Hussaini Lab Initial DLC Object Testing/analysis-pipeline/3p-analysis/c2/Test 45DLC_resnet50_m-orientationJul10shuffle1_645000.csv'
    print(csv_loc)

    load_json('json')
    usable = None

    for mask, values in user_custom['mask_locs'].items():
        if csv_loc in values:
            usable = mask
            break

    print(csv_loc, usable)


    data = load_data(csv_loc)
    snout_x, snout_y, base_x, base_y = extract_coordinates(data)
    points = create_points_from_video(video)
    create_viz_from_points(video, snout_x, snout_y, base_x, base_y, points, fps=15, ext=mask_ext)

if __name__ == "__main__":

    if not os.path.exists("json-target"): os.mkdir('json-target')
    if not os.path.exists("video-input"): os.mkdir('video-input')

    answer = ""
    while answer != "0":
        print("What do you want to do:\n[0] Quit the program\n[1] Create Masks\n[2] Add Files to Masks\n[3] Run NOR Analysis of All Files in Masks\n[4] Run NOR Analysis of One File with One Mask\n[5] Run SLR Analysis of All Files\n[6] Run NOR with Only Centerpoint Values\n[7] Run SLR with Centerpoints\n[8] Visualize\n[9] Visualize Centerpoint")
        answer = input()

        if answer == "1":
            json_name, json_target = create_masks_from_video()

        elif answer == "2":
            print("What files do you want to add:")
            mask_files = filedialog.askopenfilenames()

            print("What is the target?")
            name = filedialog.askopenfile()
            name = f"json-target/{os.path.basename(name.name)}"
            append_masks_to_json(mask_files, name)

        elif answer == "3":
            name = input("name of file: ")
            mask_ext = int(input('What is the mask pixel extension: '))
            NOR(name, mask_ext)

        elif answer == "4":
            main(mask_ext=20)

        elif answer == "5":
            name = input("name of file: ")
            mask_ext = int(input('What is the mask pixel extension: '))
            SLR(name, mask_ext)

        elif answer == "9":
            mask_ext = int(input('What is the mask pixel extension: '))
            vid_circle(mask_ext)

        elif answer == "6":
            name = input("name of file: ")
            mask_ext = int(input('What is the mask pixel extension: '))
            NOR_points(name, mask_ext)

        elif answer == "7":
            name = input("name of file: ")
            mask_ext = int(input('What is the mask pixel extension: '))
            SLR_points(name, mask_ext)

        elif answer == "8":
            mask_ext = int(input('What is the mask pixel extension: '))
            vid(mask_ext)
