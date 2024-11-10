import csv

def calculate_average_speed(csv_file_path):
    def get_valid_float_input(prompt):
        while True:
            try:
                value = float(input(prompt))
                if value < 0:
                    raise ValueError("Value must be non-negative.")
                return value
            except ValueError as e:
                print(e)

    # Allow user to enter multiple swimmer IDs
    swimmer_ids = input("Enter the swimmer IDs (comma-separated if multiple): ").split(',')
    swimmer_ids = [id.strip() for id in swimmer_ids]
    
    total_distance = get_valid_float_input("Enter the total distance covered by the swimmer (in meters): ")

    segments = []

    with open(csv_file_path, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['Swimmer ID'] in swimmer_ids:
                row['Start Time (s)'] = float(row['Start Time (s)'])
                row['End Time (s)'] = float(row['End Time (s)'])
                if 'Duration (s)' not in row or not row['Duration (s)']:
                    row['Duration (s)'] = row['End Time (s)'] - row['Start Time (s)']
                else:
                    row['Duration (s)'] = float(row['Duration (s)'])
                segments.append(row)

    if not segments:
        print(f"No segments found for swimmer IDs {', '.join(swimmer_ids)}.")
        return

    underwater_segments = []
    freestyle_segments = []

    i = 0
    while i < len(segments):
        if segments[i]['Activity'] == 'Freestyle' and i + 1 < len(segments) and segments[i + 1]['Activity'] == 'Underwater':
            # Combine freestyle segment with the following underwater segment
            combined_segment = {
                'Start Time (s)': segments[i]['Start Time (s)'],
                'End Time (s)': segments[i + 1]['End Time (s)'],
                'Duration (s)': segments[i]['Duration (s)'] + segments[i + 1]['Duration (s)'],
                'Activity': 'Underwater'
            }
            underwater_segments.append(combined_segment)
            i += 2
        else:
            if segments[i]['Activity'] == 'Underwater':
                underwater_segments.append(segments[i])
            elif segments[i]['Activity'] == 'Freestyle':
                freestyle_segments.append(segments[i])
            i += 1

    total_underwater_distance = 0
    for segment in underwater_segments:
        distance = get_valid_float_input(f"Enter the distance (in meters) for underwater segment starting at {segment['Start Time (s)']} seconds: ")
        segment['Distance (m)'] = distance
        total_underwater_distance += distance

    total_freestyle_distance = total_distance - total_underwater_distance

    print(f"Total underwater distance for swimmer IDs {', '.join(swimmer_ids)}: {round(total_underwater_distance, 2)} meters")
    print(f"Total freestyle distance for swimmer IDs {', '.join(swimmer_ids)}: {round(total_freestyle_distance, 2)} meters")

    print("\nAverage speeds for each phase:")
    for segment in underwater_segments:
        duration = float(segment['Duration (s)'])
        distance = float(segment['Distance (m)'])
        if duration > 0:
            average_speed = distance / duration
            print(f"Underwater phase starting at {round(float(segment['Start Time (s)']), 2)} seconds: {round(average_speed, 2)} meters/second")

    for segment in freestyle_segments:
        duration = float(segment['Duration (s)'])
        distance = total_freestyle_distance / len(freestyle_segments) if len(freestyle_segments) > 0 else 0
        if duration > 0:
            average_speed = distance / duration
            print(f"Freestyle phase starting at {round(float(segment['Start Time (s)']), 2)} seconds: {round(average_speed, 2)} meters/second")

# Example usage:
# calculate_average_speed('/mnt/data/file.csv')
