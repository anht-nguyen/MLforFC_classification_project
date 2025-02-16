import json
import os
import numpy as np
from datetime import datetime

def save_to_json(output_data, filename_prefix="output_data"):
    """
    Converts NumPy arrays and unsupported data types in a dictionary and saves the data as a JSON file.

    Args:
        output_data (dict or list): The data to be saved.
        filename_prefix (str): The prefix for the output JSON file (default: "output_data").
    
    Returns:
        str: The name of the saved JSON file.
    """
    # Get the current date and time
    now = datetime.now()
    current_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")  # Use underscores to avoid filename issues

    def convert_numpy(obj):
        """ Recursively converts NumPy arrays and unsupported types to JSON-compatible formats. """
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy array to list
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}  # Convert dict values
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]  # Convert list elements
        elif obj == float('inf'):
            return "Infinity"  # JSON doesn't support inf, replace with a string
        else:
            return obj  # Return as is if it's not an ndarray

    # Convert the data
    data_serializable = convert_numpy(output_data)

    # Create the output directory if it doesn't exist
    output_dir = "output_data"
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename with timestamp
    filename = os.path.join(output_dir, f"{filename_prefix}-{current_time_str}.json")

    # Save to JSON
    with open(filename, "w") as f:
        json.dump(data_serializable, f, indent=4)

    print(f"Data successfully saved to {filename}")
    return filename  # Return the filename for reference
