import json
import numpy as np

# Load your JSON file
with open('smpl.json', 'r') as file:
    data = json.load(file)

# Extract only the values (which are lists of numbers)
values = list(data.values())

# Convert to NumPy array
array = np.array(values)

# Save to a text file (each row corresponds to one key's values)
np.savetxt('smpl.txt', array, fmt='%.8f')  # Adjust precision/format as needed

print("Saved coordinates to output.txt")