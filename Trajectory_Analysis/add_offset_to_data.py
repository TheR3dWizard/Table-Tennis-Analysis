import json

# Offsets
x_offset = -190
y_offset = -100

# Input and output file paths
input_file = "ball_trajectory_v1.json"
output_file = "ball_trajectory_v1_offset.json"

# Load JSON data
with open(input_file, "r") as f:
    data = json.load(f)

# Apply offsets
for key, value in data.items():
    if "x" in value and "y" in value:
        value["x"] += x_offset
        value["y"] += y_offset

# Save updated data
with open(output_file, "w") as f:
    json.dump(data, f, indent=4)

print(f"Updated coordinates saved to {output_file}")
