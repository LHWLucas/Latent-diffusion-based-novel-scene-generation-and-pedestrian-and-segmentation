import os
import csv

files = [f for f in os.listdir("./") if f.endswith('.jpg') or f.endswith('.png')] if os.path.exists("./") else []
print(files)

image_column = []

# Open the CSV file for reading
with open('./desc.csv', 'r') as csv_file:
    # Create a CSV reader object
    csv_reader = csv.DictReader(csv_file)
    
    # Iterate through each row in the CSV file
    for row in csv_reader:
        # Append the value from the "image" column to the list
        image_column.append(row['image'])

# Print the list of image values
print(image_column)

for i in image_column:
    if i in files:
        print("1")
    else:
        print("0")