import pandas as pd

# Read the CSV file
df = pd.read_csv('FRONT_BLEND.csv')  # Replace 'your_input_file.csv' with your CSV file path

# Modify the 'image' column
# df['image'] = df['image'].apply(lambda x: x.replace('.png', 'fr.png'))

# Modify the 'prompt' column
df['prompt'] = df['prompt'] + ', front'

# Save the modified data to a new CSV file
df.to_csv('fr.csv', index=False)  # Replace 'your_output_file.csv' with your desired output file name