import pickle

# Path to the pickle file
file_path = 'C:\\Thesis_Destinus\\Script\\Naca0012.pickle'

# Open the file in binary read mode
with open(file_path, 'rb') as file:
    # Load the data from the pickle file
    data = pickle.load(file)

# Display the type of the loaded object
print(f"Type of loaded object: {type(data)}")

# Display the attributes and methods of the object
print("\nAttributes and methods of the loaded object:")
print(dir(data))

# Access and call specific methods to get the desired values
if hasattr(data, 'sobols_first'):
    sobols_first_values = data.sobols_first()
    print("\nSobol's first order indices:")
    print(sobols_first_values)

if hasattr(data, 'sobols_total'):
    sobols_total_values = data.sobols_total()
    print("\nSobol's total order indices:")
    print(sobols_total_values)
