import pickle

# Chemin vers le fichier pickle
file_path = 'C:\\Thesis_Destinus\\Script\\Naca0012.pickle'

# Ouvrir le fichier en mode lecture binaire
with open(file_path, 'rb') as file:
    # Charger les données du fichier pickle
    data = pickle.load(file)

# Afficher le type de l'objet chargé
print(f"Type of loaded object: {type(data)}")

# Afficher les attributs et les méthodes de l'objet
print("\nAttributes and methods of the loaded object:")
print(dir(data))

# Accéder et appeler les méthodes spécifiques pour obtenir les valeurs souhaitées
if hasattr(data, 'sobols_first'):
    sobols_first_values = data.sobols_first()
    print("\nSobol's first order indices:")
    print(sobols_first_values)

if hasattr(data, 'sobols_total'):
    sobols_total_values = data.sobols_total()
    print("\nSobol's total order indices:")
    print(sobols_total_values)
