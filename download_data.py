import kagglehub

# Download latest version
path = kagglehub.dataset_download("shyambhu/hands-and-palm-images-dataset")

print("Path to dataset files:", path)