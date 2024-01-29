from setuptools import setup, find_packages
import os

def list_matlab_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.m'):
                # Here, we strip the 'src/' part, as the paths in package_data should be relative to the package
                yield os.path.relpath(os.path.join(root, file), 'src')

def list_DL_models(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.h5'):
                # Here, we strip the 'src/' part, as the paths in package_data should be relative to the package
                yield os.path.relpath(os.path.join(root, file), 'src')

matlab_files_directory = 'src/BenchFRET/ebFRET'
deeplasi_models_directory = 'src/BenchFRET/DeepLASI/models'

matlab_files = list(list_matlab_files(matlab_files_directory))
deeplasi_models = list(list_DL_models(deeplasi_models_directory))

setup(
    package_data={
        # Include files in the package
        'BenchFRET': matlab_files,
        'BenchFRET': deeplasi_models,
    },
)
