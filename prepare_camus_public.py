import os
import shutil

root_dir = 'CAMUS_public'
dataset_nfti_dir = os.path.join(root_dir, 'database_nifti')
dataset_split_dir = os.path.join(root_dir, 'database_split')
subgroup_testing_file = os.path.join(dataset_split_dir, 'subgroup_testing.txt')

test_dir = os.path.join(root_dir, 'test')
training_dir = os.path.join(root_dir, 'training')

os.makedirs(test_dir, exist_ok=True)
os.makedirs(training_dir, exist_ok=True)

with open(subgroup_testing_file, 'r') as file:
    test_patients = set(line.strip() for line in file)

for patient_folder in os.listdir(dataset_nfti_dir):
    patient_path = os.path.join(dataset_nfti_dir, patient_folder)

    if os.path.isdir(patient_path):
        if patient_folder in test_patients:
            shutil.move(patient_path, os.path.join(test_dir, patient_folder))
        else:
            shutil.move(patient_path, os.path.join(training_dir, patient_folder))

