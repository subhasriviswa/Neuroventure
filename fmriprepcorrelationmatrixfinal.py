import os
import numpy as np
import pandas as pd
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.masking import apply_mask
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker
from nilearn import datasets

def load_and_process_confounds(subject_dir, session, directory_path, img_file, confound_suffix='_desc-confounds_timeseries.tsv', strategy=('high_pass', 'global_signal'), motion='full', wm_csf='basic', global_signal='basic', demean=True):
    img_path = os.path.join(directory_path, f'sub-{subject_dir}', f'ses-{session}', 'func', img_file)

    if not os.path.exists(img_path):
        raise ValueError(f"The relevant functional MRI data file '{img_file}' is not found in the directory.")
    
    # Construct confound file path using the correct suffix
    confound_file = img_file.replace('_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz', confound_suffix)
    confound_path = os.path.join(directory_path, f'sub-{subject_dir}', f'ses-{session}', 'func', confound_file)

    if not os.path.exists(confound_path):
        raise ValueError(f"The corresponding confound file '{confound_file}' is not found in the directory.")

    try:
        confounds = load_confounds(img_path, strategy=strategy, motion=motion, wm_csf=wm_csf, global_signal=global_signal, demean=demean)
        return confounds
    except ValueError as e:
        print(f"Error: {e}")
        return None

def create_masker(atlas_filename, standardize="zscore_sample", standardize_confounds="zscore_sample", memory="nilearn_cache"):
    masker = NiftiLabelsMasker(
        labels_img=atlas_filename,
        standardize=standardize,
        standardize_confounds=standardize_confounds,
        memory=memory
    )
    return masker

def extract_time_series(atlas_filename, img_path, confounds_df, masker=None):
    if masker is None:
        masker = create_masker(atlas_filename)
    
    time_series = masker.fit_transform(img_path, confounds=confounds_df)
    return time_series

def compute_correlation_matrix(time_series):
    correlation_measure = ConnectivityMeasure(kind="correlation", standardize="zscore_sample")
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    np.fill_diagonal(correlation_matrix, 0)
    return correlation_matrix

def load_and_process_data(subject_dir, session, directory_path, confound_suffix='_desc-confounds_timeseries.tsv'):
    img_file = f'sub-{subject_dir}_ses-{session}_task-rest_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
    confounds_data = load_and_process_confounds(subject_dir, session, directory_path, img_file, confound_suffix=confound_suffix)
    img_path = os.path.join(directory_path, f'sub-{subject_dir}', f'ses-{session}', 'func', img_file)
    time_series = extract_time_series(atlas_filename, img_path, confounds_data[0])
    correlation_matrix = compute_correlation_matrix(time_series)
    return correlation_matrix

# Specify the new directory path
directory_path = "/Volume/SubhaWork/fmriprep/"

# Load the BASC atlas with resolution 64
basc_atlas = datasets.fetch_atlas_basc_multiscale_2015(resolution=64)
atlas_filename = basc_atlas['maps']

# Define the sessions
sessions = ['01', '02', '03']

# Detect available subjects in the new location
available_subjects = [f"{i:03d}" for i in range(1, 156) if os.path.exists(os.path.join(directory_path, f'sub-{i:03d}'))]
subject_corr_matrices = []
# Iterate through available subjects and sessions
for subject in available_subjects:
    for session in sessions:
        try:
            subject_corr_matrix = load_and_process_data(subject, session, directory_path)
            subject_corr_matrices.append(subject_corr_matrix)
            # Do something with the correlation matrix, e.g., save or analyze it
        except Exception as e:
            print(f"Error processing subject {subject}, session {session}: {e}")

# Store the correlation matrices in a list


# Iterate through subjects and sessions
for i, subject in enumerate(available_subjects):
    for j, session in enumerate(sessions):
        # Get the index in the flat list for the current subject and session
        index = i * len(sessions) + j

        # Get the correlation matrix for the current subject and session
        corr_matrix = subject_corr_matrices[index]

        # Convert the 2D matrix to a DataFrame
        df = pd.DataFrame(corr_matrix)

        # Save the DataFrame to a CSV file without row and column labels
        output_dir = "/Volume/SubhaWork/fmriprep/correlation_matrix_output"  # Replace with your desired directory
        os.makedirs(output_dir, exist_ok=True)

        csv_file_path = os.path.join(output_dir, f'sub-{subject}_ses-{session}_correlation_matrix.csv')
        df.to_csv(csv_file_path, index=False, header=False)

