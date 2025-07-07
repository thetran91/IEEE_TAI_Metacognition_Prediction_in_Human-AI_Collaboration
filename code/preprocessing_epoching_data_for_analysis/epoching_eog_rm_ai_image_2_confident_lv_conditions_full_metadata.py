import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd
from mne.preprocessing import ICA, create_eog_epochs
from autoreject import AutoReject  # If you don't have this package, install it first

######## IMPORTANT - REQUIRE CHANGE VALUE #########

T_MIN = -0.5
T_MAX = 2.0  # Only Target searching is 3.0s

event_code = 9 # Event 18 - 2s
event_folder_to_save_epoch = 'AI_image_2_decision_making'

# Define all behavior columns that should be included as metadata
behaviour_columns = [
    'Name', 'Round', 'SecondsTarget2', 'ConfidentLevel', 'SecondsConfident', 'Score', 
    'PlayerTarget2', 'PlayerTarget1', 'SecondsTarget1', 'Images', 'Target Position', 
    'Correct_1', 'Correct_2', 'AI Position 1', 'AI Position 2', 'AI1 Correct', 'AI2 Correct'
]

###################################################

def load_data_file(fname):
    base_path = "Data/Behaviour"
    return [f"{base_path}/{fname}/Test {i}.csv" for i in range(5, 6)]  # Loading behavior data for Test 5

def perform_ica_and_remove_eye_artifacts(raw):
    # Set up ICA with the number of components equal to the number of EEG channels
    n_channels = len(raw.info['ch_names'])  # Get the number of EEG channels
    ica = ICA(n_components=n_channels, random_state=97, max_iter='auto')
    
    # Fit ICA to the raw data
    ica.fit(raw)

    # Detect eye artifacts (blinks) using EOG channels, or if no EOG channels are available, use frontal channels
    try:
        eog_epochs = create_eog_epochs(raw, reject_by_annotation=True)
        eog_inds, scores = ica.find_bads_eog(eog_epochs)
    except RuntimeError:  # If no EOG channel is present, use frontal channels (e.g., Fp1, Fp2)
        eog_inds, scores = ica.find_bads_eog(raw, ch_name=['FP1', 'FP2'])

    # Mark the eye-related components for exclusion
    ica.exclude = eog_inds

    # Apply ICA to remove the eye artifacts
    ica.apply(raw)

    return raw

def plot_data_before_after(raw, epochs, epochs_clean):
    # Plot raw data before any preprocessing
    raw.plot(n_channels=10, scalings='auto', title='Raw Data Before Preprocessing', show=True)

    # Plot original epochs
    epochs.plot(n_channels=10, scalings='auto', title='Epochs Before Preprocessing', show=True)

    # Plot cleaned epochs after preprocessing and autorejection
    epochs_clean.plot(n_channels=10, scalings='auto', title='Epochs After Preprocessing & Autorejection', show=True)

def process_subject(subject_id):
    # Load the EEG data
    data_path = f'Data/Test_5_set/{subject_id}_Test 5_Cleaned.set'
    raw = mne.io.read_raw_eeglab(data_path, preload=True)

    # Step 0: Remove non-EEG channels
    channels_to_remove = ['M1', 'M2', 'CB1', 'CB2', 'HEO', 'VEO', 'EKG', 'EMG', 'TRIGGER']
    raw.drop_channels([ch for ch in channels_to_remove if ch in raw.ch_names])

    # Step 1: Plot the raw data
    raw.plot(n_channels=10, scalings='auto', title=f'Raw Data: {subject_id}', show=True)

    # Step 2: High-pass and Low-pass Filtering (1-Hz to 100-Hz)
    raw.filter(l_freq=1., h_freq=100.)

    # Step 3: Remove line noise (typically 50 or 60 Hz depending on your region)
    raw.notch_filter(freqs=50)

    # Step 4: Perform ICA to remove eye-related artifacts
    raw_clean = perform_ica_and_remove_eye_artifacts(raw)

    # Load the behavioral data (Test 5)
    SS_behavior_data = load_data_file(subject_id)
    all_behaviour_df_raw = pd.concat([pd.read_csv(f) for f in SS_behavior_data])
    all_behaviour_df = all_behaviour_df_raw

    # Data cleaning and type conversion for relevant columns
    for col in behaviour_columns:
        all_behaviour_df.loc[all_behaviour_df[col] == 'x', col] = 0  # Handle 'x' values in the behavior data
        if col not in ['Name', 'Images']:  # Skip string columns
            all_behaviour_df[col] = all_behaviour_df[col].astype(float)

    # Extract events from the cleaned EEG data
    events, _ = mne.events_from_annotations(raw_clean)

    # Filter by the target event code (AI_image_1_shown event)
    events = events[np.in1d(events[:, 2], (event_code)), :]  # Select by event code

    if len(events) == 0:
        print(f"No events found for event code {event_code}. Please check the event code or .set file.")
        return

    tmin, tmax = T_MIN, T_MAX
    baseline = (-0.2, 0)

    # Epoch the cleaned data based on the event code
    epochs = mne.Epochs(raw_clean, events, [event_code], tmin, tmax, baseline=baseline, preload=True)

    # Step 6: Add metadata before autoreject (include all behavior columns)
    META_DF = all_behaviour_df[behaviour_columns].iloc[:len(epochs)].reset_index(drop=True)

    # Check for mismatch in epoch and metadata count, and remove the last epoch if needed
    if len(epochs) > len(META_DF):
        print(f"Mismatch detected: {len(epochs)} epochs but {len(META_DF)} metadata entries. Removing the last epoch.")
        epochs = epochs[:-1]  # Remove the last epoch to match the behavior trial count

    if len(epochs) != len(META_DF):
        raise ValueError(f"Mismatch: {len(epochs)} epochs but {len(META_DF)} metadata entries.")

    epochs.metadata = META_DF  # Attach all behavioral data as metadata

    # Step 7: Autorejection - Automatically reject bad epochs
    ar = AutoReject(random_state=42)
    epochs_clean = ar.fit_transform(epochs)

    # Define save directories for low, medium, and high confidence levels
    low_confidence_save_path = f"Data/Epoched/{event_folder_to_save_epoch}_eog_rm_confident_lv_full_metadata/Low_Confidence"
    medium_confidence_save_path = f"Data/Epoched/{event_folder_to_save_epoch}_eog_rm_confident_lv_full_metadata/Medium_Confidence"
    high_confidence_save_path = f"Data/Epoched/{event_folder_to_save_epoch}_eog_rm_confident_lv_full_metadata/High_Confidence"

    # Check and create directories if they don't exist
    for path in [low_confidence_save_path, medium_confidence_save_path, high_confidence_save_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    # Select and save Low confident epochs
    low_confident_epochs = epochs_clean[epochs_clean.metadata['ConfidentLevel'].isin([0, 1])]
    low_confident_epochs.save(f"{low_confidence_save_path}/{subject_id}-low-confident-epo-{event_code}.fif", overwrite=True)

    # Select and save Medium confident epochs
    medium_confident_epochs = epochs_clean[epochs_clean.metadata['ConfidentLevel'].isin([2, 3])]
    medium_confident_epochs.save(f"{medium_confidence_save_path}/{subject_id}-medium-confident-epo-{event_code}.fif", overwrite=True)

    # Select and save High confident epochs
    high_confident_epochs = epochs_clean[epochs_clean.metadata['ConfidentLevel'].isin([4, 5])]
    high_confident_epochs.save(f"{high_confidence_save_path}/{subject_id}-high-confident-epo-{event_code}.fif", overwrite=True)


# List of subjects to process (you can modify this list as needed)
subjects = ['SS04', 'SS05', 'SS06', 'SS07', 'SS08', 'SS09', 'SS10', 'SS11', 'SS12', 'SS13', 'SS14', 'SS15', 'SS16', 'SS17']

for subject in subjects:
    process_subject(subject)
