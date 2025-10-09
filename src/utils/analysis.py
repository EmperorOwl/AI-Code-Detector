import os
from logging import Logger

import pandas as pd


def save_predictions(logger: Logger,
                     output_df: pd.DataFrame,
                     model_name: str,
                     file_path: str) -> None:
    prediction_col = f'{model_name}_Prediction'
    confidence_col = f'{model_name}_Confidence'
    prediction_type_col = f'{model_name}_Prediction_Type'

    # Check if file already exists
    file_exists = os.path.exists(file_path)

    if file_exists:
        # Load existing file and append new model predictions
        df = pd.read_csv(file_path)
        assert df['ID'].equals(output_df['ID']), "ID columns do not match"
        df[prediction_col] = output_df['Predicted_Label']
        df[confidence_col] = output_df['Confidence']
        log_msg = f"Predictions appended to {file_path}"
    else:
        # Create new file, drop unnecessary columns, and rename columns
        df = output_df.copy()
        df = df.drop(columns=['input_ids', 'attention_mask'])
        df = df.rename(columns={
            'Predicted_Label': prediction_col,
            'Confidence': confidence_col
        })
        log_msg = f"Predictions saved to {file_path}"

    # Determine prediction type (TP, FP, TN, FN)
    prediction_types = []
    for _, row in output_df.iterrows():
        true_label = row['Label']
        predicted_label = row['Predicted_Label']

        if true_label == 1 and predicted_label == 1:
            # True Positive (correctly identified AI)
            prediction_types.append('TP')
        elif true_label == 0 and predicted_label == 1:
            # False Positive (human labeled as AI)
            prediction_types.append('FP')
        elif true_label == 0 and predicted_label == 0:
            # True Negative (correctly identified human)
            prediction_types.append('TN')
        else:
            # False Negative (AI labeled as human)
            prediction_types.append('FN')

    # Add prediction type column
    df[prediction_type_col] = prediction_types

    # Save to CSV
    df.to_csv(file_path, index=False)

    # Log message
    logger.info(log_msg + "\n")
