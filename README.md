Audio Classification Project

This project focuses on **classifying audio signals** using advanced feature extraction and **machine learning models**. It processes a dataset of spoken digits and trains a model to recognize and classify them based on a rich set of features derived from time-domain, frequency-domain, and Mel Frequency Cepstral Coefficients (MFCCs) analysis.

- Advanced Audio Feature Extraction:
  - Statistical measures (mean, standard deviation, skewness, kurtosis).
  - Frequency-domain features (FFT-based power spectrum).
  - MFCCs for speech signal analysis.

- Machine Learning Models:
  - Trained with models like XGBoost Classifier for high accuracy.
  - Normalization and scaling for improved model performance.

- Data Preprocessing:
  - Handles audio signal normalization.
  - Pads or truncates signals to ensure consistent length.

- Evaluation and Metrics:
  - Includes metrics like F1 Score, Accuracy and confusion matrix visualization.

 Technologies Used

- Python: Core language for implementation.
- Librosa: For audio processing and MFCC extraction.
- Scikit-learn: For preprocessing, feature scaling, and model evaluation.
- XGBoost: For training high-performance classification models.
- Matplotlib: For visualizing confusion matrices and other results.

How It Works

1. Load the Dataset:
   - The project uses a dataset of spoken digits (e.g., "0", "1", ..., "9").
   - Audio files are in `.wav` format.

2. Preprocessing:
   - Signals are normalized and padded/truncated to a consistent length.
   - Extracted features include:
     - Statistical: mean, standard deviation, skewness, kurtosis.
     - Frequency-domain: FFT-based power spectrum analysis.
     - MFCCs: Captures essential characteristics of speech.

3. Training:
   - Models are trained on extracted features using **XGBoost Classifier**.
   - Hyperparameters are tuned for optimal performance.

4. Evaluation:
   - Performance metrics include F1 Score, Accuracy, and Confusion Matrix.
   - Validation and test sets are used to evaluate generalization.

5. Prediction:
   - The trained model predicts the digit from unseen audio samples.

How to run
1. Clone the Repository**:
   ```bash
   git clone https://github.com/SpaceDevEngineer/audio-classification-project.git
   cd audio-classification-project
   ```

2. Install Dependencies:
   Create a virtual environment and install the required packages:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate     # For Windows
   pip install -r requirements.txt
   ```

3. Run the Script:
   Execute the main script to train the model and generate predictions:
   ```bash
   python audio-classification-project.py
   ```

4. Output:
   - The script saves the predictions in `submission.csv`.
   - Visualizes a confusion matrix to evaluate model performance.
