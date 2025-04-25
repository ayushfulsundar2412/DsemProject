🫀 Cardiovascular Disease Risk Prediction

📌 Overview
This project aims to predict the risk of cardiovascular disease (CVD) by analyzing longitudinal Electronic Health Records (EHR) using both traditional machine learning models (like Logistic Regression and Random Forest) and deep learning models (like LSTM). The goal is to assess how historical patient data can enhance prediction accuracy.

🗂️ Project Structure
kotlin
Copy
Edit
├── data/
│   ├── benchmark/
│   │   └── data.csv
│   └── temporal/
│       └── data.csv
├── models/
│   ├── benchmark_model.py
│   └── lstm_temporal_model.py
├── notebooks/
│   └── exploratory_analysis.ipynb
├── outputs/
│   ├── cvd_results.csv
│   └── model_performance_plots/
├── requirements.txt
└── README.md
⚙️ Setup Instructions
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/cvd-risk-prediction.git
cd cvd-risk-prediction
2. Create a Virtual Environment
It's recommended to use a virtual environment to manage dependencies.

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
Install the required Python packages using requirements.txt.

bash
Copy
Edit
pip install -r requirements.txt
4. Download the Dataset
Ensure that the data/benchmark/data.csv and data/temporal/data.csv files are placed in the respective directories. If these files are not available, please contact the project maintainer or refer to the data acquisition instructions.

🧪 Running the Models
Benchmark Models (Random Forest & Logistic Regression)
bash
Copy
Edit
python models/benchmark_model.py
This script will train the benchmark models and output performance metrics, saving results to outputs/cvd_results.csv.

LSTM Temporal Model
bash
Copy
Edit
python models/lstm_temporal_model.py
This script will train the LSTM model on temporal data and display evaluation metrics and plots.

📊 Results
Logistic Regression: Achieved an accuracy of 91.02% and ROC-AUC of 0.7517 on the benchmark dataset.

Random Forest: Achieved an accuracy of 90.50% and ROC-AUC of 0.7403 on the benchmark dataset.

LSTM Model: Demonstrated improved performance on temporal data, capturing sequential patterns in patient history.

Note: Detailed performance plots are available in the outputs/model_performance_plots/ directory.

📈 Visualizations
The project includes various plots to aid in understanding model performance:

Confusion matrices

ROC curves

Training vs. validation accuracy/loss graphs

These can be found in the outputs/model_performance_plots/ directory.

📝 Dependencies
The project relies on the following Python packages:

pandas

numpy

scikit-learn

tensorflow

matplotlib

seaborn

All dependencies are listed in requirements.txt.

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

🙌 Acknowledgements
Inspired by research on utilizing longitudinal EHR data for disease prediction.

Thanks to the contributors and the open-source community for their invaluable resources.
