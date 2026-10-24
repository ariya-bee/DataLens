# DataLens — Data Analysis Studio

A Python-based data analysis and machine learning application built with Streamlit.

DataLens helps users explore datasets, identify data quality issues, clean data, visualize patterns, compare machine learning models, and generate downloadable reports through a web-based interface.

## Features

### 1. Exploratory Data Analysis

* View dataset dimensions and column information.
* Explore descriptive statistics.
* Identify missing values and duplicate records.
* Visualize distributions, outliers, correlations, and category counts.
* Generate observations from the dataset.

### 2. Data Cleaning

* Remove duplicate rows.
* Handle missing values using supported cleaning methods.
* Preview cleaned data before downloading it.
* Export the cleaned dataset as a CSV file.

### 3. Machine Learning

* Select a target column and input features.
* Compare supported machine learning models.
* View evaluation metrics and prediction results.
* Explore confusion matrices or actual-versus-predicted plots, depending on the task.
* Examine feature importance or coefficient magnitudes where supported.

### 4. Downloadable Reports

* Export analysis results in supported formats.
* Download dataset column profiles.
* Generate an HTML analysis report.

## Technology Stack

* Python
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* OpenPyXL

## Getting Started

### Prerequisites

* Python 3.10 or a compatible version.
* pip, Python's package installer.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ariya-bee/DataLens.git
   ```

2. Move into the project directory:

   ```bash
   cd DataLens
   ```

3. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

4. Activate the environment on Windows:

   ```bash
   .venv\Scripts\activate
   ```

5. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

6. Start the application:

   ```bash
   streamlit run app.py
   ```

7. Open the local address shown in your terminal.

## How to Use

1. Launch DataLens.
2. Upload a CSV dataset.
3. Explore its statistics and visualizations.
4. Review data quality issues.
5. Apply cleaning operations if needed.
6. Configure and compare machine learning models when the dataset is suitable.
7. Download the available analysis results and reports.

## Project Structure

```text
DataLens/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── docs/
    └── PROJECT_DOCUMENTATION.md
```

## Limitations

* Results depend on the quality and suitability of the uploaded dataset.
* Machine learning evaluation depends on the selected target, input features, and available data.
* Model performance on a dataset does not guarantee the same performance on unseen real-world data.

## Future Improvements

* Add more visualization options.
* Expand supported machine learning algorithms.
* Improve automated data quality recommendations.
* Add more report customization options.
* Expand testing and documentation.

## Author

**Ariya Bee**

Computer Science and Engineering Student

---

*DataLens is a student project developed for learning and demonstrating data analysis and machine learning workflows.*

