# Adaptive SimHash Reproduction Project

This project is designed to reproduce the results presented in the associated research paper. The main directory contains the source files, while the necessary data files are provided in a separate compressed archive for manageability due to their large size.

The code structure is as shown in the file tree below. The `data_process` folder contains code related to data preprocessing. The `DateBase` section includes content for constructing the approximate database. The `model` folder holds the weights of our trained models. The following three index folders correspond to different indexing methods and image plotting as described in the paper. The `utils` folder contains utility functions. `main_tree` represents the concept tree, and `requirements` lists all the dependencies required for the project.

├── data_process
│   ├── data_filter.py
│   ├── data_process_structured.py
│   ├── data_process_unstrucutred.py
│   ├── filter_and_process_unstructured.py
│   ├── Max_Min_Statistic.py
│   └── Random_Select.py
├── DataBase
│   ├── SimHash.py
│   ├── Simple_Tree.py
│   └── WeightMapping.py
├── model
│   ├── best_model.pth
│   └── model_all_data_2024-11-20-17-59.pth
├── Structured_Adaptive_Weighting_Retrieval
│   ├── plot_retrieval.py
│   ├── time_statistic_Brute.py
│   └── time_statistic_Hash.py
├── Structured_Eucidean_Reterival
│   ├── MLP_Training.py
│   └── Structured_Eucidean_Retrieval.py
├── Unstrucured_Adaptive_Weighting_Retrieval
│   ├── plot_retrieval.py
│   └── plot_three_methods.py
├── utils
│   ├── embedding_achieve.py
│   ├── plot_before_after.py
│   ├── result_process_structured.py
│   └── result_process_unstructured.py
├── main_tree.py
└── requirements.txt

## Prerequisites

- **Python 3.x** (tested with Python 3.11)
- Required Python libraries should be installed. 

```
pip install -r requirements.txt
```

## Installation

### 1. Download the Data Archive

- Navigate to the release area and download the `Adaptive_Weighting_SimHash_Code_v3.zip` file.

### 2. Extract the Archive

- Unzip the downloaded `Adaptive_Weighting_SimHash_Code_v3.zip` file into a local directory of your choice.
- Note the path to this directory as it will be required to run the scripts.

## Usage

### 1. Navigate to the Unzipped Directory

- Open a terminal or command prompt.
- Change the working directory to the location where you unzipped the `Adaptive_Weighting_SimHash_Code_v3.zip` file.

### 2. Run the Python Scripts

- Execute the provided Python scripts by following the instructions in the scripts or as described in the associated research paper.
- The scripts are designed to process the data and reproduce the results presented in the research.

## Notes

- For any issues or further information, refer to the associated research paper or the comments in the source code.
- Ensure that all necessary Python libraries are installed before running the scripts.