# 🌦️ Atmospheric Early Warning System

**Live Dashboard:** [Launch the BPSO-SVM Engine](https://svm-pso-weather-forecast-engine.streamlit.app/)

## About This Project
Welcome to our repository. We are a group of undergraduate students, and this repository contains the complete computational architecture for our final-year B.Sc. technical project. 

Our core objective was to build a highly optimized meteorological prediction engine. Weather forecasting is a complex, high-dimensional problem. Instead of blindly feeding all available data into a machine learning model, we wanted to see if we could mimic natural biological behaviors to mathematically prune the data first, making the final prediction faster and more accurate.

## Our Methodology: The BPSO-SVM Framework
To achieve this, we developed a hybrid mathematical framework combining **Binary Particle Swarm Optimization (BPSO)** with a **Support Vector Machine (SVM)**.

We utilized historical Australian meteorological data as our rigorous experimental proxy. 
1. **The Swarm (BPSO):** We implemented a swarm intelligence algorithm inspired by how flocks of birds search for food. The "particles" navigate the feature space to find the most critical weather attributes, successfully pruning the dataset down to just 14 active dimensions.
2. **The Classifier (SVM):** Once the optimal features were isolated, we mapped them into a high-dimensional space using a maximum-margin Support Vector Machine to classify the atmospheric geometry as either "Rain" or "Clear Skies."

## Repository Architecture
We built this system with a strict separation of concerns to ensure engineering robustness:

* `preprocess.py`: Our ETL pipeline. Handles the ingestion, missing value imputation, and mathematical scaling (StandardScaler/LabelEncoder) of the raw data.
* `bpso_svm.py`: The theoretical heart of the research, containing the custom metaheuristic swarm optimization logic.
* `train_model.py`: The execution script that runs the swarm, isolates the final features, trains the SVM, and exports the `.pkl` artifacts.
* `app.py`: The final presentation layer. A custom-styled, coral-themed Streamlit dashboard built for our defense presentation.

## Local Execution Protocol
If you wish to run our mathematical engine on your local machine, follow this sequence:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Davey117/SVM-PSO-Weather-Forecast.git](https://github.com/Davey117/SVM-PSO-Weather-Forecast.git)
   cd SVM-PSO-Weather-Forecast
