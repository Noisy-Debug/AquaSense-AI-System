# 🧠 Employee Attrition Prediction System
<p align="justify">
This project uses <strong>deep learning and machine learning</strong> models to predict whether water is <strong>safe (potable)</strong> for drinking based on physicochemical parameters. It supports <strong>Sustainable Development Goal 6 (SDG 6)</strong> – Clean Water and Sanitation – by enabling low-cost, scalable, and automated water quality assessment. The system includes a <strong>Gradio web application</strong> and <strong>Google Gemini AI</strong> integration for real-time intelligent insights.
</p>

# 📊 Problem Statement
<p align="justify">
Traditional water quality testing is expensive, manual, and inaccessible in rural or resource-limited regions. This project aims to <strong>automate water potability prediction</strong> using open datasets and AI techniques to enable <strong>rapid, affordable decision-making</strong> for safe water consumption.
</p>

# 🔍 Key Features
- **5 Models Compared**: XGBoost, MLP, CNN, DNN, and ResNet.  
- **Best AUC**: ResNet (0.790).  
- **Best Accuracy/F1**: DNN (Accuracy: 71.8%, F1: 0.72).  
- **Data Balancing**: Used SMOTE for handling class imbalance.  
- **Gradio UI**: Web app interface for real-time testing.  
- **Gemini AI Integration**: Get smart, natural language feedback on model performance.

# 💡 Technologies Used
- Python, Pandas, NumPy, Scikit-learn.
- TensorFlow / Keras, XGBoost, Imbalanced-learn.
- Gradio (for UI).
- Google Generative AI (Gemini).
- Matplotlib / Seaborn (for evaluation charts).

# 📦 Repository Contents
| Folder/File     | Description |
|-----------------|-------------|
| `DATA/`           | Cleaned dataset sourced from Kaggle. |
| `REQUIREMENTS.txt`| List of Python dependencies. |
| `SRC/`            | Jupyter notebook with model training evaluation, and UI integration. |
| `REPORT/`         | Detailed technical report with results and architecture. |

# 🚀 Running the App Locally
```bash
# 1. Clone the repository
git clone https://github.com/your-username/Water-Potability-Prediction.git

# 2. Install required packages
pip install -r REQUIREMENTS.txt

# 3. Launch the notebook
jupyter 2-SRC/Predictive App AI.ipynb
```

# 🌐 Try It on Hugging Face
Don't want to run it locally? Try the live demo here: https://huggingface.co/spaces/Noisy-Debug/Water-Potability-Prediction-App

# 🤖 Gemini AI Integration
Real-time prompt examples:

- “Explain which model performs best and why”
- “Suggest improvements for weak classifiers”
- “Highlight top influential features in the prediction”

# 📊 Results and Insights
- **DNN Model**: Highest accuracy (71.8%) and balanced recall/F1.  
- **ResNet**: Top AUC score (0.790), showing best discrimination across classes.  
- **XGBoost**: Best among traditional ML models, efficient on tabular data.  
- **Impactful Features**: pH, Sulphates, Conductivity, Chloramines.  
- **Balanced Data**: SMOTE significantly improved model stability and recall.

# 🔮 Future Enhancements
- Deploy the app on Hugging Face or Streamlit Cloud.  
- Integrate Explainable AI (SHAP or LIME) for interpretability.  
- Expand training on multi-source water quality datasets.  
- Improve performance on minority (unsafe water) class detection.
