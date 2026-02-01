# Diabetes Risk Prediction System  
### From raw health data to real-world medical risk screening

---

## Why this project exists

Diabetes is often silent. Millions of people remain undiagnosed until complications appear.  
This project explores a practical question: **Can basic lifestyle and health indicators help identify diabetes risk early?**

Using real CDC survey data and a carefully designed machine learning pipeline, this system predicts whether a person is at **low or high risk of diabetes** and presents the result through a **clean, premium web interface**, similar to how a real healthcare screening tool would behave.

---

## What makes this project different

Most machine learning projects stop at “model trained”. This one goes further.

- Real public health dataset (CDC BRFSS)
- Medical-safe decision logic (recall-focused)
- Probability-based risk screening
- End-to-end system (model + backend + frontend)
- Premium dark-themed UI (not a notebook demo)
- Cross-environment validation (Colab → local deployment)

This is **applied machine learning**, not a tutorial project.

---

##  System overview

User enters health indicators in the web interface.  
The backend processes the inputs and passes them to the trained ML model.  
The model outputs a risk probability, which is converted into a human-readable medical risk category.

User Input → Flask Backend → ML Model → Risk Probability → Risk Decision


---

##  Dataset

- Source: CDC Behavioral Risk Factor Surveillance System (BRFSS 2015)
- Records: 253,680 individuals
- Features: 21 health and lifestyle indicators
- Target: Diabetes risk (binary)

Example features include BMI, blood pressure, cholesterol status, physical activity, smoking habits, age, education, and income.

This dataset reflects real-world class imbalance and noisy health data.

---

##  Machine learning approach

- Problem type: Binary classification (risk screening)
- Model: Gradient Boosting Classifier

### Why Gradient Boosting?
- Handles non-linear relationships effectively
- Performs well on structured medical data
- Scale-invariant (no feature scaling required)

### Medical decision logic
Instead of using a default 0.5 threshold, the system flags risk at **probability ≥ 0.30** to reduce false negatives, which is critical in healthcare applications.

---

##  Web application

The web interface is designed to look like a real healthcare SaaS product rather than a student demo.

- Two-column professional layout
- Labeled inputs for all 21 features
- Premium dark theme
- Clear risk output with probability score
- Backend powered by Flask
- Frontend built with HTML, CSS, and JavaScript

---

## Application preview

Screenshots of the UI and predictions are available in the `screenshots/` folder:
- Home interface
- Low risk prediction
- High risk prediction

---

##  Example predictions

### Low risk case
- BMI: 22
- Physically active
- No blood pressure or cholesterol issues

Output:
- Risk probability ≈ 0.01–0.10
- Prediction: Low Risk of Diabetes

### High risk case
- BMI: 34
- High blood pressure and cholesterol
- Low physical activity

Output:
- Risk probability ≈ 0.60–0.80
- Prediction: High Risk of Diabetes

Results are consistent between training (Colab) and deployment environments.

---

##  How to run locally

```bash
git clone https://github.com/vijayendravarma111/Diabetes-Risk-Prediction-System.git
cd Diabetes-Risk-Prediction-System
pip install -r requirements.txt
python app.py

