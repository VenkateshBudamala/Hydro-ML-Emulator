# 🌊 **Hydro-ML Emulator**  
**Unifying Hydrologic and Hydraulic Models Using a Machine Learning-Based Emulator**  

---

## **🔑 Overview**  

This project provides a **unified framework** to integrate and optimize hydrologic and hydraulic models using **machine learning**. It uses:  
- **XGBoost** to create an emulator.  
- **Genetic Algorithm (GA)** for optimization.  

The framework simplifies the calibration and analysis of diverse models like **SWAT**, **RAVEN**, **SUMMA**, and **HEC-RAS**, improving efficiency and reducing computational time.

---

## **📋 Quick Start Guide**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/yourusername/Hydro-ML-Emulator.git
cd Hydro-ML-Emulator
```

### **2. Install Dependencies**  
Ensure Python 3.8+ is installed. Then run:  
```bash
pip install -r requirements.txt
```

### **3. Set Your Inputs**  
Update `Input_File.py` to define the parameters for your simulation (e.g., rainfall, flow rates, etc.).  
This is the only file you need to modify.  

### **4. Run the Emulator**  
Execute the main script:  
```bash
python Main_Setup.py
```
Or, press **F5** in your Python IDE.  

### **5. View Results**  
Simulation outputs, including logs and visualizations, are saved in the `outputs/` directory.  

---

## **📂 Repository Structure**  

```
Hydro-ML-Emulator/
│
├── Main_Setup.py       # Main execution file
├── Input_File.py       # Customizable input parameters
├── Functions.py        # Core utility functions
├── requirements.txt    # Python dependencies
├── outputs/            # Auto-generated results
└── README.md           # Project documentation
```

---

## **🔧 Key Features**  

- **Automation**: No manual adjustments required—just set inputs and run.  
- **Efficiency**: Reduces computational time for model calibration.  
- **Flexibility**: Supports diverse hydrologic and hydraulic models.  

---

## **💡 Applications**  

This framework is ideal for:  
- Flood forecasting and mapping.  
- Hydrologic and hydraulic model calibration.  
- Urban flood management.  

---

## **📞 Contact**  

For questions or collaboration, feel free to reach out:  
- **Name**: Dr. Venkatesh Budamala 
- **Email**: bvenkatesh@iisc.ac.in 
- **GitHub**: [@VenkateshBudamala](https://github.com/VenkateshBudamala)  

---

This version provides simple, clear instructions for new users while emphasizing ease of use and practical applications. Let me know if you’d like further changes!
