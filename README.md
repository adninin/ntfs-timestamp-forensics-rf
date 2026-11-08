TFS Timestamp Manipulation Detection: Random Forest and SHAP

**Code for the paper** "Random Forest and SHAP for Detecting Designated NTFS Timestamp Manipulation Targets within a NIST SP 800-86-Aligned Workflow" (Adnin Atmadewati Ashrini, Telkom University).

Python scripts for parsing NTFS $MFT and $UsnJrnl (MFTECmd), building the dataset, running Leave-One-Scenario-Out Random Forest experiments, and generating SHAP explanations and figures.

**Data**
Raw artifacts are not included. The corpus comes from Oh, Lee, and Hwang, IEEE Access, vol. 12, 2024, doi: 10.1109/ACCESS.2024.3395644.

**Requirements**
Python 3.13.3, pandas 3.0.6, NumPy 2.5.3, scikit-learn 1.9.1, SHAP 0.52.0, Matplotlib 3.11.2, MFTECmd 2026.5.0.

**Note**
Results describe model behavior on a controlled corpus with twelve designated targets and are decision support only, not independent evidence of timestamp manipulation.
