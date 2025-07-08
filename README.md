# 📊 Data Validation Panel

A smart **data validation tool** built as an add-on feature for enterprise software. This application allows users to upload data files (CSV or Excel), automatically detects inconsistencies, highlights them, and enables in-panel editing before pushing the cleaned data back to the main software.

---

## 🔍 Features

- ✅ Accepts CSV and Excel files as input
- ⚠️ Automatically detects data inconsistencies (e.g., missing values, invalid formats)
- ✏️ Interactive panel to view and edit errors inline
- 🔁 Seamless push of updated data to the main software system
- 💡 Enhances data quality and reduces manual preprocessing effort

---

## 🚀 How It Works

1. **Upload** your dataset (CSV or Excel)
2. **Review** highlighted inconsistencies
3. **Edit** directly within the validation panel
4. **Push** the cleaned data to your connected software

---

## 🛠️ Tech Stack

- **Frontend**: React.js 
- **Backend**: Python (Pandas, Openpyxl, etc.) + Flask 
- **File Handling**: CSV, Excel (XLSX)
- **Integration**: API connection to main software 

---

## 📂 Usage

```bash
# Clone the repository
git clone https://github.com/MalaiarasuGRaj/Super-app-final.git

# Navigate to the folder
cd Super-app-final

# Run the app (Streamlit example)
streamlit run app.py
