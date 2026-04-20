# 🛍️ ShopIQ — AI Customer Segmentation Engine

![ShopIQ Banner](https://img.shields.io/badge/ShopIQ-Customer%20Segmentation-A3E635?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![sklearn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> **Live Demo** → [shopiq.streamlit.app](https://shopiq.streamlit.app) *(replace with your deployed URL)*

A production-grade **unsupervised machine learning web app** that segments mall customers into 5 distinct groups using K-Means clustering, detects anomalies using Isolation Forest, and delivers actionable business recommendations — all in a sleek dark-themed dashboard.

---

## 📸 Screenshots

| Segments View | EDA View | Anomaly Detection |
|---|---|---|
| *(add screenshot)* | *(add screenshot)* | *(add screenshot)* |

---

## 🧠 What it does

ShopIQ takes raw customer data (age, income, spending score) and automatically:

1. **Explores the data** — distributions, correlations, gender breakdown
2. **Compresses features** — StandardScaler normalisation before clustering
3. **Finds the optimal K** — Elbow method plotted interactively
4. **Segments customers** — K-Means identifies 5 distinct customer types
5. **Flags anomalies** — Isolation Forest catches 10 unusual customers
6. **Gives business recommendations** — actionable marketing strategy per segment

---

## 🎯 The 5 Customer Segments

| Segment | Income | Spending | Strategy |
|---|---|---|---|
| ⭐ Premium Targets | High (~87k) | High (~82) | VIP program, premium brands |
| 🛒 Impulsive Buyers | Low (~26k) | High (~79) | Flash sales, EMI, loyalty points |
| 👤 Average Joes | Mid (~55k) | Mid (~50) | Seasonal offers, bundle deals |
| 💼 Rich but Reluctant | High (~88k) | Low (~17) | Premium events, exclusive experiences |
| 🪙 Careful Savers | Low (~26k) | Low (~21) | Discounts, clearance sales |

> **Key insight:** Income vs Spending correlation = **0.01** — wealthy customers do NOT automatically spend more. This is why segmentation is non-trivial and genuinely valuable.

---

## 🛠️ Tech Stack

```
Python 3.10+
├── streamlit          — web app framework
├── scikit-learn       — KMeans, IsolationForest, StandardScaler, PCA
├── pandas             — data manipulation
├── numpy              — numerical computing
├── matplotlib         — plotting
└── seaborn            — heatmaps and styled charts
```

---

## 🤖 Algorithms Used

### K-Means Clustering
- Groups 200 customers into K segments
- Elbow method used to select optimal K=5
- `n_init=10` for stable results

### Isolation Forest
- Detects anomalous customers that don't fit any segment
- `contamination=0.05` (expects ~5% outliers)
- Output: 1 = normal, -1 = anomaly

### StandardScaler
- Applied before K-Means — mandatory
- Prevents income (large numbers) dominating spending score (small numbers)

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/shopiq.git
cd shopiq
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
shopiq/
│
├── app.py                    ← Main Streamlit application
├── Mall_Customers.csv        ← Default dataset (200 customers)
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
│
└── assets/                   ← (optional) screenshots for README
    ├── screenshot_segments.png
    ├── screenshot_eda.png
    └── screenshot_anomalies.png
```

---

## 📊 Dataset

**Mall Customer Segmentation Data**
- Source: [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- 200 customers, 5 columns
- Features: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)
- 0 missing values — clean dataset

You can also **upload your own CSV** through the sidebar — as long as it has `Annual Income` and `Spending Score` columns.

---

## ☁️ Deploy on Streamlit Cloud (Free)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit — ShopIQ Customer Segmentation App"
git branch -M main
git remote add origin https://github.com/yourusername/shopiq.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app**
4. Select your repository → `shopiq`
5. Set main file path → `app.py`
6. Click **Deploy**

Your app will be live at `https://yourusername-shopiq-app-xxxx.streamlit.app`

### Step 3 — Update the live demo badge in this README
Replace the URL in the badge at the top with your actual deployed URL.

---

## 🔧 Customisation

### Change number of segments
In the sidebar, drag the **"Number of segments (K)"** slider. The elbow chart updates automatically.

### Change anomaly sensitivity
In the sidebar, drag the **"Anomaly sensitivity"** slider. Higher = flags more customers as unusual.

### Use your own data
Click **"Upload your CSV"** in the sidebar. Your file must have columns named (or containing):
- `Annual Income` (any unit)
- `Spending Score`

---

## 📈 Results

| Metric | Value |
|---|---|
| Total customers analysed | 200 |
| Segments discovered | 5 |
| Anomalies flagged | 10 (5%) |
| Income vs Spending correlation | 0.01 (almost zero) |
| Algorithm | K-Means (K=5, n_init=10) |
| Anomaly detector | Isolation Forest (contamination=0.05) |

---

## 💡 Key Learnings (for portfolio)

This project demonstrates:
- **Unsupervised ML** — clustering without labelled data
- **EDA first** — exploring data before any algorithm
- **Feature scaling** — why StandardScaler is mandatory before K-Means
- **Elbow method** — principled approach to choosing K
- **Anomaly detection** — Isolation Forest on real data
- **Business translation** — converting cluster numbers into actionable segment names
- **Production deployment** — Streamlit app deployed to cloud

---

## 🤝 Contributing

Pull requests welcome. For major changes, open an issue first.

---

## 📄 License

MIT License — free to use, modify and distribute.

---

## 👨‍💻 Author

Built as a final project for the **Unsupervised Learning** module.

Algorithms covered: K-Means · DBSCAN · PCA · Isolation Forest · Apriori

---

*If this project helped you, give it a ⭐ on GitHub!*
