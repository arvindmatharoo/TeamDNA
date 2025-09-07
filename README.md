# 🧬 TeamDNA – Football Analytics Dashboard

**TeamDNA** is an interactive **Streamlit app** for football analytics using [StatsBomb Open Data](https://github.com/statsbomb/open-data).  
Explore matches, team stats, player contributions, heatmaps, passing networks, xG maps, and knockout progression for various competitions and seasons.

---

## 🚀 Features

- Browse available competitions and seasons  
- View match results, scores, and winners  
- Team-level summary statistics (wins, draws, losses, goals scored/conceded, goal difference)  
- Knockout stage progression visualization  
- Explore match events: passes, shots, dribbles, fouls  
- Player contributions and key event summaries  
- Player heatmaps and team passing networks  
- Shot maps with Expected Goals (xG)  
- Defensive actions per player  

---

## 🛠️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/arvindmatharoo/TeamDNA.git
cd TeamDNA
```
2. **Create a Virtual Environment(recommended)
## Activate the environment:
## Windows :
```bash
venv\Scripts\activate
```
## Linux/Mac
```bash
source venv/bin/activate
```
3. **Install dependencies**
4. ```bash
   pip install streamlit pandas matplotlib seaborn networkx requests
   ```
# Usage 
## Run the app 
```bash
streamlit run app.py
```

# 📂 Folder Structure
TeamDNA/
│
├── app.py                **Main Streamlit application**
├── README.md             **Project documentation**
├── requirements.txt      **Optional: pip freeze of dependencies**
└── assets/               **Screenshots or images for README**
