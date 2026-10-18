# 🚦 AI-Powered Traffic Violation Detection System
YOLO11 + Streamlit. Classes: car, motorcycle, helmet, no_helmet, number_plate.

## Folder structure
```
├── app.py
├── best.pt              <- Colab se download ki hui trained model (YOU ADD THIS)
├── requirements.txt
├── packages.txt
├── .gitignore
└── .streamlit/config.toml
```

## Local run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud deploy
1. `best.pt` ko is folder mein rakho.
2. GitHub par new **public** repo banao aur saari files push karo (neeche commands).
3. https://share.streamlit.io par GitHub se login karo → **Create app**.
4. Repo select karo, Branch `main`, Main file path `app.py`.
5. Advanced settings → Python version **3.11** → **Deploy**.
6. 3-5 minute baad app live ho jayegi.

```bash
git init
git add .
git commit -m "Traffic violation detection app"
git branch -M main
git remote add origin https://github.com/<USERNAME>/<REPO>.git
git push -u origin main
```
