from fastapi import FastAPI
from model import recommend_for_user, compare_models

app = FastAPI()

# -------------------
# HOME
# -------------------
@app.get("/")
def home():
    return {"status": "API running"}

# -------------------
# RECOMMANDATION
# -------------------
@app.get("/recommend/{user_id}")
def recommend(user_id: int):
    return {
        "user_id": user_id,
        "recommendations": recommend_for_user(user_id)
    }

# -------------------
# COMPARAISON
# -------------------
@app.get("/compare")
def compare():
    return compare_models()