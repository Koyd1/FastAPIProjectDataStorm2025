# import io
# import joblib
# import pandas as pd
# from fastapi import FastAPI, Request, Form, UploadFile, File
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# import lightgbm as lgb

# app = FastAPI(title="Anomaly Detection API")

# # === Подключаем шаблоны ===
# templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # === Загрузка модели ===
# model = joblib.load("models/lgbm_anomaly_multi.pkl")
# meta = joblib.load("models/lgbm_anomaly_meta.pkl")

# feature_cols = meta["features"]
# classes = meta["classes"]

# categorical_cols = [
#     'issuer_bank', 'region', 'cluster_name_expected', 'channel',
#     'merchant_category', 'merchant_country', 'currency',
#     'auth_method', 'three_ds_result', 'device_type'
# ]


# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})


# @app.post("/upload_csv", response_class=HTMLResponse)
# async def upload_csv(request: Request, file: UploadFile = File(...)):
#     """Обработка CSV и вывод таблицы с предсказаниями"""
#     content = await file.read()
#     df = pd.read_csv(io.StringIO(content.decode("utf-8")))

#     # Преобразуем категориальные и числовые
#     for col, mapping in {
#         'account_type': {'Individual': 0, 'Business': 1},
#         'card_type': {'Debit': 0, 'Credit': 1},
#         'product_tier': {'Standard': 0, 'Gold': 1, 'Platinum': 2}
#     }.items():
#         if col in df.columns:
#             df[col] = df[col].map(mapping)

#     df_enc = pd.get_dummies(df, columns=categorical_cols, dtype=int)
#     missing_cols = set(feature_cols) - set(df_enc.columns)
#     for col in missing_cols:
#         df_enc[col] = 0
#     df_enc = df_enc[feature_cols]

#     preds = model.predict(df_enc)
#     probs = model.predict_proba(df_enc)

#     result_df = pd.DataFrame({
#         "Predicted_Class": preds
#     })
#     for i, cls in enumerate(classes):
#         result_df[f"Prob_{cls}"] = probs[:, i]

#     html_table = result_df.head(20).to_html(classes='table table-striped', index=False)

#     return templates.TemplateResponse(
#         "result.html",
#         {"request": request, "table": html_table}
#     )


# @app.get("/manual", response_class=HTMLResponse)
# async def manual_form(request: Request):
#     """Форма для ручного ввода"""
#     return templates.TemplateResponse("manual.html", {"request": request})


# @app.post("/manual", response_class=HTMLResponse)
# async def manual_predict(
#     request: Request,
#     issuer_bank: str = Form(...),
#     region: str = Form(...),
#     amount_mdl: float = Form(...),
#     card_type: str = Form("Debit"),
#     product_tier: str = Form("Standard"),
#     account_type: str = Form("Individual")
# ):
#     """Предсказание для одной транзакции"""
#     df = pd.DataFrame([{
#         "issuer_bank": issuer_bank,
#         "region": region,
#         "amount_mdl": amount_mdl,
#         "card_type": card_type,
#         "product_tier": product_tier,
#         "account_type": account_type
#     }])

#     # Преобразования
#     for col, mapping in {
#         'account_type': {'Individual': 0, 'Business': 1},
#         'card_type': {'Debit': 0, 'Credit': 1},
#         'product_tier': {'Standard': 0, 'Gold': 1, 'Platinum': 2}
#     }.items():
#         if col in df.columns:
#             df[col] = df[col].map(mapping)

#     df_enc = pd.get_dummies(df, columns=categorical_cols, dtype=int)
#     missing_cols = set(feature_cols) - set(df_enc.columns)
#     for col in missing_cols:
#         df_enc[col] = 0
#     df_enc = df_enc[feature_cols]

#     pred = model.predict(df_enc)[0]
#     prob = model.predict_proba(df_enc)[0]

#     results = dict(zip(classes, [round(x, 3) for x in prob]))

#     return templates.TemplateResponse(
#         "manual_result.html",
#         {"request": request, "pred": pred, "probs": results}
#     )

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
