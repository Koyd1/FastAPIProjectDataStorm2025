import io
import base64
from logging import error
from fastapi import logger
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_create_stats_graphs(store):
    client = store.supabase_client
    if client is None:
        return []

    table_name = store.settings.supabase_requests_table
    schemas_splited = table_name.split(".")
    schemas_table = schemas_splited[1]
    schema = schemas_splited[0]

    try:
        response = client.schema(schema).table(schemas_table).select("*").execute()
        if error in response:
            raise Exception(response.error.message)
        data = response.data
        if not data:
            return []

        df = pd.DataFrame(data)

        stats_graphs = []

        # Пример 1: Распределение количества транзакций по регионам
        if "region" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            region_counts = df["region"].value_counts().nlargest(10)
            sns.barplot(x=region_counts.index, y=region_counts.values, ax=ax)
            ax.set_title("Распределение транзакций по регионам (ТОП 10)")
            ax.set_ylabel("Количество")
            ax.set_xlabel("Регион")
            plt.xticks(rotation=45)
            img_buf = io.BytesIO()
            plt.tight_layout()
            fig.savefig(img_buf, format="png")
            plt.close(fig)
            img_buf.seek(0)
            img_base64 = base64.b64encode(img_buf.read()).decode()
            stats_graphs.append({"title": "Распределение транзакций по регионам (ТОП 10)", "image_base64": img_base64})

        # Пример 2: Корреляционная матрица по числовым признакам
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if "id" in numeric_cols:
            numeric_cols.remove("id")

        # Фильтруем колонки, оставляя только те числовые, где уникальных значений больше 2 (не бинарные)
        numeric_cols = [
            col for col in numeric_cols
            if df[col].nunique() > 2
        ]

        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
            ax.set_title("Корреляционная матрица числовых признаков")
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png")
            plt.close(fig)
            img_buf.seek(0)
            img_base64 = base64.b64encode(img_buf.read()).decode()
            stats_graphs.append({"title": "Корреляционная матрица числовых признаков", "image_base64": img_base64})

        # Пример 3: Распределение суммы транзакций
        if "amount_mdl" in df.columns:
            fig, ax = plt.subplots(figsize=(8,4))
            sns.histplot(df["amount_mdl"], bins=50, kde=True, ax=ax)
            ax.set_title("Распределение суммы транзакций (amountmdl)")
            ax.set_xlabel("Сумма (MDL)")
            plt.tight_layout()
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png")
            plt.close(fig)
            img_buf.seek(0)
            img_base64 = base64.b64encode(img_buf.read()).decode()
            stats_graphs.append({"title": "Распределение суммы транзакций (amountmdl)", "image_base64": img_base64})

        return stats_graphs

    except Exception as e:
        print(f"Ошибка при загрузке или построении статистики из {table_name}: {str(e)}")
        return []
