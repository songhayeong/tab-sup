# run_density_eval.py
import pandas as pd
import torch
from metrics import TabMetrics
import json

# === 사용자 입력 ===
real_path = "data/shoppers/online_shoppers_intention.csv"        # 원본 데이터 경로
syn_path = "samples/shoppers_block_absorbing_gen_modi.csv"              # Tab-SEDD로 생성한 샘플 경로
info_path = "start/dataset/shoppers/info.json"        # shoppers용 info 파일 경로
device = torch.device("cpu")

# === info 로드 ===
with open(info_path, "r") as f:
    info = json.load(f)

# === syn_data 로드 ===
syn_data = pd.read_csv(syn_path)
real_data = pd.read_csv(real_path)

# real_data 컬럼 순서 기준으로 syn_data 정렬
syn_data = syn_data[real_data.columns]

# dtype 확인 (optional)
# print(syn_data.dtypes)

# === metric 초기화 ===
metric = TabMetrics(
    real_data_path=real_path,
    test_data_path=None,
    val_data_path=None,
    info=info,
    device=device,
    metric_list=['density']  # density만 계산
)

# === shape/trend 평가 ===
metrics, extras = metric.evaluate(syn_data)

## 컬럼별로
# (1) shape
shape_df = extras["shapes"].copy()
# 인덱스 → 컬럼명 매핑
if pd.api.types.is_integer_dtype(shape_df["Column"]):
    shape_df["Column"] = shape_df["Column"].map(lambda i: syn_data.columns[i])
shape_df = shape_df.rename(columns={"Column": "col_name", "Score": "shape_score"})
shape_df = shape_df[["col_name", "shape_score"]]

# (2) trend
trend_df = extras["trends"].copy()
# 인덱스 → 컬럼명 매핑
if pd.api.types.is_integer_dtype(trend_df["Column 1"]):
    trend_df["Column 1"] = trend_df["Column 1"].map(lambda i: syn_data.columns[i])
if pd.api.types.is_integer_dtype(trend_df["Column 2"]):
    trend_df["Column 2"] = trend_df["Column 2"].map(lambda i: syn_data.columns[i])

trend_df = trend_df.rename(columns={"Column 1": "col1", "Column 2": "col2", "Score": "trend_score"})

trend_per_col = {}
for col in syn_data.columns:
    # 해당 컬럼이 포함된 모든 컬럼쌍(row) 선택
    related_pairs = trend_df[(trend_df["col1"] == col) | (trend_df["col2"] == col)]
    if not related_pairs.empty:
        trend_per_col[col] = related_pairs["trend_score"].mean()
    else:
        trend_per_col[col] = None

trend_df_final = pd.DataFrame.from_dict(trend_per_col, orient="index", columns=["trend_score"]).reset_index()
trend_df_final = trend_df_final.rename(columns={"index": "col_name"})

# (3) 합치기
shape_df["col_name"] = shape_df["col_name"].astype(str)
trend_df_final["col_name"] = trend_df_final["col_name"].astype(str)

col_results = pd.merge(shape_df, trend_df_final, on="col_name", how="outer")
col_results = col_results.sort_values(by="col_name").reset_index(drop=True)

print("==== Per-column Shape & Trend Results ====")
print(col_results)

# 저장(optional)
col_results.to_csv("eval/custom/per_column_shape_trend.csv", index=False)

# === 결과 출력 ===
print("==== Density Evaluation Results ====")
print(json.dumps(metrics, indent=4))
print("\n==== Shape Details ====")
print(extras["shapes"].head())
print("\n==== Trend Details ====")
print(extras["trends"].head())
# print(real_data.dtypes)
# print(syn_data.dtypes)