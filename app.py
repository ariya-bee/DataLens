from io import BytesIO
from datetime import datetime
import hashlib
import html
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score,
                             confusion_matrix, classification_report)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier

st.set_page_config(page_title="DataLens | Data Analysis Studio", layout="wide")

# Visual system: intentionally avoids decorative icons and emoji labels.
st.markdown("""
<style>
:root { --bg:#0b1020; --panel:#111a2e; --border:rgba(148,163,184,.18); --text:#e5ecf7; --muted:#9aa9c2; --accent:#8b5cf6; }
.stApp { background:radial-gradient(ellipse at 8% 0%,rgba(91,33,182,.16),transparent 34%),var(--bg); color:var(--text); }
[data-testid="stHeader"] { background:rgba(11,16,32,.78); }
[data-testid="stSidebar"] { background:#10182a; border-right:1px solid var(--border); }
[data-testid="stSidebar"] * { color:var(--text); }
.block-container { max-width:1450px; padding-top:3.25rem; padding-bottom:3rem; }\n#MainMenu { visibility:hidden; }\n[data-testid="stDeployButton"] { display:none; }
h1,h2,h3 { color:#f3f6ff !important; letter-spacing:-.02em; }
h1 { font-size:clamp(2rem,4vw,3rem)!important; font-weight:780!important; }
h2 { border-bottom:1px solid var(--border); padding-bottom:.45rem; margin-top:1.25rem!important; }
p,label,[data-testid="stMarkdownContainer"] { color:var(--text); }
[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(24,35,59,.95),rgba(17,26,46,.88)); border:1px solid var(--border); border-radius:13px; padding:16px 18px; }
[data-testid="stMetricLabel"] { color:var(--muted)!important; }
[data-testid="stMetricValue"] { color:#f8fafc!important; font-weight:750; }
div[data-testid="stDataFrame"],div[data-testid="stTable"] { border:1px solid var(--border); border-radius:10px; overflow:hidden; }
.stButton>button,.stDownloadButton>button,button[kind="primary"] { border:1px solid rgba(139,92,246,.65); border-radius:9px; background:linear-gradient(100deg,#6d28d9,#1d4ed8); color:white; font-weight:600; }
.stButton>button:hover,.stDownloadButton>button:hover { border-color:#a78bfa; color:white; }
[data-testid="stFileUploader"] { background:rgba(17,26,46,.6); border:1px dashed rgba(148,163,184,.4); border-radius:12px; padding:10px; }
[data-testid="stExpander"] { background:rgba(17,26,46,.45); border:1px solid var(--border); border-radius:10px; }
small,.stCaption,[data-testid="stCaptionContainer"] { color:var(--muted)!important; }
[data-baseweb="select"]>div,[data-baseweb="input"]>div,[data-testid="stNumberInput"] input { background-color:#111a2e; border-color:var(--border); color:var(--text); }
hr { border-color:var(--border)!important; }
</style>
""", unsafe_allow_html=True)

st.title("DataLens")
st.markdown("**Data Analysis Studio**  \nExplore dataset quality, visualize patterns, compare machine-learning models, and export reproducible results.")
st.caption("A local-first workflow: the app analyses the CSV you upload in this session.")

with st.sidebar:
    st.markdown("## DataLens")
    st.caption("DATA ANALYSIS STUDIO")
    st.write("Upload a CSV file to access the analysis workspace.")
    st.markdown("---")
    st.markdown("**Workspace sections**")
    st.markdown("Dataset profile\n\nData quality\n\nVisual exploration\n\nModel comparison\n\nFeature importance\n\nExport centre")

uploaded = st.file_uploader("Upload dataset", type=["csv"])
if uploaded is None:
    st.markdown("### Start an analysis")
    st.write("Upload a CSV dataset to generate a profile, inspect data quality, explore distributions and relationships, compare models, and export a report.")
    st.markdown("**Included capabilities**")
    st.markdown("- Dataset summary and column-level profile\n- Missing values, duplicates, and cleaning preview\n- Numerical distributions, category counts, and correlations\n- Regression or classification model comparison\n- Permutation/model-based feature importance where supported\n- Excel workbook, cleaned CSV, and report exports\n- Documentation and internship presentation material")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as exc:
    st.error("The file could not be read as a CSV. Check its delimiter and encoding, then try again.")
    st.caption(f"Technical detail: {exc}")
    st.stop()
if df.empty or len(df.columns) == 0:
    st.error("The uploaded CSV has no usable rows or columns.")
    st.stop()

# Reset dataset-specific state whenever the uploaded file content changes.
upload_bytes = uploaded.getvalue()
upload_fingerprint = hashlib.sha256(upload_bytes).hexdigest()
if st.session_state.get("datalens_upload_fingerprint") != upload_fingerprint:
    st.session_state["datalens_upload_fingerprint"] = upload_fingerprint
    st.session_state["datalens_uploaded_name"] = uploaded.name
    st.session_state.pop("datalens_cleaned_df", None)
    st.session_state.pop("datalens_cleaning_summary", None)
    st.session_state.pop("datalens_model_result", None)

numeric = df.select_dtypes(include=np.number)
categorical = df.select_dtypes(include=["object", "category", "bool"])
missing_total = int(df.isna().sum().sum())
duplicates_total = int(df.duplicated().sum())
structure = pd.DataFrame({
    "Column": df.columns.astype(str), "Data type": df.dtypes.astype(str).values,
    "Missing values": df.isna().sum().values,
    "Missing (%)": (df.isna().mean().values*100).round(2),
    "Unique values": df.nunique(dropna=True).values
})
summary = df.describe(include="all").transpose()

st.success(f"Loaded {uploaded.name}")
m1,m2,m3,m4 = st.columns(4)
m1.metric("Rows", f"{len(df):,}")
m2.metric("Columns", f"{len(df.columns):,}")
m3.metric("Missing cells", f"{missing_total:,}")
m4.metric("Duplicate rows", f"{duplicates_total:,}")

with st.expander("Dataset preview", expanded=True):
    st.dataframe(df.head(100), use_container_width=True)
with st.expander("Column profile and statistical summary"):
    left,right=st.columns(2)
    with left:
        st.markdown("#### Column profile")
        st.dataframe(structure, use_container_width=True)
    with right:
        st.markdown("#### Descriptive statistics")
        st.dataframe(summary, use_container_width=True)

st.header("Data quality")
q1,q2=st.columns(2)
with q1:
    st.markdown("#### Missing values")
    miss=structure.loc[structure["Missing values"]>0, ["Column","Missing values","Missing (%)"]]
    if miss.empty: st.success("No missing values detected.")
    else: st.dataframe(miss, use_container_width=True)
with q2:
    st.markdown("#### Duplicate records")
    if duplicates_total: st.warning(f"{duplicates_total:,} fully duplicated rows found.")
    else: st.success("No fully duplicated rows found.")

st.header("Visual exploration")
if not numeric.empty:
    selected_num=st.selectbox("Numerical column", numeric.columns.tolist(), key="viz_num")
    vals=numeric[selected_num].replace([np.inf,-np.inf],np.nan).dropna()
    if not vals.empty:
        a,b=st.columns(2)
        with a:
            fig,ax=plt.subplots(figsize=(7,4)); sns.histplot(vals,kde=len(vals)>1,ax=ax); ax.set_title(f"Distribution: {selected_num}"); ax.set_xlabel(selected_num); st.pyplot(fig); plt.close(fig)
        with b:
            fig,ax=plt.subplots(figsize=(7,4)); sns.boxplot(x=vals,ax=ax); ax.set_title(f"Spread and potential outliers: {selected_num}"); st.pyplot(fig); plt.close(fig)
    if len(numeric.columns)>=2:
        st.markdown("#### Correlation matrix")
        corr=numeric.replace([np.inf,-np.inf],np.nan).corr()
        fig,ax=plt.subplots(figsize=(max(7,min(14,len(corr.columns)*1.1)),max(4,min(10,len(corr.columns)*.75))))
        sns.heatmap(corr,annot=len(corr.columns)<=12,cmap="coolwarm",center=0,fmt=".2f",ax=ax)
        st.pyplot(fig); plt.close(fig)
        st.caption("Correlation indicates association; it does not establish causation.")
else:
    st.info("No numerical columns were detected, so numerical plots and correlation analysis are unavailable.")
if not categorical.empty:
    selected_cat=st.selectbox("Categorical column", categorical.columns.tolist(), key="viz_cat")
    counts=df[selected_cat].fillna("Missing").astype(str).value_counts().head(20)
    st.bar_chart(counts)
else:
    st.info("No categorical columns were detected.")

st.header("Automatic observations")
observations=[f"The dataset contains {len(df):,} rows and {len(df.columns):,} columns.", f"{missing_total:,} missing cells were found across {int((df.isna().sum()>0).sum())} columns.", f"{duplicates_total:,} duplicate rows were found."]
if not numeric.empty:
    variable=numeric.nunique(dropna=True)
    usable=variable[variable>1]
    if not usable.empty:
        observations.append(f"{len(usable)} numerical columns have more than one distinct observed value.")
if len(df)<100:
    observations.append("This is a relatively small dataset; model scores and statistical summaries may be unstable.")
for item in observations: st.write(item)

# Data cleaning preview: the source DataFrame is never modified.
st.header("Smart data cleaning")
st.write("Choose cleaning actions, review the output, and download a cleaned copy. The uploaded original remains unchanged.")
with st.expander("Configure cleaning", expanded=False):
    remove_dupes=st.checkbox("Remove fully duplicated rows", value=False)
    missing_strategy=st.selectbox("Missing-value handling", ["Leave unchanged","Drop rows with any missing values","Fill numerical columns with median","Fill numerical columns with mean","Fill each column with its mode"])
    if st.button("Apply cleaning to preview", key="apply_cleaning"):
        cleaned=df.copy(); before_rows=len(cleaned)
        if remove_dupes: cleaned=cleaned.drop_duplicates()
        rows_dropped_missing=0
        if missing_strategy=="Drop rows with any missing values":
            old=len(cleaned); cleaned=cleaned.dropna(); rows_dropped_missing=old-len(cleaned)
        elif missing_strategy in ("Fill numerical columns with median","Fill numerical columns with mean"):
            for col in cleaned.select_dtypes(include=np.number).columns:
                fill=cleaned[col].median() if missing_strategy.endswith("median") else cleaned[col].mean()
                if pd.notna(fill): cleaned[col]=cleaned[col].fillna(fill)
        elif missing_strategy=="Fill each column with its mode":
            for col in cleaned.columns:
                modes=cleaned[col].mode(dropna=True)
                if cleaned[col].isna().any() and not modes.empty: cleaned[col]=cleaned[col].fillna(modes.iloc[0])
        st.session_state["datalens_cleaned_df"]=cleaned
        st.session_state["datalens_cleaning_summary"]={"duplicates_removed":before_rows-len(df.drop_duplicates()) if remove_dupes else 0,"rows_removed_for_missing":rows_dropped_missing,"strategy":missing_strategy}
if "datalens_cleaned_df" in st.session_state:
    cleaned=st.session_state["datalens_cleaned_df"]
    cs=st.session_state.get("datalens_cleaning_summary",{})
    x,y,z=st.columns(3); x.metric("Rows after cleaning",f"{len(cleaned):,}"); y.metric("Missing cells remaining",f"{int(cleaned.isna().sum().sum()):,}"); z.metric("Duplicate rows remaining",f"{int(cleaned.duplicated().sum()):,}")
    st.caption(f"Missing-value strategy: {cs.get('strategy','Not specified')} | Duplicate rows removed: {cs.get('duplicates_removed',0):,} | Rows removed for missing values: {cs.get('rows_removed_for_missing',0):,}")
    st.dataframe(cleaned.head(100),use_container_width=True)
    st.download_button("Download cleaned CSV",cleaned.to_csv(index=False).encode("utf-8"),file_name="datalens_cleaned_dataset.csv",mime="text/csv")
    if st.button("Reset cleaning preview"):
        st.session_state.pop("datalens_cleaned_df",None); st.session_state.pop("datalens_cleaning_summary",None); st.rerun()
else:
    cleaned=df.copy()

# Machine learning: compare models on the same held-out split; choose task type explicitly.
st.header("Machine-learning model comparison")
st.write("Select a target column and compare multiple baseline models on a held-out test set. Choose the task type to match the meaning of your target.")
all_cols=df.columns.tolist()
if len(all_cols)<2:
    st.info("At least two columns are required to train a model.")
else:
    with st.form("ml_configuration"):
        target=st.selectbox("Target to predict",all_cols,index=len(all_cols)-1)
        task=st.radio("Prediction task",["Regression","Classification"],horizontal=True,help="Regression predicts a numeric value. Classification predicts a category or class label.")
        feature_options=[c for c in all_cols if c!=target]
        default_features=[c for c in feature_options if c in numeric.columns][:min(5,len(feature_options))]
        features=st.multiselect("Input features",feature_options,default=default_features or feature_options[:min(3,len(feature_options))])
        test_percent=st.slider("Test set size (%)",20,40,25,5)
        random_seed=st.number_input("Random seed",min_value=0,max_value=99999,value=42,step=1)
        compare_clicked=st.form_submit_button("Compare models",type="primary")
    if compare_clicked:
        st.session_state.pop("datalens_model_result", None)
        if not features:
            st.error("Choose at least one input feature.")
        else:
            model_data=df[features+[target]].replace([np.inf,-np.inf],np.nan).dropna(subset=[target]).copy()
            model_data=model_data.loc[model_data[target].notna()]
            if len(model_data)<12:
                st.error("At least 12 rows with a known target are required. Larger datasets are recommended.")
            elif model_data[target].nunique()<2:
                st.error("The target must contain at least two distinct values.")
            elif task=="Classification" and model_data[target].nunique()>50:
                st.error("The target has more than 50 distinct classes. Check whether Regression is more appropriate or simplify the target labels.")
            else:
                X=model_data[features].copy(); y=model_data[target].copy()
                if task=="Classification":
                    counts=y.value_counts()
                    # Stratify only when both partitions can reasonably contain every class.
                    proposed_test_rows=max(1, int(np.ceil(len(y) * test_percent / 100)))
                    proposed_train_rows=len(y)-proposed_test_rows
                    stratify = y if counts.min() >= 2 and proposed_test_rows >= y.nunique() and proposed_train_rows >= y.nunique() else None
                    if stratify is None:
                        st.warning("A stratified split is not possible for this dataset; results may be less stable for rare classes.")
                    scoring_names=["Accuracy","Precision (weighted)","Recall (weighted)","F1 (weighted)"]
                    model_specs={"Logistic Regression":LogisticRegression(max_iter=1500),"Random Forest":RandomForestClassifier(n_estimators=250,random_state=int(random_seed),class_weight="balanced"),"Extra Trees":ExtraTreesClassifier(n_estimators=250,random_state=int(random_seed),class_weight="balanced")}
                else:
                    stratify=None
                    scoring_names=["MAE","RMSE","R²"]
                    model_specs={"Linear Regression":LinearRegression(),"Random Forest":RandomForestRegressor(n_estimators=250,random_state=int(random_seed)),"Extra Trees":ExtraTreesRegressor(n_estimators=250,random_state=int(random_seed))}
                try:
                    X_train,X_test,y_train,y_test=train_test_split(
                        X,y,test_size=test_percent/100,random_state=int(random_seed),stratify=stratify
                    )
                except ValueError as exc:
                    st.error(f"Could not create a train/test split: {exc}")
                    st.stop()
                numeric_features=X_train.select_dtypes(include=np.number).columns.tolist()
                categorical_features=[c for c in X_train.columns if c not in numeric_features]
                transformers=[]
                if numeric_features:
                    transformers.append(("numeric",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),numeric_features))
                if categorical_features:
                    transformers.append(("categorical",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("encoder",OneHotEncoder(handle_unknown="ignore"))]),categorical_features))
                preprocessor=ColumnTransformer(transformers,remainder="drop")
                rows=[]; fitted={}; predictions_by_model={}; errors=[]
                progress=st.progress(0,text="Training and evaluating models...")
                for i,(name,estimator) in enumerate(model_specs.items(),start=1):
                    pipe=Pipeline([("preprocessor",preprocessor),("model",estimator)])
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            pipe.fit(X_train,y_train)
                            pred=pipe.predict(X_test)
                        if task=="Classification":
                            scores={"Model":name,"Accuracy":accuracy_score(y_test,pred),"Precision (weighted)":precision_score(y_test,pred,average="weighted",zero_division=0),"Recall (weighted)":recall_score(y_test,pred,average="weighted",zero_division=0),"F1 (weighted)":f1_score(y_test,pred,average="weighted",zero_division=0)}
                        else:
                            scores={"Model":name,"MAE":mean_absolute_error(y_test,pred),"RMSE":float(np.sqrt(mean_squared_error(y_test,pred))),"R²":r2_score(y_test,pred) if len(y_test)>1 else np.nan}
                        rows.append(scores); fitted[name]=pipe; predictions_by_model[name]=pred
                    except Exception as exc:
                        errors.append(f"{name}: {exc}")
                    progress.progress(i/len(model_specs),text=f"Evaluated {i} of {len(model_specs)} models")
                progress.empty()
                if not rows:
                    st.error("No model could be trained. Check the target values and selected features.")
                    for err in errors: st.caption(err)
                else:
                    metrics=pd.DataFrame(rows)
                    rank_col="F1 (weighted)" if task=="Classification" else "R²"
                    metrics=metrics.sort_values(rank_col,ascending=False,na_position="last")
                    best_name=str(metrics.iloc[0]["Model"])
                    st.session_state["datalens_model_result"]={"metrics":metrics,"best_name":best_name,"task":task,"target":target,"features":features,"models":fitted,"predictions":predictions_by_model,"X_test":X_test,"y_test":y_test,"random_seed":int(random_seed),"test_percent":test_percent}
                    st.success(f"Model comparison completed. Best-ranked model on this split: {best_name}.")
                    st.dataframe(metrics.style.format({c:"{:.4f}" for c in metrics.columns if c!="Model"}),use_container_width=True)
                    st.caption("Ranking is based on this test split only. A higher score is not proof that a model will generalize to every dataset.")
                    st.markdown("#### Best model predictions")
                    best_pred=predictions_by_model[best_name]
                    pred_frame=pd.DataFrame({"Actual":np.asarray(y_test),"Predicted":np.asarray(best_pred)})
                    if task=="Regression": pred_frame["Residual (actual - predicted)"]=pred_frame["Actual"]-pred_frame["Predicted"]
                    else: pred_frame["Correct prediction"]=pred_frame["Actual"].astype(str)==pred_frame["Predicted"].astype(str)
                    st.dataframe(pred_frame.reset_index(drop=True),use_container_width=True)
                    if task=="Classification":
                        st.markdown("#### Confusion matrix")
                        labels=sorted(pd.Series(y_test).astype(str).unique().tolist())
                        cm=confusion_matrix(pd.Series(y_test).astype(str),pd.Series(best_pred).astype(str),labels=labels)
                        fig,ax=plt.subplots(figsize=(max(5,min(10,len(labels)*.8)),max(4,min(9,len(labels)*.7))))
                        sns.heatmap(cm,annot=len(labels)<=12,fmt="d",cmap="Blues",xticklabels=labels,yticklabels=labels,ax=ax)
                        ax.set_xlabel("Predicted label"); ax.set_ylabel("Actual label"); st.pyplot(fig); plt.close(fig)
                    else:
                        fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(y_test,best_pred,alpha=.75)
                        low=min(float(np.min(y_test)),float(np.min(best_pred))); high=max(float(np.max(y_test)),float(np.max(best_pred)))
                        ax.plot([low,high],[low,high],linestyle="--"); ax.set_xlabel("Actual values"); ax.set_ylabel("Predicted values"); ax.set_title("Actual vs predicted")
                        st.pyplot(fig); plt.close(fig)
                    st.markdown("#### Feature importance")
                    best_pipe=fitted[best_name]; estimator=best_pipe.named_steps["model"]
                    importance=None
                    if hasattr(estimator,"feature_importances_"):
                        try:
                            names=best_pipe.named_steps["preprocessor"].get_feature_names_out()
                            importance=pd.DataFrame({"Feature":names,"Importance":estimator.feature_importances_}).sort_values("Importance",ascending=False).head(20)
                        except Exception: importance=None
                    elif hasattr(estimator,"coef_"):
                        try:
                            names=best_pipe.named_steps["preprocessor"].get_feature_names_out()
                            coef=np.asarray(estimator.coef_)
                            magnitude=np.mean(np.abs(coef),axis=0) if coef.ndim>1 else np.abs(coef)
                            importance=pd.DataFrame({"Feature":names,"Importance":magnitude}).sort_values("Importance",ascending=False).head(20)
                        except Exception: importance=None
                    if importance is not None and not importance.empty:
                        st.dataframe(importance,use_container_width=True)
                        chart=importance.sort_values("Importance",ascending=True)
                        fig,ax=plt.subplots(figsize=(8,max(4,len(chart)*.28))); ax.barh(chart["Feature"],chart["Importance"]); ax.set_xlabel("Model importance / coefficient magnitude"); st.pyplot(fig); plt.close(fig)
                        st.caption("Importance describes how the fitted model uses features; it does not establish causation. Coefficient magnitude is scale-dependent and should be interpreted carefully.")
                    else:
                        st.info("This model does not expose a direct feature-importance measure.")
                    if errors:
                        with st.expander("Models that could not be evaluated"):
                            for err in errors: st.write(err)

# Exports: build report content from this uploaded dataset and, if available, the current model run.
st.header("Export centre")
st.write("Create a workbook containing the dataset profile, descriptive statistics, and quality findings. If you trained models in this session, their comparison metrics are included too.")
excel=BytesIO()
with pd.ExcelWriter(excel,engine="openpyxl") as writer:
    structure.to_excel(writer,sheet_name="Column Profile",index=False)
    summary.to_excel(writer,sheet_name="Statistics")
    df.head(10000).to_excel(writer,sheet_name="Data Preview",index=False)
    quality=pd.DataFrame({"Metric":["Rows","Columns","Missing cells","Duplicate rows"],"Value":[len(df),len(df.columns),missing_total,duplicates_total]})
    quality.to_excel(writer,sheet_name="Quality Summary",index=False)
    if "datalens_model_result" in st.session_state:
        mr=st.session_state["datalens_model_result"]
        if mr.get("target") in df.columns:
            mr["metrics"].to_excel(writer,sheet_name="Model Comparison",index=False)
    if "datalens_cleaned_df" in st.session_state:
        st.session_state["datalens_cleaned_df"].head(10000).to_excel(writer,sheet_name="Cleaned Preview",index=False)
st.download_button("Download analysis workbook (Excel)",data=excel.getvalue(),file_name="datalens_analysis_report.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
st.download_button("Download column profile (CSV)",data=structure.to_csv(index=False).encode("utf-8"),file_name="datalens_column_profile.csv",mime="text/csv")

# Portable, formatted HTML report: works without extra PDF libraries.
report_rows="".join(
    f"<tr><td>{html.escape(str(r['Column']))}</td>"
    f"<td>{html.escape(str(r['Data type']))}</td>"
    f"<td>{int(r['Missing values'])}</td>"
    f"<td>{float(r['Missing (%)']):.2f}%</td>"
    f"<td>{int(r['Unique values'])}</td></tr>"
    for _, r in structure.iterrows()
)
model_html=""
if "datalens_model_result" in st.session_state:
    mr=st.session_state["datalens_model_result"]
    model_html="<h2>Model comparison</h2>"+mr["metrics"].to_html(index=False,border=0,float_format=lambda x:f"{x:.4f}")
safe_source_name = html.escape(str(uploaded.name))
safe_observations = ''.join('<li>' + html.escape(str(o)) + '</li>' for o in observations)
html_report=f"""<!doctype html><html><head><meta charset="utf-8"><title>DataLens Analysis Report</title><style>body{{font-family:Arial,sans-serif;max-width:1000px;margin:40px auto;color:#182235;line-height:1.55}}h1{{color:#263b70}}h2{{border-bottom:2px solid #dbe2ef;padding-bottom:6px;margin-top:32px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #d7deea;padding:8px;text-align:left}}th{{background:#edf1f8}}.metrics{{display:flex;gap:16px;flex-wrap:wrap}}.metric{{background:#f2f5fa;padding:14px 18px;border-radius:8px;min-width:130px}}</style></head><body><h1>DataLens Analysis Report</h1><p>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | Source file: {safe_source_name}</p><h2>Dataset overview</h2><div class="metrics"><div class="metric"><b>Rows</b><br>{len(df):,}</div><div class="metric"><b>Columns</b><br>{len(df.columns):,}</div><div class="metric"><b>Missing cells</b><br>{missing_total:,}</div><div class="metric"><b>Duplicate rows</b><br>{duplicates_total:,}</div></div><h2>Column profile</h2><table><thead><tr><th>Column</th><th>Data type</th><th>Missing</th><th>Missing %</th><th>Unique</th></tr></thead><tbody>{report_rows}</tbody></table><h2>Observations</h2><ul>{safe_observations}</ul>{model_html}<p><small>Generated by DataLens. Interpret model results in the context of dataset quality, sample size, and the real-world problem.</small></p></body></html>"""
st.download_button("Download formatted HTML report",data=html_report.encode("utf-8"),file_name="datalens_report.html",mime="text/html")
st.caption("Tip: open the HTML report in a browser and use Print → Save as PDF if you need a PDF copy.")

st.divider()
st.caption("DataLens | Data analysis, data preparation, visualization, and model evaluation")


