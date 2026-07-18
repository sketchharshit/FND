"""
train.py — Full offline training pipeline.
Run: python train.py
"""
import os, sys, time, pickle, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc)
import xgboost as xgb
from preprocessing import preprocess_text

BASE   = os.path.dirname(os.path.abspath(__file__))
D_DIR  = os.path.join(BASE, "dataset")
M_DIR  = os.path.join(BASE, "model")
S_DIR  = os.path.join(BASE, "static", "images")
CACHE  = os.path.join(D_DIR, "processed_cache.pkl")
SEED   = 42
os.makedirs(M_DIR, exist_ok=True); os.makedirs(S_DIR, exist_ok=True)

PALETTE = {"Fake":"#E55C5C","Real":"#4A90D9"}
plt.rcParams.update({"figure.facecolor":"#FAFAFA","axes.facecolor":"#FAFAFA",
                     "axes.spines.top":False,"axes.spines.right":False})

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000,C=1.0,solver="lbfgs",random_state=SEED,n_jobs=-1),
    "Naive Bayes":         MultinomialNB(alpha=0.1),
    "Decision Tree":       DecisionTreeClassifier(max_depth=20,min_samples_split=5,random_state=SEED),
    "Random Forest":       RandomForestClassifier(n_estimators=100,n_jobs=-1,random_state=SEED),
    "SVM (Linear)":        LinearSVC(C=0.5,max_iter=2000,random_state=SEED),
    "XGBoost":             xgb.XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.2,
                               subsample=0.8,colsample_bytree=0.5,eval_metric="logloss",
                               random_state=SEED,n_jobs=-1,tree_method="hist"),
}

def load_data():
    print("[1/5] Loading dataset …")
    df_fake = pd.read_csv(os.path.join(D_DIR,"Fake.csv")); df_fake["label"]=0
    df_true = pd.read_csv(os.path.join(D_DIR,"True.csv")); df_true["label"]=1
    df = pd.concat([df_fake,df_true],ignore_index=True)
    df.drop_duplicates(inplace=True); df.dropna(subset=["title","text"],inplace=True)
    df["content"] = df["title"].fillna("")+" "+df["text"].fillna("")
    print(f"    {len(df):,} articles  Fake={df.label.eq(0).sum():,}  Real={df.label.eq(1).sum():,}")
    return df

def preprocess(df):
    print("[2/5] Preprocessing …")
    if os.path.exists(CACHE):
        with open(CACHE,"rb") as f: p=pickle.load(f)
        if len(p)==len(df):
            print("    Using cache"); return p
    t0=time.time()
    p = df["content"].apply(preprocess_text)
    print(f"    Done in {time.time()-t0:.1f}s")
    with open(CACHE,"wb") as f: pickle.dump(p,f)
    return p

def train_models(X_tr,X_te,y_train,y_test):
    print("[3/5] Training models …")
    rows,details = [],{}
    for name,clf in MODELS.items():
        t0=time.time(); clf.fit(X_tr,y_train); tt=time.time()-t0
        yp=clf.predict(X_te)
        if hasattr(clf,"predict_proba"): yprob=clf.predict_proba(X_te)[:,1]
        elif hasattr(clf,"decision_function"):
            raw=clf.decision_function(X_te); yprob=1/(1+np.exp(-raw))
        else: yprob=yp.astype(float)
        acc=accuracy_score(y_test,yp); prec=precision_score(y_test,yp,zero_division=0)
        rec=recall_score(y_test,yp,zero_division=0); f1=f1_score(y_test,yp,zero_division=0)
        fpr,tpr,_=roc_curve(y_test,yprob); ra=auc(fpr,tpr)
        rows.append({"Model":name,"Accuracy":round(acc*100,2),"Precision":round(prec*100,2),
                     "Recall":round(rec*100,2),"F1 Score":round(f1*100,2),
                     "AUC-ROC":round(ra*100,2),"Train Time":round(tt,2)})
        details[name]={"clf":clf,"y_pred":yp,"y_prob":yprob,
                       "cm":confusion_matrix(y_test,yp),"fpr":fpr,"tpr":tpr,"roc_auc":ra}
        print(f"    ✓ {name:22s}  Acc={acc*100:.2f}%  F1={f1*100:.2f}%  [{tt:.1f}s]")
    return pd.DataFrame(rows).sort_values("F1 Score",ascending=False), details

def save_plots(results_df,details):
    print("[4/5] Saving charts …")
    # Model comparison
    fig,ax=plt.subplots(figsize=(14,6))
    metrics=["Accuracy","Precision","Recall","F1 Score","AUC-ROC"]
    x=np.arange(len(results_df)); w=0.15
    for i,(m,c) in enumerate(zip(metrics,["#4A90D9","#5BC0BE","#6ECB63","#F4A261","#E55C5C"])):
        ax.bar(x+i*w,results_df[m],w,label=m,color=c,alpha=0.88,edgecolor="white")
    ax.set_xticks(x+w*2); ax.set_xticklabels(results_df["Model"],rotation=15,ha="right")
    ax.set_ylim(80,104); ax.set_ylabel("Score (%)"); ax.legend(loc="lower right",ncol=3)
    ax.set_title("Model Performance Comparison",fontsize=14,fontweight="bold")
    ax.yaxis.grid(True,alpha=0.35,linestyle="--")
    plt.tight_layout(); plt.savefig(f"{S_DIR}/model_comparison.png",dpi=150,bbox_inches="tight"); plt.close()
    # ROC
    fig,ax=plt.subplots(figsize=(9,7))
    ax.plot([0,1],[0,1],"k--",alpha=0.4,label="Random (AUC=0.50)")
    for (nm,d),col in zip(details.items(),["#4A90D9","#E55C5C","#6ECB63","#F4A261","#9B59B6","#E67E22"]):
        ax.plot(d["fpr"],d["tpr"],lw=2,color=col,label=f"{nm} (AUC={d['roc_auc']*100:.2f}%)")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curves",fontsize=14,fontweight="bold"); ax.legend(loc="lower right",fontsize=9)
    plt.tight_layout(); plt.savefig(f"{S_DIR}/roc_curves.png",dpi=150,bbox_inches="tight"); plt.close()
    # Confusion matrices
    top4=results_df.head(4)["Model"].tolist()
    fig,axes=plt.subplots(2,2,figsize=(12,10)); axes=axes.ravel()
    for i,nm in enumerate(top4):
        cm=details[nm]["cm"]; cn=cm.astype(float)/cm.sum(axis=1,keepdims=True)
        sns.heatmap(pd.DataFrame(cn,index=["Fake","Real"],columns=["Fake","Real"]),
                    annot=True,fmt=".2%",cmap="Blues",ax=axes[i],cbar=False,
                    annot_kws={"size":13,"weight":"bold"})
        axes[i].set_title(nm,fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{S_DIR}/confusion_matrices.png",dpi=150,bbox_inches="tight"); plt.close()
    # Heatmap
    hdf=results_df.set_index("Model")[["Accuracy","Precision","Recall","F1 Score","AUC-ROC"]]
    fig,ax=plt.subplots(figsize=(10,5))
    sns.heatmap(hdf,annot=True,fmt=".2f",cmap="YlOrRd",vmin=90,vmax=100,
                linewidths=0.5,ax=ax,annot_kws={"size":11,"weight":"bold"})
    ax.set_title("Metrics Heatmap",fontsize=14,fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{S_DIR}/metrics_heatmap.png",dpi=150,bbox_inches="tight"); plt.close()
    # Training time
    fig,ax=plt.subplots(figsize=(9,4))
    cols=["#6ECB63" if m in results_df.iloc[0]["Model"] else "#4A90D9" for m in results_df["Model"]]
    ax.barh(results_df["Model"],results_df["Train Time"],color=cols,alpha=0.85,edgecolor="white",height=0.55)
    ax.set_xlabel("Seconds"); ax.set_title("Training Time per Model",fontweight="bold")
    ax.xaxis.grid(True,alpha=0.35,linestyle="--")
    for i,(m,t) in enumerate(zip(results_df["Model"],results_df["Train Time"])):
        ax.text(t+0.3,i,f"{t:.1f}s",va="center",fontsize=9)
    plt.tight_layout(); plt.savefig(f"{S_DIR}/training_time.png",dpi=150,bbox_inches="tight"); plt.close()
    print("    All 5 charts saved")

def save_model(results_df,details,vec):
    print("[5/5] Saving best model …")
    best=results_df.iloc[0]; nm=best["Model"]; clf=details[nm]["clf"]
    with open(f"{M_DIR}/fake_news_model.pkl","wb") as f: pickle.dump(clf,f)
    with open(f"{M_DIR}/tfidf_vectorizer.pkl","wb") as f: pickle.dump(vec,f)
    meta={"model_name":nm,"accuracy":best["Accuracy"],"precision":best["Precision"],
          "recall":best["Recall"],"f1_score":best["F1 Score"],"auc_roc":best["AUC-ROC"],
          "results_df":results_df,"is_proba_clf":hasattr(clf,"predict_proba")}
    with open(f"{M_DIR}/model_meta.pkl","wb") as f: pickle.dump(meta,f)
    print(f"    Best: {nm}  Acc={best['Accuracy']}%  F1={best['F1 Score']}%")
def main():
    t_all = time.time()
    print("=" * 55)
    print("  Fake News Detector — Training Pipeline")
    print("=" * 55)

    df = load_data()
    proc = preprocess(df)

    X = proc.astype(str).to_numpy()
    y = df["label"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    print(f"    Train={len(X_train):,}  Test={len(X_test):,}")

    vec = TfidfVectorizer(
        max_features=100000,
        ngram_range=(1,2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)

    results_df, details = train_models(X_tr, X_te, y_train, y_test)

    print("\n── Results ──────────────────────────────────────────")
    print(results_df.to_string(index=False))

    save_plots(results_df, details)
    save_model(results_df, details, vec)

    print(f"\n✓ Done in {time.time()-t_all:.1f}s  |  Run: python app.py")
    print("=" * 55)

if __name__ == "__main__":
    main()