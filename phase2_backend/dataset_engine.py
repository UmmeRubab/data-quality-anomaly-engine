"""Dataset profiling and leakage-aware exploratory ML experiments."""
from __future__ import annotations
import io
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

MAX_BYTES = 8 * 1024 * 1024
MAX_ROWS = 60000
MAX_COLS = 120


def parse_csv(data: bytes) -> pd.DataFrame:
    if not data or len(data) > MAX_BYTES:
        raise ValueError('CSV must be nonempty and no larger than 8 MB')
    try:
        frame = pd.read_csv(io.BytesIO(data), sep=None, engine='python', encoding='utf-8-sig', on_bad_lines='error')
    except (TypeError, ValueError):
        try: frame = pd.read_csv(io.BytesIO(data), sep=None, engine='python', encoding='utf-8-sig', on_bad_lines='error')
        except Exception as exc: raise ValueError('Unable to parse CSV; check delimiter and UTF-8 encoding') from exc
    except Exception as exc:
        raise ValueError('Unable to parse CSV; check delimiter and UTF-8 encoding') from exc
    if frame.empty or frame.shape[1] < 1 or frame.shape[0] > MAX_ROWS or frame.shape[1] > MAX_COLS:
        raise ValueError('CSV must have 1–60,000 rows and 1–120 columns')
    if frame.columns.duplicated().any():raise ValueError('Duplicate column headers are not supported')
    frame.columns = [str(c).strip() for c in frame.columns]
    if len(set(frame.columns)) != len(frame.columns):raise ValueError('Duplicate headers after trimming whitespace')
    return frame


def normalize_unknown(df: pd.DataFrame) -> pd.DataFrame:
    result=df.copy()
    for col in result.select_dtypes(include=['object','string']).columns:
        result[col]=result[col].replace(r'^\s*(\?|unknown|n/a|null|none)\s*$',np.nan,regex=True)
    return result


def profile(df: pd.DataFrame, name: str='Uploaded CSV') -> dict:
    clean=normalize_unknown(df)
    columns=[]
    for col in clean.columns:
        s=clean[col]
        numeric=pd.api.types.is_numeric_dtype(s)
        inferred='numeric' if numeric else 'categorical'
        if not numeric:
            sample=s.dropna().astype(str).head(150)
            if len(sample) and sample.str.len().median()>65:inferred='text'
            elif len(sample) and sample.str.contains(r'[-/]',regex=True).mean()>.7:
                dates=pd.to_datetime(sample,errors='coerce')
                if dates.notna().mean()>.85:inferred='datetime'
        if s.nunique(dropna=True)==len(s) and len(s)>20 and not numeric:inferred='possible_identifier'
        info={'name':col,'type':inferred,'pandas_type':str(s.dtype),'missing':int(s.isna().sum()),'missing_pct':round(float(s.isna().mean()*100),2),'unique':int(s.nunique(dropna=True)),'examples':[str(v)[:90] for v in s.dropna().unique()[:5]]}
        if numeric:
            vals=pd.to_numeric(s,errors='coerce').dropna()
            if len(vals):
                q1,q3=vals.quantile([.25,.75]);iqr=q3-q1
                info.update(min=round(float(vals.min()),3),max=round(float(vals.max()),3),mean=round(float(vals.mean()),3),median=round(float(vals.median()),3),outliers_iqr=int(((vals<q1-1.5*iqr)|(vals>q3+1.5*iqr)).sum()) if iqr else 0)
        else:
            info['top_values']=[{'value':str(k)[:80],'count':int(v)} for k,v in s.value_counts().head(5).items()]
        columns.append(info)
    return {'dataset':name,'rows':len(clean),'columns':len(clean.columns),'missing_cells':int(clean.isna().sum().sum()),'missing_pct':round(float(clean.isna().sum().sum()/clean.size*100),2),'duplicate_rows':int(clean.duplicated().sum()),'column_profiles':columns,'preview':clean.head(12).replace({np.nan:None}).astype(object).where(pd.notna(clean.head(12)),None).to_dict(orient='records'),'notes':['Unknown, ?, N/A, null and none are treated as missing for profiling.','Inferred types are heuristic and should be verified before ML.','IQR outliers are statistical flags, not proof of errors.']}


def run_experiment(df:pd.DataFrame,target:str,model_name:str='logistic_regression',corruption:str='label_noise',rate:float=.2,seed:int=42,drop_duration:bool=True)->dict:
    if target not in df.columns:raise ValueError('Target column does not exist')
    if not 0<=rate<=.5:raise ValueError('Corruption rate must be between 0 and 0.5')
    if model_name not in ('logistic_regression','random_forest','gradient_boosting'):raise ValueError('Unsupported model')
    if corruption not in ('label_noise','missing_values','outliers'):raise ValueError('Unsupported corruption type')
    frame=normalize_unknown(df).dropna(subset=[target]).copy()
    if len(frame)<80:raise ValueError('At least 80 labeled rows are required')
    y=frame[target].astype(str)
    if y.nunique()<2 or y.nunique()>20:raise ValueError('Target must contain 2–20 classes')
    if y.value_counts().min()<3:raise ValueError('Each class needs at least three observations')
    X=frame.drop(columns=[target])
    removed=[]
    if drop_duration and 'duration' in X.columns and target=='y':X=X.drop(columns=['duration']);removed.append('duration (post-call leakage risk)')
    # Avoid automatically using high-cardinality identifiers and free text as features.
    ignored=[c for c in X if X[c].nunique(dropna=True)>max(50,int(len(X)*.8)) and not pd.api.types.is_numeric_dtype(X[c])]
    X=X.drop(columns=ignored);removed.extend(ignored)
    numeric=X.select_dtypes(include='number').columns.tolist()
    categorical=[c for c in X if c not in numeric]
    if not numeric and not categorical:raise ValueError('No usable input features')
    X=X.copy()
    for c in categorical:X[c]=X[c].astype('object').where(X[c].notna(),np.nan)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=seed,stratify=y)
    rng=np.random.default_rng(seed)
    damaged=Xtr.copy();noisy_y=ytr.copy()
    if corruption=='label_noise':
        idx=rng.choice(len(noisy_y),size=int(len(noisy_y)*rate),replace=False)
        classes=np.array(sorted(y.unique()))
        values=noisy_y.to_numpy(copy=True)
        for i in idx:
            other=classes[classes!=values[i]];values[i]=rng.choice(other)
        noisy_y=pd.Series(values,index=ytr.index)
    elif corruption=='missing_values':
        for c in damaged:
            idx=rng.choice(len(damaged),size=int(len(damaged)*rate),replace=False)
            damaged.loc[damaged.index[idx],c]=np.nan
    else:
        for c in numeric:
            vals=pd.to_numeric(damaged[c],errors='coerce')
            spread=float(vals.std())
            if not np.isfinite(spread) or spread==0:continue
            idx=rng.choice(len(damaged),size=int(len(damaged)*rate),replace=False)
            damaged[c]=pd.to_numeric(damaged[c],errors='coerce').astype(float)
            damaged.loc[damaged.index[idx],c]=damaged.loc[damaged.index[idx],c]+rng.choice([-1,1],len(idx))*5*spread
    def build():
        if model_name=='logistic_regression':
            npipe=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]);cpipe=Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]);est=LogisticRegression(max_iter=600,random_state=seed)
        elif model_name=='random_forest':
            npipe=Pipeline([('impute',SimpleImputer(strategy='median'))]);cpipe=Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))]);est=RandomForestClassifier(n_estimators=80,max_depth=12,n_jobs=1,random_state=seed)
        else:
            npipe=Pipeline([('impute',SimpleImputer(strategy='median'))]);cpipe=Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))]);est=HistGradientBoostingClassifier(max_iter=80,random_state=seed)
        return Pipeline([('prep',ColumnTransformer([('num',npipe,numeric),('cat',cpipe,categorical)])),('model',est)])
    def evaluate(features,labels):
        model=build();model.fit(features,labels);pred=model.predict(Xte)
        return {'accuracy':round(float(accuracy_score(yte,pred)),4),'macro_f1':round(float(f1_score(yte,pred,average='macro',zero_division=0)),4),'balanced_accuracy':round(float(balanced_accuracy_score(yte,pred)),4)}
    clean=evaluate(Xtr,ytr);corrupted=evaluate(damaged,noisy_y)
    return {'dataset_rows':len(frame),'train_rows':len(Xtr),'test_rows':len(Xte),'target':target,'model':model_name,'corruption':corruption,'rate':rate,'seed':seed,'clean':clean,'corrupted':corrupted,'accuracy_change':round(corrupted['accuracy']-clean['accuracy'],4),'f1_change':round(corrupted['macro_f1']-clean['macro_f1'],4),'excluded_features':removed,'protocol':'Corruption only in training data; imputation and encoding fitted on training split; untouched test split.','limitations':'Single split; exploratory results are not confidence intervals or evidence of generalization.'}

BANK_URL='https://archive.ics.uci.edu/static/public/222/bank+marketing.zip'
def bank_marketing() -> pd.DataFrame:
    """Load the UCI bank-full.csv variant (45,211 records); cache outside Git."""
    import pathlib,zipfile,urllib.request
    path=pathlib.Path(__file__).resolve().parents[1]/'data'/'bank-full.csv'
    if path.exists():return pd.read_csv(path,sep=';')
    try:
        with urllib.request.urlopen(BANK_URL,timeout=25) as response:payload=response.read(4_000_001)
        if len(payload)>4_000_000:raise ValueError('Dataset archive too large')
        with zipfile.ZipFile(io.BytesIO(payload)) as outer:
            matches=[p for p in outer.namelist() if p.endswith('bank.zip')]
            if not matches:raise ValueError('Unexpected UCI archive structure')
            with zipfile.ZipFile(io.BytesIO(outer.read(matches[0]))) as inner:
                csv_bytes=inner.read('bank-full.csv')
        df=pd.read_csv(io.BytesIO(csv_bytes),sep=';')
        if len(df)!=45211:raise ValueError('Unexpected Bank Marketing record count')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(csv_bytes)
        return df
    except Exception as exc:
        raise ValueError('Unable to download Bank Marketing from UCI. Check internet access or upload bank-full.csv manually.') from exc
