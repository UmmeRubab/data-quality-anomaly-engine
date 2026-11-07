import pandas as pd
import pytest
from phase2_backend.dataset_engine import parse_csv,profile,run_experiment

def test_profile_arbitrary_headers():
    df=parse_csv(b'ClientNumber,Revenue_USD,OrderTimestamp\na1,23,2025-01-01\na2,,2025-01-02\n')
    p=profile(df)
    assert p['rows']==2 and p['columns']==3
    assert p['missing_cells']==1
    assert p['column_profiles'][1]['name']=='Revenue_USD'

def test_rejects_oversize():
    with pytest.raises(ValueError):parse_csv(b'x'*(8*1024*1024+1))

def test_leakage_aware_experiment():
    n=160
    df=pd.DataFrame({'age':[20+i%35 for i in range(n)],'job':['admin' if i%2 else 'services' for i in range(n)],'duration':[i*2 for i in range(n)],'y':['yes' if i%3 else 'no' for i in range(n)]})
    r=run_experiment(df,'y',rate=.1)
    assert r['test_rows']==40
    assert any('duration' in s for s in r['excluded_features'])
    assert 0<=r['clean']['accuracy']<=1
