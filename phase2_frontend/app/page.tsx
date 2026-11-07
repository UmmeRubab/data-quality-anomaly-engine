'use client';
import {useEffect,useState} from 'react';
// Same-origin requests are forwarded to the Python API by Next.js.
const API='';
type Column={name:string;type:string;pandas_type:string;missing:number;missing_pct:number;unique:number;examples:string[];mean?:number;median?:number;min?:number;max?:number;outliers_iqr?:number;top_values?:{value:string;count:number}[]};
type Profile={dataset:string;rows:number;columns:number;missing_cells:number;missing_pct:number;duplicate_rows:number;column_profiles:Column[];preview:Record<string,unknown>[];notes:string[]};
type Metrics={accuracy:number;macro_f1:number;balanced_accuracy:number};
type Experiment={clean:Metrics;corrupted:Metrics;accuracy_change:number;f1_change:number;train_rows:number;test_rows:number;excluded_features:string[];protocol:string;limitations:string};
export default function Home(){
 const [source,setSource]=useState<'bank'|'upload'>('bank');const [file,setFile]=useState<File|null>(null);const [data,setData]=useState<Profile|null>(null);
 const [target,setTarget]=useState('y');const [model,setModel]=useState('logistic_regression');const [corruption,setCorruption]=useState('label_noise');const [rate,setRate]=useState(.2);const [seed,setSeed]=useState(42);
 const [result,setResult]=useState<Experiment|null>(null);const [loading,setLoading]=useState(false);const [running,setRunning]=useState(false);const [error,setError]=useState('');
 async function analyze(kind:'bank'|'upload',selectedFile?:File){setLoading(true);setError('');setData(null);setResult(null);
  try{const opts:RequestInit=kind==='bank'?{}:{method:'POST',body:(()=>{const form=new FormData();form.append('file',selectedFile!);return form;})()};
   const res=await fetch(`${API}/api/datasets/${kind==='bank'?'bank/profile':'upload/profile'}`,opts);const body=await res.json();if(!res.ok)throw Error(body.detail||'Profiling failed');setData(body);setTarget(kind==='bank'?'y':(body.column_profiles.find((c:Column)=>c.name==='target'||c.name==='label'||c.name==='y')?.name||body.column_profiles[body.column_profiles.length-1].name));}
  catch(e){setError(e instanceof Error?e.message:'Profiling failed');}finally{setLoading(false);}}
 useEffect(()=>{analyze('bank');},[]);
 async function run(){setRunning(true);setError('');setResult(null);try{let res:Response;
  if(source==='bank')res=await fetch(`${API}/api/datasets/bank/experiment`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({target,model,corruption,rate,seed})});
  else{if(!file)throw Error('Choose a CSV first');const form=new FormData();form.append('file',file);const q=new URLSearchParams({target,model,corruption,rate:String(rate),seed:String(seed)});res=await fetch(`${API}/api/datasets/upload/experiment?${q}`,{method:'POST',body:form});}
  const body=await res.json();if(!res.ok)throw Error(body.detail||'Experiment failed');setResult(body);
 }catch(e){setError(e instanceof Error?e.message:'Experiment failed');}finally{setRunning(false);}}
 function download(name:string,content:string,type='text/csv'){const url=URL.createObjectURL(new Blob([content],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();URL.revokeObjectURL(url);}
 const csv=(rows:Record<string,unknown>[])=>{if(!rows.length)return '';const headers=Object.keys(rows[0]);return [headers.map(quote).join(','),...rows.map(r=>headers.map(h=>quote(r[h])).join(','))].join('\n');};
 function quote(v:unknown){const s=v==null?'':String(v);return '"'+s.replaceAll('"','""')+'"';}
 const pct=(n:number)=>`${(n*100).toFixed(2)}%`;
 return <main><header><p className="eyebrow">DATA-CENTRIC AI · EXPLORATORY RESEARCH</p><h1>DataGuard AI</h1><p>Understand dataset structure, diagnose quality issues, and test model sensitivity to training-data corruption.</p></header>
 <section className="panel"><h2>01 · Dataset explorer</h2><p>Use the UCI Bank Marketing benchmark or upload your own tabular CSV. Files are processed in memory; uploads are not saved by this API.</p>
 <div className="controls"><label>Data source<select value={source} onChange={e=>{const v=e.target.value as 'bank'|'upload';setSource(v);setData(null);setResult(null);setError('');if(v==='bank')analyze('bank');}}><option value="bank">UCI Bank Marketing (45,211 rows)</option><option value="upload">Upload my CSV</option></select></label>
 {source==='upload'&&<label>CSV file (max 8 MB)<input type="file" accept=".csv,text/csv" onChange={e=>{const f=e.target.files?.[0]||null;setFile(f);setData(null);}}/></label>}
 {source==='upload'&&<button disabled={!file||loading} onClick={()=>file&&analyze('upload',file)}>{loading?'Analyzing…':'Analyze CSV'}</button>}</div>
 {loading&&<p>Loading and profiling the dataset…</p>}{error&&<p className="error" role="alert">{error}</p>}
 {data&&<><div className="metrics"><article><span>Total rows</span><strong>{data.rows.toLocaleString()}</strong></article><article><span>Columns</span><strong>{data.columns}</strong></article><article><span>Missing cells</span><strong>{data.missing_pct}%</strong></article><article><span>Duplicate rows</span><strong>{data.duplicate_rows.toLocaleString()}</strong></article></div>
 <div className="controls"><button onClick={()=>download('dataguard-column-profile.csv',csv(data.column_profiles as unknown as Record<string,unknown>[]))}>Export column profile</button><button onClick={()=>download('dataguard-profile.json',JSON.stringify(data,null,2),'application/json')}>Export JSON report</button></div>
 <h3>Column intelligence</h3><div className="tablewrap"><table><thead><tr><th>Column</th><th>Inferred type</th><th>Missing</th><th>Unique</th><th>Useful statistics / examples</th></tr></thead><tbody>{data.column_profiles.map(c=><tr key={c.name}><td><strong>{c.name}</strong></td><td>{c.type}</td><td>{c.missing.toLocaleString()} ({c.missing_pct}%)</td><td>{c.unique.toLocaleString()}</td><td>{c.mean!==undefined?`Mean ${c.mean} · Median ${c.median} · Range ${c.min}–${c.max} · IQR flags ${c.outliers_iqr}`:c.top_values?.map(v=>`${v.value}: ${v.count}`).join(' · ')||c.examples.join(', ')}</td></tr>)}</tbody></table></div>
 <h3>Preview · first {data.preview.length} rows</h3><div className="tablewrap"><table><thead><tr>{data.column_profiles.map(c=><th key={c.name}>{c.name}</th>)}</tr></thead><tbody>{data.preview.map((row,i)=><tr key={i}>{data.column_profiles.map(c=><td key={c.name}>{row[c.name]==null?'—':String(row[c.name])}</td>)}</tr>)}</tbody></table></div>
 <p className="note">{data.notes.join(' ')}</p></>}
 </section>
 <section className="panel"><h2>02 · ML robustness laboratory</h2><p>Select the target and test how training-label noise, missing values, or extreme numerical values affect performance on an untouched test set.</p>
 <div className="controls"><label>Target column<select value={target} onChange={e=>setTarget(e.target.value)}>{data?.column_profiles.map(c=><option key={c.name} value={c.name}>{c.name}</option>)}</select></label>
 <label>Model<select value={model} onChange={e=>setModel(e.target.value)}><option value="logistic_regression">Logistic Regression</option><option value="random_forest">Random Forest</option><option value="gradient_boosting">Gradient Boosting</option></select></label>
 <label>Corruption<select value={corruption} onChange={e=>setCorruption(e.target.value)}><option value="label_noise">Label noise</option><option value="missing_values">Missing values</option><option value="outliers">Numerical outliers</option></select></label>
 <label>Corruption rate · {Math.round(rate*100)}%<input type="range" min="0" max="0.5" step="0.05" value={rate} onChange={e=>setRate(Number(e.target.value))}/></label>
 <label>Random seed<input type="number" min="0" max="4294967295" value={seed} onChange={e=>setSeed(Number(e.target.value))}/></label>
 <button disabled={!data||running} onClick={run}>{running?'Running…':'Run comparison'}</button></div>
 {result&&<><div className="metrics"><article><span>Clean accuracy</span><strong>{pct(result.clean.accuracy)}</strong></article><article><span>Corrupted accuracy</span><strong>{pct(result.corrupted.accuracy)}</strong></article><article><span>Accuracy change</span><strong>{(result.accuracy_change*100).toFixed(2)} pp</strong></article><article><span>Macro F1 change</span><strong>{(result.f1_change*100).toFixed(2)} pp</strong></article></div>
 <h3>Model performance comparison</h3>{(['accuracy','macro_f1','balanced_accuracy'] as const).map(k=><div className="comparison" key={k}><strong>{k.replaceAll('_',' ')}</strong><div className="barrow"><span>Clean</span><div className="bartrack"><div className="bar clean" style={{width:`${Math.max(0,result.clean[k]*100)}%`}}/></div><b>{pct(result.clean[k])}</b></div><div className="barrow"><span>Corrupted</span><div className="bartrack"><div className="bar damaged" style={{width:`${Math.max(0,result.corrupted[k]*100)}%`}}/></div><b>{pct(result.corrupted[k])}</b></div></div>)}
 <p className="note">Train: {result.train_rows.toLocaleString()} · Test: {result.test_rows.toLocaleString()}. {result.protocol} {result.excluded_features.length>0?`Excluded: ${result.excluded_features.join(', ')}.`:''} {result.limitations}</p>
 <button onClick={()=>download('dataguard-experiment.json',JSON.stringify({dataset:data?.dataset,target,model,corruption,rate,seed,...result},null,2),'application/json')}>Download experiment report</button></>}
 </section><footer>DataGuard AI · Research prototype · UCI Bank Marketing attribution: UCI Machine Learning Repository · Never upload sensitive personal data to a public demo.</footer></main>;
}
