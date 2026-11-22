import difflib, ipaddress, random, math, io, zipfile, requests
from collections import Counter
from urllib.parse import urlparse
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

st.set_page_config(page_title="LinkGuard", page_icon="🛡️", layout="wide")

# ── Real public data sources ──
OPENPHISH_URL = "https://openphish.com/feed.txt"
PHISHTANK_URL = "https://data.phishtank.com/data/online-valid.csv"
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"
MAX_PER_CLASS = 6000  # lower to 4000-5000 if Streamlit Cloud runs out of memory

FALLBACK_DOMAINS = [
    "google.com","youtube.com","facebook.com","amazon.com","wikipedia.org","twitter.com","instagram.com",
    "linkedin.com","reddit.com","netflix.com","microsoft.com","apple.com","github.com","stackoverflow.com",
    "paypal.com","bing.com","yahoo.com","twitch.tv","cnn.com","bbc.com","nytimes.com","theguardian.com",
    "reuters.com","wsj.com","forbes.com","bloomberg.com","medium.com","quora.com","slack.com","zoom.us",
    "dropbox.com","adobe.com","salesforce.com","shopify.com","stripe.com","cloudflare.com","notion.so",
    "figma.com","canva.com","discord.com","spotify.com","imdb.com","tripadvisor.com","airbnb.com",
    "booking.com","indeed.com","coursera.org","edx.org","khanacademy.org","duolingo.com","kaggle.com",
    "huggingface.co","openai.com","perplexity.ai","claude.ai","vercel.app","netlify.app","arena.ai",
    "chase.com","wellsfargo.com","bankofamerica.com","citibank.com","capitalone.com",
]

BRANDS = ["google","microsoft","apple","amazon","paypal","netflix","facebook","chase","instagram","youtube","whatsapp","binance","linkedin"]
SUSPICIOUS = ["login","signin","verify","update","secure","account","confirm","password","credential","banking","wallet","recover","billing","invoice","unusual","suspended"]
RISKY_TLD = {"xyz","top","click","site","online","ml","ga","cf","tk","work","live","buzz","icu","cam","rest","quest"}
SECOND_SUFFIX = {"co.uk","org.uk","ac.uk","gov.uk","com.au","co.in","com.br","co.jp","co.nz","com.sg"}

LEGIT_PATHS = ["","/","/about","/contact","/help","/support","/docs","/blog","/news","/pricing","/features",
    "/login","/signup","/signin","/account","/settings","/profile","/search?q=hello","/products","/shop",
    "/en/us/store","/docs/getting-started","/api/v2/users","/download/file.zip","/articles/some-long-article-title-2024",
    "/account/security","/support/tickets","/wikipedia/wiki/Some_Page","/watch?v=dQw4w9WgXcQ"]

# ───────── URL helpers ─────────
def is_ip(host):
    try:
        ipaddress.ip_address(host); return True
    except ValueError:
        return False

def registered_from_host(host):
    if not host or is_ip(host): return host
    parts = host.split(".")
    if len(parts) <= 2: return host
    return ".".join(parts[-3:]) if ".".join(parts[-2:]) in SECOND_SUFFIX else ".".join(parts[-2:])

def main_label(host):
    reg = registered_from_host(host)
    return reg.split(".")[0] if reg and not is_ip(host) else ""

def subdomain_from_host(host):
    reg = registered_from_host(host)
    return host[:-(len(reg)+1)] if host and reg and host.endswith("."+reg) else ""

def features(url):
    p = urlparse(url); h = (p.hostname or "").lower(); low = url.lower()
    try: port = p.port
    except ValueError: port = None
    sub = subdomain_from_host(h); label = main_label(h)
    freq = Counter(url)
    entropy = -sum((c/len(url))*math.log2(c/len(url)) for c in freq.values()) if url else 0.0
    path = p.path or ""; query = p.query or ""
    tld = h.rsplit(".",1)[-1] if "." in h and not is_ip(h) else ""
    return {
        "url_length": len(url), "host_length": len(h), "path_length": len(path), "query_length": len(query),
        "dots": url.count("."), "hyphens": url.count("-"), "underscores": url.count("_"),
        "digits": sum(c.isdigit() for c in url), "params": len(query.split("&")) if query else 0,
        "has_ip": int(is_ip(h)), "has_at": int("@" in url), "has_punycode": int("xn--" in low),
        "nonstandard_port": int(port not in (None,80,443)), "is_https": int(p.scheme=="https"),
        "subdomain_count": len(sub.split(".")) if sub else 0,
        "path_depth": path.count("/"), "path_segments": len([s for s in path.split("/") if s]),
        "special_chars": sum(1 for c in url if not c.isalnum()),
        "risky_tld": int(tld in RISKY_TLD),
        "tld_length": len(tld),
        "brand_in_subdomain": int(any(b in sub.split(".") for b in BRANDS) and label not in BRANDS) if sub else 0,
        "brand_lookalike_host": int(any(label != b and difflib.SequenceMatcher(None,b,label).ratio()>=0.80 for b in BRANDS) if label else 0),
        "entropy": round(entropy,3),
        "percent_encoded": url.count("%"),
        "hex_chars": sum(1 for c in low if c in "0123456789abcdef") ,
        "host_keywords": sum(1 for w in SUSPICIOUS if w in h),
        "path_keywords": sum(1 for w in SUSPICIOUS if w in (path+query).lower()),
        "double_slash_path": int("//" in path),
        "domain_label_len": len(label),
    }

def domain_info(url):
    p = urlparse(url); h = p.hostname or ""
    return {"Hostname": h, "Registered domain": registered_from_host(h), "Subdomain": subdomain_from_host(h) or "-",
            "Main domain name": main_label(h), "Scheme": p.scheme, "Path": p.path or "/", "Uses IP address": is_ip(h)}

# ───────── Data loading (REAL sources first, fallbacks second) ─────────
def fetch_openphish(limit):
    urls = []
    try:
        r = requests.get(OPENPHISH_URL, timeout=25)
        if r.status_code == 200:
            for line in r.text.splitlines():
                u = line.strip()
                if u.startswith("http"): urls.append(u)
                if len(urls) >= limit: break
    except Exception:
        pass
    return urls, "OpenPhish verified feed" if urls else ""

def fetch_phishtank(limit):
    urls = []
    try:
        r = requests.get(PHISHTANK_URL, timeout=30)
        if r.status_code == 200:
            df = pd.read_csv(io.BytesIO(r.content), usecols=["url"])
            urls = df["url"].astype(str).str.strip().tolist()
            urls = [u for u in urls if u.startswith("http")][:limit]
    except Exception:
        pass
    return urls, "PhishTank online-valid" if urls else ""

def fetch_tranco_domains(n=3000):
    try:
        r = requests.get(TRANCO_URL, timeout=40)
        if r.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                name = z.namelist()[0]
                with z.open(name) as f:
                    df = pd.read_csv(f, header=None, nrows=n)
                    doms = df[1].astype(str).str.strip().tolist()
                    if doms: return doms, f"Tranco Top-1M (top {len(doms)} domains)"
    except Exception:
        pass
    return FALLBACK_DOMAINS, "Fallback curated domain list (Tranco unreachable)"

def make_legit_from_domain(domain, rng):
    sub = rng.choice(["", "", "www.", "m.", "app.", "mail.", "docs.", "support.", "blog.", "shop."])
    scheme = "https" if rng.random() < 0.95 else "http"
    path = rng.choice(LEGIT_PATHS)
    if path == "/search?q=hello" and rng.random() < 0.5:
        path = "/search?q=" + "+".join(rng.sample(["learn","python","news","sports","music"], 2))
    return f"{scheme}://{sub}{domain}{path}"

def make_phish_pattern(rng):
    brand = rng.choice(BRANDS)
    w = list(brand); i = rng.randrange(len(w)); r = rng.random()
    if r < 0.4: w[i] = rng.choice("abcdefghijklmnopqrstuvwxyz")
    elif r < 0.7: w.insert(i, rng.choice("abcdefghijklmnopqrstuvwxyz"))
    elif r < 0.85 and len(w) > 3: w.pop(i)
    else:
        j = min(i+1, len(w)-1); w[i], w[j] = w[j], w[i]
    fake = "".join(w)
    tld = rng.choice(list(RISKY_TLD)); scheme = rng.choice(["http","http","https"]); k = rng.randrange(8)
    if k == 0: return f"{scheme}://{fake}-login.{tld}/login"
    if k == 1: return f"https://secure.{fake}.{tld}/account/update?verify=1"
    if k == 2: return f"http://{rng.randint(1,223)}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(0,255)}/bank/login"
    if k == 3: return f"https://www.{fake}.{tld}/signin?password=reset&token={rng.randint(1000,99999)}"
    if k == 4: return f"https://{brand}.login-verify.{fake}.{tld}/confirm/account"
    if k == 5: return f"https://{brand}.com@{fake}.{tld}/login"
    if k == 6: return f"https://xn--{fake}80a.{tld}/secure/verify"
    return f"http://{fake}.{tld}/login"

def load_data():
    rng = random.Random(42)
    source = {}
    phish_real, s1 = fetch_openphish(MAX_PER_CLASS)
    phish_pt, s2 = fetch_phishtank(MAX_PER_CLASS)
    real_phish = list(dict.fromkeys(phish_real + phish_pt))[:MAX_PER_CLASS]
    # keep 70% real, fill the rest with pattern-based to stabilize volume
    n_gen = int(MAX_PER_CLASS * 0.3)
    gen_phish = [make_phish_pattern(rng) for _ in range(n_gen)]
    phish_urls = real_phish + gen_phish
    source["phish"] = " + ".join(x for x in [s1, s2] if x) or "fallback patterns"
    source["phish"] += f" ({len(real_phish)} real, {n_gen} pattern)"

    domains, s3 = fetch_tranco_domains(3000)
    legit_urls = []
    while len(legit_urls) < MAX_PER_CLASS:
        for d in rng.sample(domains, min(len(domains), 200)):
            legit_urls.append(make_legit_from_domain(d, rng))
            if len(legit_urls) >= MAX_PER_CLASS: break
    legit_urls = list(dict.fromkeys(legit_urls))[:MAX_PER_CLASS]
    source["legit"] = f"{s3} -> {len(legit_urls)} URLs"

    # balance classes
    n = min(len(phish_urls), len(legit_urls))
    urls = legit_urls[:n] + phish_urls[:n]
    labels = [0]*n + [1]*n
    df = pd.DataFrame({"url": urls, "label": labels})
    df = df[df["url"].map(lambda u: urlparse(u).scheme in ("http","https") and bool(urlparse(u).hostname))].reset_index(drop=True)
    df["group"] = df["url"].apply(lambda u: registered_from_host(urlparse(u).hostname or ""))
    return df, source

def split_by_domain(df, seed=123):
    rng = random.Random(seed)
    groups = sorted(df["group"].unique()); rng.shuffle(groups)
    n = len(groups)
    n_test = max(1, int(n*0.20)); n_val = max(1, int((n-n_test)*0.25))
    te = df["group"].isin(set(groups[:n_test]))
    va = df["group"].isin(set(groups[n_test:n_test+n_val]))
    tr = ~(te | va)
    def ok(m): return set(df.loc[m,"label"].unique()) == {0,1}
    if not (ok(tr) and ok(va) and ok(te)):  # fallback: row-level split
        idx = df.index.to_numpy(); random.Random(7).shuffle(idx)
        a, b = int(n*0.6), int(n*0.8)
        tr = df.index.isin(idx[:a]); va = df.index.isin(idx[a:b]); te = df.index.isin(idx[b:])
    return tr, va, te

@st.cache_resource(show_spinner="Downloading real datasets and training models... first load only, please wait")
def train_models():
    df, source = load_data()
    tr, va, te = split_by_domain(df)

    vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5), max_features=30000, min_df=2, sublinear_tf=True)
    char_m = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    char_m.fit(vec.fit_transform(df.loc[tr,"url"]), df.loc[tr,"label"])

    x_all = pd.DataFrame([features(u) for u in df["url"]]); cols = list(x_all.columns)
    struct_m = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight="balanced_subsample")
    struct_m.fit(x_all[tr], df.loc[tr,"label"])

    def probs(mask):
        pc = char_m.predict_proba(vec.transform(df.loc[mask,"url"]))[:,1]
        ps = struct_m.predict_proba(x_all[mask])[:,1]
        return pc, ps, 0.5*pc + 0.5*ps

    # validation: choose/verify thresholds
    y_v = df.loc[va,"label"].values
    _, _, ens_v = probs(va)
    thresholds = []
    for t in [0.30,0.40,0.50,0.60,0.70,0.80]:
        pred = (ens_v >= t).astype(int)
        cm = confusion_matrix(y_v, pred, labels=[0,1]); tn,fp,fn,tp = cm.ravel()
        pr,rc,f1,_ = precision_recall_fscore_support(y_v, pred, average="binary", zero_division=0)
        thresholds.append({"Threshold":t,"Precision":round(float(pr),3),"Recall":round(float(rc),3),
                           "F1":round(float(f1),3),"FPR":round(fp/max(fp+tn,1),3),"False Negatives":int(fn)})

    # test metrics
    y_t = df.loc[te,"label"].values
    pc_t, ps_t, ens_t = probs(te)
    fpr_at = {}
    for t in [0.40,0.50,0.70]:
        cm = confusion_matrix(y_t,(ens_t>=t).astype(int),labels=[0,1]); tn,fp,fn,tp = cm.ravel()
        fpr_at[t] = round(fp/max(fp+tn,1),3)

    metrics = {"train":int(tr.sum()),"val":int(va.sum()),"test":int(te.sum()),"source":source,"models":{}}
    for name, prob in [("Character Model",pc_t),("Structural Model",ps_t),("Ensemble (50/50)",ens_t)]:
        pred = (prob>=0.5).astype(int)
        pr,rc,f1,_ = precision_recall_fscore_support(y_t,pred,average="binary",zero_division=0)
        metrics["models"][name] = {"Precision":round(float(pr),3),"Recall":round(float(rc),3),
                                   "F1":round(float(f1),3),"Accuracy":round(float(accuracy_score(y_t,pred)),3)}
    metrics["cm"] = confusion_matrix(y_t,(ens_t>=0.5).astype(int),labels=[0,1]).tolist()
    metrics["fpr_at"] = fpr_at
    metrics["thresholds"] = thresholds

    tdf = df.loc[te].copy(); tdf["score"] = ens_t
    metrics["fp_cases"] = tdf[(y_t==0)&(ens_t>=0.5)].nlargest(3,"score")[["url","score"]].values.tolist()
    metrics["fn_cases"] = tdf[(y_t==1)&(ens_t<0.5)].nsmallest(3,"score")[["url","score"]].values.tolist()

    trdf = df.loc[tr]
    tld_of = lambda u:(urlparse(u).hostname or "").rsplit(".",1)[-1]
    metrics["phish_tlds"] = trdf[trdf["label"]==1]["url"].apply(tld_of).value_counts().head(5).to_dict()
    metrics["legit_tlds"] = trdf[trdf["label"]==0]["url"].apply(tld_of).value_counts().head(5).to_dict()

    # probe: google.com character score (for the "why does google score 0.28" explanation)
    gpc = float(char_m.predict_proba(vec.transform(["https://www.google.com"]))[0][1])
    gps = float(struct_m.predict_proba(pd.DataFrame([features("https://www.google.com")])[cols])[0][1])
    metrics["probe_google"] = {"char": round(gpc,3), "struct": round(gps,3), "combined": round(0.5*gpc+0.5*gps,3)}

    return vec, char_m, struct_m, cols, metrics

vec, char_m, struct_m, cols, metrics = train_models()

# ───────── Analysis ─────────
def indicators(url):
    p = urlparse(url); h = (p.hostname or "").lower(); f = features(url); notes = []
    if f["has_ip"]: notes.append("Uses an IP address instead of a domain name")
    if p.scheme != "https": notes.append("Does not use HTTPS")
    if "@" in url: notes.append("Contains '@' - can hide the real destination")
    if "xn--" in url.lower(): notes.append("Punycode domain - can mimic a real brand")
    if f["subdomain_count"] >= 2: notes.append("Has multiple subdomains")
    if f["hyphens"] >= 3: notes.append("Has many hyphens")
    if f["digits"] >= 8: notes.append("Has many digits")
    if f["host_keywords"] > 0: notes.append("Account-related word in the hostname")
    if f["path_keywords"] > 0: notes.append("Account-related word in the path/query")
    if f["risky_tld"]: notes.append("Uses a high-abuse TLD (.xyz, .top, ...)")
    if f["percent_encoded"] >= 3: notes.append("Contains several percent-encoded characters")
    if f["brand_lookalike_host"]: notes.append("Domain name closely resembles a known brand")
    label = main_label(h); sub = subdomain_from_host(h)
    for b in BRANDS:
        if label and label != b and difflib.SequenceMatcher(None,b,label).ratio() >= 0.75:
            notes.append(f"Lookalike domain: '{label}' is similar to brand '{b}'")
        if sub and b in sub.split(".") and label != b:
            notes.append(f"Brand '{b}' appears in the subdomain, not the registered domain")
    return notes

def analyze(url):
    pc = float(char_m.predict_proba(vec.transform([url]))[0][1])
    ps = float(struct_m.predict_proba(pd.DataFrame([features(url)])[cols])[0][1])
    return pc, ps, 0.5*pc + 0.5*ps

def risk_label(score):
    if score >= 0.70: return "High phishing risk"
    if score >= 0.40: return "Suspicious — investigate carefully"
    return "Lower model-estimated risk"

def set_url(u): st.session_state["url_input"] = u
if "url_input" not in st.session_state: st.session_state["url_input"] = "https://google.com"

# ───────── UI ─────────
st.markdown("""
<style>
.scan-hero{padding:clamp(1.4rem,4vw,3.2rem);margin-bottom:1rem;border:1px solid #315469;border-radius:20px;
background:radial-gradient(ellipse at top right,#1e504d,transparent 65%),#102233}
.scan-hero h1{color:#f4faff;font-size:clamp(2rem,5vw,3.4rem);line-height:1.06;margin:.6rem 0;padding:0}
.scan-hero p{color:#cfdee8;max-width:38rem;margin:.7rem 0 0}
</style>
<section class="scan-hero"><h1>Inspect the link. Before the click.</h1>
<p>Paste a URL to check for phishing signals without visiting the destination.</p></section>
""", unsafe_allow_html=True)

src = metrics["source"]
st.caption(f"Phishing: {src['phish']} · Legitimate: {src['legit']} · Train/Val/Test by domain: {metrics['train']:,}/{metrics['val']:,}/{metrics['test']:,}")

with st.sidebar:
    st.header("How to read a result")
    st.markdown("- **High risk:** ≥ 0.70\n- **Suspicious:** 0.40–0.69\n- **Lower risk:** < 0.40 (not a guarantee of safety)")
    st.divider(); st.caption("Scores are model estimates, not calibrated probabilities.")

tab1, tab2, tab3, tab4 = st.tabs(["Scan a URL", "Batch Check", "Model Performance", "How It Works"])

with tab1:
    with st.container(border=True):
        st.subheader("Check a link")
        for col,(label,example) in zip(st.columns(3), [("Regular link","https://example.org/"),("Lookalike example","https://paypa1.invalid/"),("IP example","http://192.0.2.25/login")]):
            col.button(label, on_click=set_url, args=(example,), use_container_width=True)
        url = st.text_input("URL to inspect", key="url_input", max_chars=8192)
        submitted = st.button("Analyze URL", type="primary", use_container_width=True)
    if submitted:
        c = url.strip(); st.session_state.pop("result", None)
        try:
            p = urlparse(c); _ = p.port
            valid = p.scheme in ("http","https") and bool(p.hostname) and c.isprintable() and not any(ch.isspace() for ch in c)
        except ValueError: valid = False
        if not valid: st.error("Enter a complete HTTP or HTTPS URL.")
        else:
            pc, ps, cb = analyze(c)
            st.session_state["result"] = {"url": c, "char": pc, "struct": ps, "combined": cb, "notes": indicators(c), "info": domain_info(c)}
    r = st.session_state.get("result")
    if r and r["url"] == url.strip():
        v = risk_label(r["combined"])
        st.subheader("Assessment")
        (st.error if v.startswith("High") else st.warning if v.startswith("Suspicious") else st.info)(v)
        st.code(r["url"], language=None, wrap_lines=True)
        a,b,c = st.columns(3)
        a.metric("Character model", f"{r['char']:.3f}"); b.metric("Structural model", f"{r['struct']:.3f}"); c.metric("Combined", f"{r['combined']:.3f}")
        st.progress(r["combined"])
        st.caption("Combined = 0.5 × Character + 0.5 × Structural")
        with st.expander("Rule-based indicators", expanded=True):
            for n in r["notes"]: st.text("• "+n)
            if not r["notes"]: st.write("No rule-based indicators triggered.")
        with st.expander("Domain details"): st.json(r["info"])
        rep = f"LinkGuard Report\nURL: {r['url']}\nAssessment: {v}\nChar: {r['char']:.3f}\nStruct: {r['struct']:.3f}\nCombined: {r['combined']:.3f}\nIndicators:\n" + "\n".join("- "+n for n in r["notes"] or ["None"])
        st.download_button("Download report", rep, "linkguard_report.txt", "text/plain")

with tab2:
    text = st.text_area("Paste URLs, one per line:", "https://google.com\nhttps://gooogle.xyz/login?verify=1", height=140)
    if st.button("Run Batch Check", type="primary"):
        rows = []
        for u in [x.strip() for x in text.splitlines() if x.strip()]:
            p = urlparse(u)
            if p.scheme in ("http","https") and p.hostname:
                pc,ps,cb = analyze(u)
                rows.append({"URL":u,"Char":round(pc,3),"Struct":round(ps,3),"Combined":round(cb,3),"Risk":"High" if cb>=0.7 else ("Suspicious" if cb>=0.4 else "Lower")})
            else: rows.append({"URL":u,"Char":"-","Struct":"-","Combined":"-","Risk":"Invalid"})
        out = pd.DataFrame(rows); st.dataframe(out, use_container_width=True)
        st.download_button("Download CSV", out.to_csv(index=False), "results.csv", "text/csv")

with tab3:
    st.subheader("Test-set performance (unseen domains)")
    st.dataframe(pd.DataFrame(metrics["models"]).T, use_container_width=True)
    st.divider(); st.subheader("Threshold analysis (validation set)")
    st.caption("0.40 = Suspicious boundary (high recall, acceptable FPR). 0.70 = High risk boundary (very high precision, low FPR).")
    st.dataframe(pd.DataFrame(metrics["thresholds"]), use_container_width=True, hide_index=True)
    fa = metrics["fpr_at"]; m1,m2,m3 = st.columns(3)
    m1.metric("Test FPR @0.40", f"{fa[0.40]:.1%}"); m2.metric("Test FPR @0.50", f"{fa[0.50]:.1%}"); m3.metric("Test FPR @0.70", f"{fa[0.70]:.1%}")
    st.divider(); st.subheader("Confusion matrix (ensemble @0.50, test)")
    st.dataframe(pd.DataFrame(metrics["cm"], index=["Actual Legit","Actual Phish"], columns=["Pred Legit","Pred Phish"]))
    st.divider(); st.subheader("Dataset bias check — top TLDs per class (train)")
    b1,b2 = st.columns(2); b1.write("**Phishing:**"); b1.json(metrics["phish_tlds"]); b2.write("**Legitimate:**"); b2.json(metrics["legit_tlds"])
    st.divider(); st.subheader("Failure cases (test set)")
    st.write("**False positives (legit flagged):**")
    for u,s in metrics["fp_cases"]: st.text(f"score {s:.3f} | {u[:90]}")
    if not metrics["fp_cases"]: st.text("None at 0.50.")
    st.write("**False negatives (phishing missed):**")
    for u,s in metrics["fn_cases"]: st.text(f"score {s:.3f} | {u[:90]}")
    if not metrics["fn_cases"]: st.text("None at 0.50.")
    st.divider(); st.subheader("Feature importance (structural model)")
    imp = pd.DataFrame({"Feature":cols,"Importance":struct_m.feature_importances_}).sort_values("Importance")
    st.bar_chart(imp.set_index("Feature"))

with tab4:
    pg = metrics["probe_google"]
    st.code("URL -> validate -> char n-grams + numeric features -> 2 models -> 50/50 average -> risk + indicators")
    st.markdown(f"""
**Models** — Character: TF-IDF char 3–5 n-grams + Logistic Regression. Structural: Random Forest (400 trees) on {len(cols)} numeric features. Ensemble: 0.5×char + 0.5×struct.

**Why does google.com score non-zero on the character model?**
Live probe: `https://www.google.com` → character {pg['char']}, structural {pg['struct']}, combined {pg['combined']}.
The character model sees only letter n-grams; common English fragments in "google" also occur in some phishing URLs, so it can't be sure alone. The structural model scores it near 0 (short, HTTPS, no IP, no suspicious host keyword), pulling the ensemble down. This is why two models are combined.

**Threshold choice** — from the validation table in the Performance tab; test FPRs at 0.40/0.50/0.70 shown there. Practical cutoffs, not probabilities.

**Data** — Phishing: OpenPhish + PhishTank. Legitimate: Tranco Top-1M domains. Split by registered domain (train/val/test), so test domains are unseen.
""")

st.caption("Scores are model estimates. A low score does not guarantee a site is safe.")
