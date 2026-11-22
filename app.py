import difflib, ipaddress, random
from urllib.parse import urlparse
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

st.set_page_config(page_title="Phishing URL Detector", page_icon="🛡️", layout="wide")

BRANDS = ["google","microsoft","apple","amazon","paypal","netflix","facebook","chase","instagram","youtube","whatsapp","binance","linkedin"]
SUSPICIOUS = ["login","signin","verify","update","secure","account","confirm","password","credential","banking","wallet","recover"]
RISKY_TLD = {"xyz","top","click","site","online","ml","ga","cf","tk","work","live","buzz"}
SECOND_SUFFIX = {"co.uk","org.uk","ac.uk","gov.uk","com.au","co.in","com.br","co.jp"}
LEGIT_DOMAINS = ["google.com","youtube.com","github.com","wikipedia.org","microsoft.com","apple.com","amazon.com","bbc.com","nytimes.com","openai.com","stackoverflow.com","linkedin.com","instagram.com","facebook.com","netflix.com","paypal.com","chase.com","mozilla.org","wordpress.org","adobe.com","reuters.com","cnn.com","notion.so","figma.com","canva.com","discord.com","twitch.tv","medium.com","duolingo.com","kaggle.com","huggingface.co","linear.app","vercel.app","netlify.app","perplexity.ai","claude.ai"]
WORD1 = ["bright","green","blue","river","city","global","smart","prime","united","national","digital","open","north","south","royal","happy","quick","fresh","great","modern"]
WORD2 = ["labs","health","learn","books","travel","foods","tech","media","news","shop","school","college","library","clinic","studio","works","solutions","group","garden","energy"]
TOPICS = ["technology","students","learning","health","climate","business","sports","science","education","travel","cooking","music","history","future","digital","safety","research","family","nature","economy"]
SYLLABLES = ["a","be","ca","do","en","fi","go","ha","ka","lo","mi","no","pa","ra","re","ro","sa","ta","to","va","ve","za","ar","er","in","on","or","al","el","il","an","un","ex"]
LEGIT_LOGIN = ["https://accounts.google.com/signin/v2/identifier","https://login.microsoftonline.com/","https://www.amazon.com/ap/signin","https://www.paypal.com/signin","https://www.facebook.com/login.php","https://www.instagram.com/accounts/login/","https://www.netflix.com/login","https://github.com/login","https://www.linkedin.com/login","https://www.chase.com/digital/login"]

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
    p = urlparse(url); h = p.hostname or ""; low = url.lower()
    try: port = p.port
    except ValueError: port = None
    sub = subdomain_from_host(h); label = main_label(h)
    return {
        "url_length": len(url), "host_length": len(h), "path_length": len(p.path or ""), "query_length": len(p.query or ""),
        "dots": url.count("."), "hyphens": url.count("-"), "underscores": url.count("_"),
        "digits": sum(c.isdigit() for c in url), "params": len(p.query.split("&")) if p.query else 0,
        "has_ip": int(is_ip(h)), "has_at": int("@" in url), "has_punycode": int("xn--" in low),
        "nonstandard_port": int(port not in (None,80,443)), "is_https": int(p.scheme=="https"),
        "subdomain_count": len(sub.split(".")) if sub else 0,
        "suspicious_words": sum(1 for w in SUSPICIOUS if w in low),
        "path_depth": (p.path or "").count("/"), "special_chars": sum(1 for c in url if not c.isalnum()),
        "risky_tld": int(h.rsplit(".",1)[-1] in RISKY_TLD if "." in h and not is_ip(h) else 0),
        "brand_in_subdomain": int(any(b in sub.split(".") for b in BRANDS) and label not in BRANDS) if sub else 0,
    }

def domain_info(url):
    p = urlparse(url); h = p.hostname or ""
    return {"Hostname": h, "Registered domain": registered_from_host(h), "Subdomain": subdomain_from_host(h) or "-", "Main domain name": main_label(h), "Scheme": p.scheme, "Path": p.path or "/", "Uses IP address": is_ip(h)}

def misspell(w, rng):
    w = list(w); i = rng.randrange(len(w)); r = rng.random()
    if r < 0.4: w[i] = rng.choice("abcdefghijklmnopqrstuvwxyz")
    elif r < 0.7: w.insert(i, rng.choice("abcdefghijklmnopqrstuvwxyz"))
    elif r < 0.85 and len(w) > 3: w.pop(i)
    else:
        j = min(i+1, len(w)-1); w[i], w[j] = w[j], w[i]
    return "".join(w)

def make_legit(rng):
    kind = rng.randrange(6)
    if kind == 0:
        base = rng.choice(LEGIT_DOMAINS); sub = rng.choice(["www","docs","support","help","blog","mail",""])
        return f"https://{sub+'.' if sub else ''}{base}{rng.choice(['','/','/about','/support','/docs','/blog','/products','/account/settings','/news/world'])}"
    if kind == 1:
        return rng.choice(LEGIT_LOGIN) + rng.choice(["", "?hl=en", "?next=home"])
    if kind == 2:
        sep = "-" if rng.random() < 0.2 else ""
        domain = f"{rng.choice(WORD1)}{sep}{rng.choice(WORD2)}.{rng.choice(['com','org','net','info'])}"
        return f"https://{rng.choice(['www.','support.','blog.',''])}{domain}{rng.choice(['/about','/services','/contact','/blog','/shop','/article/'+str(rng.randint(100,99999)),'/team/our-staff'])}"
    if kind == 3:
        slug = "-".join(rng.choices(TOPICS, k=5))
        return f"https://www.{rng.choice(['bbc.com','nytimes.com','reuters.com','cnn.com','theguardian.com'])}/news/articles/{slug}-{rng.randint(10000,99999)}"
    if kind == 4:
        choice = rng.randrange(3)
        if choice == 0:
            code = "".join(rng.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_", k=11))
            return f"https://www.youtube.com/watch?v={code}&list=learning-videos-for-students&index={rng.randint(1,50)}"
        if choice == 1:
            q = "+".join(rng.choices(TOPICS, k=4))
            return f"https://www.google.com/search?q={q}+guide+for+beginners&source=web"
        return f"https://www.amazon.com/product-name-for-home-and-office-use/dp/B{rng.randint(1000000,9999999)}?ref=search&lang=en"
    domain = rng.choice(["stanford.edu","mit.edu","harvard.edu","oxford.ac.uk","cambridge.ac.uk","gov.uk","usa.gov"])
    return f"https://{rng.choice(['www.','students.','courses.',''])}{domain}{rng.choice(['/courses/introduction-to-computer-science/module-1','/students/resources/library-and-study-guide','/about/departments-and-faculty-members','/news/research-and-innovation-update'])}"

def make_general_legit(rng):
    # NEW: normal new/startup websites like arena.ai - short word + modern ending
    word = "".join(rng.choices(SYLLABLES, k=rng.randint(2,4)))
    tld = rng.choice(["com","io","ai","co","app","dev","net","org","so","me"])
    sub = rng.choice(["", "", "", "www.", "app.", "docs.", "blog."])
    path = rng.choice(["", "/", "/about", "/features", "/pricing", "/blog", "/product", "/solutions", "/login", "/signup", "/dashboard", "/search?q=technology+guide"])
    return f"https://{sub}{word}.{tld}{path}"

def make_phish(rng):
    brand = rng.choice(BRANDS); fake = misspell(brand, rng) if rng.random() < 0.65 else brand
    tld = rng.choice(list(RISKY_TLD)); scheme = rng.choice(["http","http","https"]); k = rng.randrange(8)
    if k == 0: return f"{scheme}://{fake}-login.{tld}/login"
    if k == 1: return f"https://secure.{fake}.{tld}/account/update?verify=1"
    if k == 2: return f"http://{rng.randint(1,223)}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(0,255)}/bank/login"
    if k == 3: return f"https://www.{fake}.{tld}/signin?password=reset&token={rng.randint(1000,99999)}"
    if k == 4: return f"https://{brand}.login-verify.{fake}.{tld}/confirm/account"
    if k == 5: return f"https://{brand}.com@{fake}.{tld}/login"
    if k == 6: return f"https://xn--{fake}80a.{tld}/secure/verify"
    return f"http://{fake}.{tld}/login"

@st.cache_resource(show_spinner="Training improved models... first load only, please wait")
def train_models():
    rng = random.Random(42); urls, labels = [], []
    for _ in range(2200):
        urls.append(make_legit(rng)); labels.append(0)
        urls.append(make_phish(rng)); labels.append(1)
    for _ in range(1800):
        urls.append(make_general_legit(rng)); labels.append(0)
        urls.append(make_phish(rng)); labels.append(1)
    df = pd.DataFrame({"url": urls, "label": labels})
    df["group"] = df["url"].apply(lambda u: registered_from_host(urlparse(u).hostname or ""))
    groups = list(df["group"].unique()); random.Random(123).shuffle(groups)
    test_groups = set(groups[:max(1, int(len(groups)*0.25))]); test = df["group"].isin(test_groups); train = ~test

    vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5), max_features=7000)
    char_m = LogisticRegression(max_iter=1000, class_weight="balanced")
    char_m.fit(vec.fit_transform(df.loc[train,"url"]), df.loc[train,"label"])

    x_all = pd.DataFrame([features(u) for u in df["url"]]); cols = list(x_all.columns)
    struct_m = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1, class_weight="balanced")
    struct_m.fit(x_all[train], df.loc[train,"label"])

    y = df.loc[test,"label"].values
    pc = char_m.predict_proba(vec.transform(df.loc[test,"url"]))[:,1]
    ps = struct_m.predict_proba(x_all[test])[:,1]
    ens = 0.5*pc + 0.5*ps
    metrics = {"train": int(train.sum()), "test": int(test.sum()), "models": {}}
    for name, prob in [("Character Model",pc),("Structural Model",ps),("Ensemble (50/50)",ens)]:
        pred = (prob>=0.5).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        metrics["models"][name] = {"Precision": round(float(pr),3), "Recall": round(float(rc),3), "F1": round(float(f1),3), "Accuracy": round(float(accuracy_score(y,pred)),3)}
    cm = confusion_matrix(y, (ens>=0.5).astype(int)).tolist()
    metrics["cm"] = cm
    metrics["fpr"] = round(cm[0][1] / max(cm[0][0]+cm[0][1], 1), 3)
    return vec, char_m, struct_m, cols, metrics

vec, char_m, struct_m, cols, metrics = train_models()

def indicators(url):
    p = urlparse(url); h = p.hostname or ""; f = features(url); notes = []
    if f["has_ip"]: notes.append(" Uses an IP address instead of a domain name")
    if p.scheme != "https": notes.append(" Does not use HTTPS")
    if "@" in url: notes.append(" Contains '@' - this can hide the real website")
    if "xn--" in url.lower(): notes.append(" Punycode domain - can be used to copy a real brand")
    if f["subdomain_count"] >= 2: notes.append("Has many subdomains")
    if f["hyphens"] >= 3: notes.append(" Has many hyphens")
    if f["digits"] >= 8: notes.append(" Has many numbers")
    if f["suspicious_words"] > 0: notes.append(" Contains words like login, verify, secure or password")
    if f["risky_tld"]: notes.append(" Uses a TLD like .xyz or .top, often seen in phishing examples")
    label = main_label(h); sub = subdomain_from_host(h)
    for b in BRANDS:
        if label and label != b and difflib.SequenceMatcher(None,b,label).ratio() >= 0.75:
            notes.append(f" Lookalike domain: '{label}' is very similar to real brand '{b}'")
        if sub and b in sub.split(".") and label != b:
            notes.append(f" Real brand '{b}' is hidden in the subdomain")
    return notes

def analyze(url):
    pc = float(char_m.predict_proba(vec.transform([url]))[0][1])
    ps = float(struct_m.predict_proba(pd.DataFrame([features(url)])[cols])[0][1])
    return pc, ps, 0.5*pc + 0.5*ps

def set_url(u): st.session_state["url_input"] = u
if "url_input" not in st.session_state: st.session_state["url_input"] = "https://google.com"

st.markdown("#  Intelligent Phishing URL Detection & Analysis")
st.markdown("### Inspect the link. Before the click.")
st.write("Two simple models check the URL: one reads small character pieces, the other counts URL features. We average both scores 50/50. Now also trained on normal new websites like .ai / .io startups, so unseen does not automatically mean phishing.")
st.success(f" Improved models trained | Training URLs: {metrics['train']} | Test URLs from unseen domains: {metrics['test']} | Safe links wrongly flagged in test: {metrics['fpr']:.1%}")

with st.sidebar:
    st.header(" Risk Guide")
    st.write(" High risk: score 0.70 or above"); st.write(" Suspicious: score 0.40 to 0.69"); st.write(" Lower risk: score below 0.40")
    st.caption("This is a model estimate, not a 100% guarantee.")

tab1, tab2, tab3, tab4 = st.tabs([" Single URL"," Batch Check"," Model Performance"," How It Works"])

with tab1:
    st.write("**Try an example:**")
    e1, e2, e3 = st.columns(3)
    e1.button(" Safe example", on_click=set_url, args=("https://www.google.com",), use_container_width=True)
    e2.button(" Phishing example", on_click=set_url, args=("https://gooogle.xyz/login?verify=1",), use_container_width=True)
    e3.button(" IP example", on_click=set_url, args=("http://192.168.1.10/bank/login",), use_container_width=True)
    url = st.text_input("Enter URL to check:", key="url_input")
    if st.button("🔍 Analyze URL", type="primary", use_container_width=True):
        p = urlparse(url)
        if p.scheme not in ("http","https") or not p.hostname: st.error("Please enter a valid URL like https://example.com")
        else:
            pc, ps, combined = analyze(url)
            st.session_state["result"] = {"url": url, "char": pc, "struct": ps, "combined": combined, "notes": indicators(url), "info": domain_info(url)}
    if "result" in st.session_state:
        r = st.session_state["result"]; st.write(f"Last checked: `{r['url']}`")
        c1,c2,c3 = st.columns(3)
        c1.metric("Character Model", f"{r['char']:.1%}"); c2.metric("Structural Model", f"{r['struct']:.1%}"); c3.metric("Combined Score", f"{r['combined']:.1%}")
        st.progress(r["combined"])
        if r["combined"] >= 0.70: st.error(" High phishing risk")
        elif r["combined"] >= 0.40: st.warning("Suspicious — investigate carefully")
        else: st.success(" Lower model-estimated risk")
        st.subheader(" Reasons")
        for n in r["notes"]: st.write(f"• {n}")
        if not r["notes"]: st.write("No obvious warning signs found.")
        col1, col2 = st.columns(2)
        with col1: st.subheader("Domain Info"); st.json(r["info"])
        with col2: st.subheader(" Scores"); st.json({"Character score": round(r["char"],4), "Structural score": round(r["struct"],4), "Combined = 0.5 x Character + 0.5 x Structural": round(r["combined"],4)})
        report = f"Phishing URL Check Report\n\nURL: {r['url']}\nCharacter Model: {r['char']:.1%}\nStructural Model: {r['struct']:.1%}\nCombined Score: {r['combined']:.1%}\n\nReasons:\n" + ("\n".join("- "+n for n in r["notes"]) if r["notes"] else "- No obvious warning signs found.")
        st.download_button(" Download Report", report, file_name="phishing_report.txt", mime="text/plain")

with tab2:
    text = st.text_area("Paste many URLs, one per line:", "https://google.com\nhttps://gooogle.xyz/login?verify=1", height=140)
    if st.button(" Run Batch Check", type="primary"):
        rows = []
        for u in [x.strip() for x in text.splitlines() if x.strip()]:
            pp = urlparse(u)
            if pp.scheme in ("http","https") and pp.hostname:
                pc,ps,cb = analyze(u)
                rows.append({"URL":u,"Character":round(pc,3),"Structural":round(ps,3),"Combined":round(cb,3),"Risk":"High" if cb>=0.7 else ("Suspicious" if cb>=0.4 else "Lower")})
            else: rows.append({"URL":u,"Character":"-","Structural":"-","Combined":"-","Risk":"Invalid"})
        out = pd.DataFrame(rows); st.dataframe(out, use_container_width=True)
        st.download_button(" Download Results CSV", out.to_csv(index=False), "results.csv", "text/csv")

with tab3:
    st.subheader("Test results on domains the model never saw in training")
    perf = pd.DataFrame(metrics["models"]).T; st.dataframe(perf, use_container_width=True); st.bar_chart(perf[["Precision","Recall","F1","Accuracy"]])
    st.metric("False Positive Rate - safe links wrongly flagged", f"{metrics['fpr']:.1%}")
    st.write("**Confusion Matrix - Ensemble model:**")
    st.dataframe(pd.DataFrame(metrics["cm"], index=["Actual Legit","Actual Phish"], columns=["Predicted Legit","Predicted Phish"]))
    st.subheader("Which counted features did the Random Forest use most?")
    imp = pd.DataFrame({"Feature": cols, "Importance": struct_m.feature_importances_}).sort_values("Importance")
    st.bar_chart(imp.set_index("Feature"))
    st.caption("Simple meaning: bigger bar = the trees used that feature more when deciding.")

with tab4:
    st.code("URL -> Check format -> Count features + Read character pieces -> 2 Models -> Average score 50/50 -> Risk + Reasons")
    st.markdown("""
**Simple viva explanation:**

- I did not whitelist arena.ai. I added many normal new websites with endings like .ai, .io, .app and .dev, so the model learns that a new domain alone is not phishing.
- **False Positive** means a safe link wrongly flagged. I show its rate in the Performance tab.
- **Character Model:** breaks the URL into pieces of 3 to 5 characters and learns phishing spelling patterns.
- **Structural Model:** many decision trees vote together using counted features like length, dots, hyphens, HTTPS, IP address and subdomains.
- **Domain-separated test:** the same domain is not in both training and testing.
""")

st.caption("Research prototype only - a low score does not guarantee a site is 100% safe.")
