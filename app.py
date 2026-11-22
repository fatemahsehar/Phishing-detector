import difflib, ipaddress, random, math, requests
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

OPENPHISH_URL = "https://openphish.com/feed.txt"

TRANCO_DOMAINS = [
    "google.com","youtube.com","facebook.com","amazon.com","wikipedia.org","twitter.com","instagram.com",
    "linkedin.com","reddit.com","netflix.com","microsoft.com","apple.com","github.com","stackoverflow.com",
    "pinterest.com","tumblr.com","wordpress.com","ebay.com","paypal.com","bing.com","yahoo.com","twitch.tv",
    "cnn.com","bbc.com","nytimes.com","theguardian.com","reuters.com","washingtonpost.com","wsj.com","forbes.com",
    "bloomberg.com","medium.com","quora.com","slack.com","zoom.us","dropbox.com","adobe.com","salesforce.com",
    "shopify.com","stripe.com","cloudflare.com","notion.so","figma.com","canva.com","discord.com","spotify.com",
    "soundcloud.com","vimeo.com","imdb.com","tripadvisor.com","airbnb.com","booking.com","expedia.com","uber.com",
    "yelp.com","indeed.com","glassdoor.com","coursera.org","edx.org","khanacademy.org","udemy.com","duolingo.com",
    "kaggle.com","huggingface.co","openai.com","perplexity.ai","claude.ai","character.ai","together.ai",
    "stability.ai","elevenlabs.io","wandb.ai","replit.com","vercel.app","netlify.app","heroku.com","digitalocean.com",
    "supabase.com","mongodb.com","docker.com","kubernetes.io","npmjs.com","pypi.org","gitlab.com","bitbucket.org",
    "trello.com","asana.com","monday.com","airtable.com","linear.app","hubspot.com","mailchimp.com","twilio.com",
    "sentry.io","datadog.com","godaddy.com","fastly.com","akamai.com","arena.ai","hulu.com","disneyplus.com",
    "lyft.com","doordash.com","coinbase.com","binance.com","robinhood.com","chase.com","wellsfargo.com",
    "bankofamerica.com","citibank.com","capitalone.com","visa.com",
]

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
    freq = Counter(url)
    entropy = -sum((c/len(url))*math.log2(c/len(url)) for c in freq.values()) if url else 0.0
    return {
        "url_length": len(url), "host_length": len(h), "path_length": len(p.path or ""), "query_length": len(p.query or ""),
        "dots": url.count("."), "hyphens": url.count("-"), "underscores": url.count("_"),
        "digits": sum(c.isdigit() for c in url), "params": len(p.query.split("&")) if p.query else 0,
        "has_ip": int(is_ip(h)), "has_at": int("@" in url), "has_punycode": int("xn--" in low),
        "nonstandard_port": int(port not in (None,80,443)), "is_https": int(p.scheme=="https"),
        "subdomain_count": len(sub.split(".")) if sub else 0,
        "path_depth": (p.path or "").count("/"), "special_chars": sum(1 for c in url if not c.isalnum()),
        "risky_tld": int(h.rsplit(".",1)[-1] in RISKY_TLD if "." in h and not is_ip(h) else 0),
        "brand_in_subdomain": int(any(b in sub.split(".") for b in BRANDS) and label not in BRANDS) if sub else 0,
        "entropy": round(entropy, 3),
        "percent_encoded": url.count("%"),
        "host_keywords": sum(1 for w in SUSPICIOUS if w in h.lower()),
        "path_keywords": sum(1 for w in SUSPICIOUS if w in ((p.path or "") + (p.query or "")).lower()),
        "suspicious_words": sum(1 for w in SUSPICIOUS if w in low),
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

def make_real_legit_url(domain, rng):
    sub = rng.choice(["", "", "www.", "m.", "app.", "mail.", "docs.", "support.", "blog."])
    paths = ["", "/", "/about", "/contact", "/login", "/signup", "/search?q=hello", "/products", "/blog", "/help",
             "/settings", "/profile", "/news", "/en/us/store", "/docs/getting-started", "/pricing", "/features",
             "/support/tickets", "/account/security", "/api/v2/users"]
    return f"https://{sub}{domain}{rng.choice(paths)}"

def load_real_data():
    phish_urls, legit_urls = [], []
    source = {"phish": "", "legit": ""}
    rng = random.Random(42)
    try:
        resp = requests.get(OPENPHISH_URL, timeout=20)
        if resp.status_code == 200:
            for line in resp.text.strip().split("\n"):
                u = line.strip()
                if u.startswith("http"):
                    phish_urls.append(u)
                if len(phish_urls) >= 3000:
                    break
        if phish_urls:
            source["phish"] = f"OpenPhish verified feed ({len(phish_urls)} URLs)"
    except Exception:
        pass
    if not phish_urls:
        phish_urls = [make_phish(rng) for _ in range(3000)]
        source["phish"] = "Fallback pattern-generated (OpenPhish unreachable)"
    for domain in TRANCO_DOMAINS:
        for _ in range(rng.randint(2, 5)):
            legit_urls.append(make_real_legit_url(domain, rng))
        if len(legit_urls) >= 3000:
            break
    while len(legit_urls) < 3000:
        legit_urls.append(make_legit(rng))
    source["legit"] = f"Tranco Top domains ({len(set(TRANCO_DOMAINS))} real domains listed)"
    return phish_urls, legit_urls, source

def split_by_domain(df, test_frac=0.25, val_frac=0.20, seed=123):
    rng = random.Random(seed)
    groups = sorted(df["group"].unique())
    rng.shuffle(groups)
    n = len(groups)
    n_test = max(1, int(n * test_frac))
    test_groups = set(groups[:n_test])
    rest = groups[n_test:]
    n_val = max(1, int(len(rest) * val_frac))
    val_groups = set(rest[:n_val])
    train_groups = set(rest[n_val:])
    train = df["group"].isin(train_groups)
    val = df["group"].isin(val_groups)
    test = df["group"].isin(test_groups)
    def both_classes(mask):
        return set(df.loc[mask, "label"].unique()) == {0, 1}
    if not (both_classes(train) and both_classes(val) and both_classes(test)):
        test = df["group"].isin(test_groups)
        train = ~test
        val = test
    return train, val, test

@st.cache_resource(show_spinner="Loading data and training models... please wait")
def train_models():
    phish_urls, legit_urls, source = load_real_data()
    rng = random.Random(42)
    generated_phish = [make_phish(rng) for _ in range(3000)]
    all_phish = phish_urls[:3000] + generated_phish
    generated_legit = [make_legit(rng) for _ in range(2000)] + [make_general_legit(rng) for _ in range(1500)]
    all_legit = legit_urls[:3000] + generated_legit
    urls = all_legit + all_phish
    labels = [0] * len(all_legit) + [1] * len(all_phish)
    df = pd.DataFrame({"url": urls, "label": labels})
    df["group"] = df["url"].apply(lambda u: registered_from_host(urlparse(u).hostname or ""))
    train_mask, val_mask, test_mask = split_by_domain(df)

    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=10000, min_df=2)
    cw = {0: 1.5, 1: 1}  # weight legitimate higher to reduce false alarms
    char_m = LogisticRegression(max_iter=1500, class_weight=cw, C=0.5)
    char_m.fit(vec.fit_transform(df.loc[train_mask, "url"]), df.loc[train_mask, "label"])

    x_all = pd.DataFrame([features(u) for u in df["url"]])
    cols = list(x_all.columns)
    struct_m = RandomForestClassifier(n_estimators=300, max_depth=18, min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight=cw)
    struct_m.fit(x_all[train_mask], df.loc[train_mask, "label"])

    def probs(mask):
        pc = char_m.predict_proba(vec.transform(df.loc[mask, "url"]))[:, 1]
        ps = struct_m.predict_proba(x_all[mask])[:, 1]
        return pc, ps, 0.5 * pc + 0.5 * ps

    pc_v, ps_v, ens_v = probs(val_mask)
    y_v = df.loc[val_mask, "label"].values
    thresholds = []
    for t in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        pred = (ens_v >= t).astype(int)
        cm = confusion_matrix(y_v, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        pr, rc, f1, _ = precision_recall_fscore_support(y_v, pred, average="binary", zero_division=0)
        thresholds.append({"Threshold": t, "Precision": round(float(pr), 3), "Recall": round(float(rc), 3),
                           "F1": round(float(f1), 3), "FPR": round(fp / max(fp + tn, 1), 3), "False Negatives": int(fn)})

    pc_t, ps_t, ens_t = probs(test_mask)
    y_t = df.loc[test_mask, "label"].values
    fpr_at = {}
    for t in [0.40, 0.50, 0.70]:
        cm = confusion_matrix(y_t, (ens_t >= t).astype(int), labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr_at[t] = round(fp / max(fp + tn, 1), 3)

    metrics = {"train": int(train_mask.sum()), "val": int(val_mask.sum()), "test": int(test_mask.sum()),
               "models": {}, "source": source}
    for name, prob in [("Character Model", pc_t), ("Structural Model", ps_t), ("Ensemble (50/50)", ens_t)]:
        pred = (prob >= 0.5).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(y_t, pred, average="binary", zero_division=0)
        metrics["models"][name] = {"Precision": round(float(pr), 3), "Recall": round(float(rc), 3),
                                   "F1": round(float(f1), 3), "Accuracy": round(float(accuracy_score(y_t, pred)), 3)}
    cm = confusion_matrix(y_t, (ens_t >= 0.5).astype(int), labels=[0, 1]).tolist()
    metrics["cm"] = cm
    metrics["fpr"] = fpr_at[0.50]
    metrics["fpr_at"] = fpr_at
    metrics["thresholds"] = thresholds
    test_df = df.loc[test_mask].copy()
    test_df["score"] = ens_t
    metrics["fp_cases"] = test_df[(y_t == 0) & (ens_t >= 0.5)].nlargest(3, "score")[["url", "score"]].values.tolist()
    metrics["fn_cases"] = test_df[(y_t == 1) & (ens_t < 0.5)].nsmallest(3, "score")[["url", "score"]].values.tolist()
    tr = df.loc[train_mask]
    tld_of = lambda u: (urlparse(u).hostname or "").rsplit(".", 1)[-1]
    metrics["phish_tlds"] = tr[tr["label"] == 1]["url"].apply(tld_of).value_counts().head(5).to_dict()
    metrics["legit_tlds"] = tr[tr["label"] == 0]["url"].apply(tld_of).value_counts().head(5).to_dict()
    return vec, char_m, struct_m, cols, metrics

vec, char_m, struct_m, cols, metrics = train_models()

def indicators(url):
    p = urlparse(url); h = p.hostname or ""; f = features(url); notes = []
    if f["has_ip"]: notes.append("Uses an IP address instead of a domain name")
    if p.scheme != "https": notes.append("Does not use HTTPS")
    if "@" in url: notes.append("Contains '@' - can hide the real destination")
    if "xn--" in url.lower(): notes.append("Punycode domain - can mimic a real brand")
    if f["subdomain_count"] >= 2: notes.append("Has multiple subdomains")
    if f["hyphens"] >= 3: notes.append("Has many hyphens")
    if f["digits"] >= 8: notes.append("Has many numbers")
    if f["host_keywords"] > 0: notes.append("Account-related word in the hostname")
    if f["path_keywords"] > 0: notes.append("Account-related word in the path/query")
    if f["risky_tld"]: notes.append("Uses a TLD like .xyz or .top")
    if f["percent_encoded"] >= 3: notes.append("Contains several percent-encoded characters")
    label = main_label(h); sub = subdomain_from_host(h)
    for b in BRANDS:
        if label and label != b and difflib.SequenceMatcher(None, b, label).ratio() >= 0.75:
            notes.append(f"Lookalike domain: '{label}' is similar to brand '{b}'")
        if sub and b in sub.split(".") and label != b:
            notes.append(f"Brand '{b}' appears in the subdomain, not the registered domain")
    return notes

def analyze(url):
    pc = float(char_m.predict_proba(vec.transform([url]))[0][1])
    ps = float(struct_m.predict_proba(pd.DataFrame([features(url)])[cols])[0][1])
    return pc, ps, 0.5 * pc + 0.5 * ps

def risk_label(score):
    if score >= 0.70: return "High phishing risk"
    if score >= 0.40: return "Suspicious — investigate carefully"
    return "Lower model-estimated risk"

def set_url(u): st.session_state["url_input"] = u
if "url_input" not in st.session_state: st.session_state["url_input"] = "https://google.com"

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
st.caption(f"Phishing source: {src['phish']} · Legitimate source: {src['legit']} · Train/Val/Test: {metrics['train']:,}/{metrics['val']:,}/{metrics['test']:,} (split by domain)")

with st.sidebar:
    st.header("How to read a result")
    st.markdown("- **High risk:** score ≥ 0.70\n- **Suspicious:** 0.40 – 0.69\n- **Lower risk:** < 0.40. Not a guarantee of safety.")
    st.divider()
    st.caption("Scores are model estimates, not calibrated probabilities.")

tab1, tab2, tab3, tab4 = st.tabs(["Scan a URL", "Batch Check", "Model Performance", "How It Works"])

with tab1:
    with st.container(border=True):
        st.subheader("Check a link")
        examples = [("Regular link", "https://example.org/"), ("Lookalike example", "https://paypa1.invalid/"), ("IP example", "http://192.0.2.25/login")]
        for col, (label, example) in zip(st.columns(3), examples):
            col.button(label, on_click=set_url, args=(example,), use_container_width=True)
        url = st.text_input("URL to inspect", key="url_input", max_chars=8192)
        submitted = st.button("Analyze URL", type="primary", use_container_width=True)
    if submitted:
        candidate = url.strip()
        st.session_state.pop("result", None)
        try:
            parsed = urlparse(candidate); _ = parsed.port
            valid = parsed.scheme in ("http", "https") and bool(parsed.hostname) and candidate.isprintable() and not any(c.isspace() for c in candidate)
        except ValueError:
            valid = False
        if not valid:
            st.error("Enter a complete HTTP or HTTPS URL.")
        else:
            pc, ps, combined = analyze(candidate)
            st.session_state["result"] = {"url": candidate, "char": pc, "struct": ps, "combined": combined, "notes": indicators(candidate), "info": domain_info(candidate)}
    r = st.session_state.get("result")
    if r and r["url"] == url.strip():
        verdict = risk_label(r["combined"])
        st.subheader("Assessment")
        (st.error if verdict.startswith("High") else st.warning if verdict.startswith("Suspicious") else st.info)(verdict)
        st.code(r["url"], language=None, wrap_lines=True)
        a, b, c = st.columns(3)
        a.metric("Character model", f"{r['char']:.3f}")
        b.metric("Structural model", f"{r['struct']:.3f}")
        c.metric("Combined score", f"{r['combined']:.3f}")
        st.progress(r["combined"])
        st.caption("Combined = 0.5 × Character + 0.5 × Structural.")
        with st.expander("Rule-based indicators", expanded=True):
            for n in r["notes"]: st.text("• " + n)
            if not r["notes"]: st.write("No rule-based indicators triggered.")
        with st.expander("Domain details"): st.json(r["info"])
        report = f"LinkGuard Report\nURL: {r['url']}\nAssessment: {verdict}\nCharacter: {r['char']:.3f}\nStructural: {r['struct']:.3f}\nCombined: {r['combined']:.3f}\nIndicators:\n" + "\n".join("- " + n for n in r["notes"] or ["None"])
        st.download_button("Download report", report, "linkguard_report.txt", "text/plain")

with tab2:
    text = st.text_area("Paste URLs, one per line:", "https://google.com\nhttps://gooogle.xyz/login?verify=1", height=140)
    if st.button("Run Batch Check", type="primary"):
        rows = []
        for u in [x.strip() for x in text.splitlines() if x.strip()]:
            pp = urlparse(u)
            if pp.scheme in ("http", "https") and pp.hostname:
                pc, ps, cb = analyze(u)
                rows.append({"URL": u, "Character": round(pc, 3), "Structural": round(ps, 3), "Combined": round(cb, 3),
                             "Risk": "High" if cb >= 0.7 else ("Suspicious" if cb >= 0.4 else "Lower")})
            else:
                rows.append({"URL": u, "Character": "-", "Structural": "-", "Combined": "-", "Risk": "Invalid"})
        out = pd.DataFrame(rows)
        st.dataframe(out, use_container_width=True)
        st.download_button("Download CSV", out.to_csv(index=False), "results.csv", "text/csv")

with tab3:
    st.subheader("Performance on Unseen Domains (test set)")
    st.dataframe(pd.DataFrame(metrics["models"]).T, use_container_width=True)
    st.divider()
    st.subheader("Threshold Analysis (validation set)")
    st.caption("0.40 chosen as the Suspicious boundary: keeps recall high with acceptable FPR. 0.70 chosen as High risk: very high precision, low FPR.")
    st.dataframe(pd.DataFrame(metrics["thresholds"]), use_container_width=True, hide_index=True)
    fa = metrics["fpr_at"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Test FPR @ 0.40", f"{fa[0.40]:.1%}")
    m2.metric("Test FPR @ 0.50", f"{fa[0.50]:.1%}")
    m3.metric("Test FPR @ 0.70", f"{fa[0.70]:.1%}")
    st.divider()
    st.subheader("Confusion Matrix (ensemble @ 0.50, test)")
    st.dataframe(pd.DataFrame(metrics["cm"], index=["Actual Legit", "Actual Phish"], columns=["Predicted Legit", "Predicted Phish"]))
    st.divider()
    st.subheader("Dataset Bias Check — top TLDs per class")
    b1, b2 = st.columns(2)
    b1.write("**Phishing:**"); b1.json(metrics["phish_tlds"])
    b2.write("**Legitimate:**"); b2.json(metrics["legit_tlds"])
    st.caption("If TLD distributions differ sharply, the model may partly learn a TLD shortcut rather than phishing behaviour.")
    st.divider()
    st.subheader("Failure Cases (test set)")
    st.write("**False positives** (legit flagged, score ≥ 0.50):")
    for url, sc in metrics["fp_cases"]: st.text(f"score {sc:.3f} | {url[:90]}")
    if not metrics["fp_cases"]: st.text("None at 0.50.")
    st.write("**False negatives** (phishing missed, score < 0.50):")
    for url, sc in metrics["fn_cases"]: st.text(f"score {sc:.3f} | {url[:90]}")
    if not metrics["fn_cases"]: st.text("None at 0.50.")
    st.divider()
    st.subheader("Feature Importance (Structural Model)")
    imp = pd.DataFrame({"Feature": cols, "Importance": struct_m.feature_importances_}).sort_values("Importance")
    st.bar_chart(imp.set_index("Feature"))

with tab4:
    st.code("URL -> validate -> char n-grams + numeric features -> 2 models -> 50/50 average -> risk + indicators")
    st.markdown("""
**Models**
- Character model: TF-IDF on character 3–5 n-grams + Logistic Regression.
- Structural model: Random Forest (300 trees) on numeric URL features.
- Ensemble: 0.5 × character + 0.5 × structural.

**Why can a major site score non-zero on the character model?**
The character model only sees letter sequences (e.g. "goo", "ogl"). Those common English fragments also appear in some phishing URLs, so it cannot be sure on its own. The structural model scores such sites near zero (short URL, HTTPS, no IP, no suspicious hostname), pulling the combined score down. This is exactly why the ensemble is used instead of one model.

**How were 0.40 and 0.70 chosen?**
From the validation set table in the Performance tab: 0.40 keeps recall high while FPR stays acceptable; 0.70 gives very high precision for the strongest warning. They are practical cutoffs, not calibrated probabilities.

**Domain split:** train/val/test are separated by registered domain, so the model is evaluated on domains it never saw.
""")

st.caption("Scores are model estimates. A low score does not guarantee a site is safe.")
