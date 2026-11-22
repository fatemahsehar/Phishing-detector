import difflib, io, ipaddress, math, random, zipfile
from collections import Counter
from urllib.parse import urlparse
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

st.set_page_config(page_title="LinkGuard", page_icon="🛡️", layout="wide")

# ── Live Public Data Sources ──
OPENPHISH_URL = "https://openphish.com/feed.txt"
PHISHING_DB_URL = "https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/master/phishing-links-ACTIVE.txt"
TRANCO_ZIP_URL = "https://tranco-list.eu/top-1m.csv.zip"
OPENDNS_TOP_URL = "https://raw.githubusercontent.com/opendns/public-domain-lists/master/opendns-top-domains.txt"

FALLBACK_DOMAINS = [
    "google.com","youtube.com","facebook.com","amazon.com","wikipedia.org",
    "instagram.com","linkedin.com","reddit.com","netflix.com","microsoft.com",
    "apple.com","github.com","stackoverflow.com","wordpress.org","ebay.com",
    "paypal.com","bing.com","yahoo.com","twitch.tv","cnn.com","bbc.com",
    "nytimes.com","theguardian.com","reuters.com","forbes.com","bloomberg.com",
    "medium.com","quora.com","slack.com","zoom.us","dropbox.com","adobe.com",
    "shopify.com","stripe.com","cloudflare.com","notion.so","figma.com",
    "canva.com","discord.com","spotify.com","imdb.com","airbnb.com",
    "booking.com","uber.com","indeed.com","coursera.org","edx.org",
    "khanacademy.org","udemy.com","duolingo.com","kaggle.com","huggingface.co",
    "openai.com","perplexity.ai","claude.ai","arena.ai","vercel.app",
    "netlify.app","supabase.com","docker.com","gitlab.com","linear.app",
    "chase.com","wellsfargo.com","bankofamerica.com","coinbase.com","binance.com",
]

BRANDS = [
    "google","microsoft","apple","amazon","paypal","netflix","facebook",
    "chase","instagram","youtube","whatsapp","binance","linkedin","outlook",
    "office365","dropbox","coinbase","dhl","fedex","wellsfargo","bankofamerica",
    "icloud","steam","spotify","netflix","adobe","github","stripe"
]
SUSPICIOUS = [
    "login","signin","verify","update","secure","account","confirm",
    "password","credential","banking","wallet","recover","suspend","unlock",
    "billing","invoice","auth","validate","security","webscr"
]
RISKY_TLD = {"xyz","top","click","site","online","ml","ga","cf","tk","work","live","buzz","icu","cyou","cfd","sbs","gq","pw","cc","club","info","vip"}
COMMON_TLD = {"com","org","net","edu","gov","ai","io","app","dev","co","me","so","uk","de","fr","ca","au","jp","in","nl","es","it","br"}
SECOND_SUFFIX = {"co.uk","org.uk","ac.uk","gov.uk","com.au","co.in","com.br","co.jp","com.mx","com.tr","co.za","com.sg"}
TOPICS = ["technology","students","learning","health","climate","business","sports","science","education","travel","music","history","digital","safety","research","economy","design","coding","guide","news"]


def safe_urlparse(url):
    try:
        return urlparse(str(url).strip())
    except ValueError:
        return None


def is_valid_http_url(url):
    u = str(url).strip()
    if not u or len(u) > 2048 or not u.isprintable() or any(c.isspace() for c in u):
        return False
    p = safe_urlparse(u)
    if p is None or p.scheme not in ("http", "https") or not p.hostname:
        return False
    try:
        _ = p.port
    except ValueError:
        return False
    return True


def is_ip(host):
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def registered_from_host(host):
    if not host or is_ip(host):
        return host
    parts = host.lower().split(".")
    if len(parts) <= 2:
        return ".".join(parts)
    if ".".join(parts[-2:]) in SECOND_SUFFIX and len(parts) >= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def main_label(host):
    reg = registered_from_host(host)
    return reg.split(".")[0] if reg and not is_ip(host) else ""


def subdomain_from_host(host):
    h = (host or "").lower()
    reg = registered_from_host(h)
    return h[:-(len(reg) + 1)] if h and reg and h.endswith("." + reg) else ""


def normalize_lookalike(text):
    return (
        text.lower()
        .replace("0", "o")
        .replace("1", "l")
        .replace("3", "e")
        .replace("4", "a")
        .replace("5", "s")
        .replace("7", "t")
        .replace("8", "b")
        .replace("rn", "m")
        .replace("vv", "w")
    )


def calc_entropy(text):
    if not text:
        return 0.0
    freq = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def brand_lookalike_score(label):
    if not label:
        return 0.0, False, ""
    low = label.lower()
    norm = normalize_lookalike(low)
    best_sim = 0.0
    matched_brand = ""
    is_near = False
    for b in BRANDS:
        if low == b:
            return 1.0, False, b
        # Direct digit/character swap match (e.g., paypa1 -> paypal)
        if norm == b and low != b:
            return 0.99, True, b
        # Brand embedded with extra words/hyphens (e.g., paypal-verify)
        if b in norm and low != b and len(low) <= len(b) + 12:
            is_near = True
            if 0.90 > best_sim:
                best_sim = 0.90
                matched_brand = b
        sim = max(
            difflib.SequenceMatcher(None, b, low).ratio(),
            difflib.SequenceMatcher(None, b, norm).ratio(),
        )
        if sim > best_sim:
            best_sim = sim
            matched_brand = b
        if sim >= 0.80 and low != b:
            is_near = True
    return best_sim, is_near, matched_brand


def features(url):
    p = safe_urlparse(url)
    if p is None:
        p = urlparse("")
    h = (p.hostname or "").lower()
    path = p.path or ""
    query = p.query or ""
    low = url.lower()
    try:
        port = p.port
    except ValueError:
        port = None
    sub = subdomain_from_host(h)
    label = main_label(h)
    tld = h.rsplit(".", 1)[-1] if "." in h and not is_ip(h) else ""
    url_len = max(len(url), 1)
    digits = sum(c.isdigit() for c in url)
    specials = sum(not c.isalnum() for c in url)
    sim, is_near, _ = brand_lookalike_score(label)

    return {
        "url_length": len(url),
        "host_length": len(h),
        "domain_length": len(label),
        "path_length": len(path),
        "query_length": len(query),
        "dots": url.count("."),
        "hyphens": url.count("-"),
        "hyphens_in_host": h.count("-"),
        "underscores": url.count("_"),
        "digits": digits,
        "digits_in_host": sum(c.isdigit() for c in h),
        "digit_ratio": round(digits / url_len, 4),
        "special_ratio": round(specials / url_len, 4),
        "params": len(query.split("&")) if query else 0,
        "path_depth": path.count("/"),
        "percent_encoded": url.count("%"),
        "has_ip": int(is_ip(h)),
        "has_at": int("@" in url),
        "has_punycode": int("xn--" in low),
        "nonstandard_port": int(port not in (None, 80, 443)),
        "is_https": int(p.scheme == "https"),
        "is_www": int(sub == "www"),
        "is_homepage": int(path in ("", "/") and not query),
        "subdomain_count": len(sub.split(".")) if sub else 0,
        "suspicious_words": sum(1 for w in SUSPICIOUS if w in low),
        "host_keywords": sum(1 for w in SUSPICIOUS if w in h),
        "path_keywords": sum(1 for w in SUSPICIOUS if w in (path + query).lower()),
        "risky_tld": int(tld in RISKY_TLD),
        "is_common_tld": int(tld in COMMON_TLD),
        "exact_brand_domain": int(label in BRANDS),
        "near_brand_domain": int(is_near),
        "brand_similarity": round(float(sim), 3),
        "brand_in_subdomain": int(any(b in sub for b in BRANDS) and label not in BRANDS) if sub else 0,
        "brand_in_path": int(any(b in (path + query).lower() for b in BRANDS) and label not in BRANDS),
        "url_entropy": round(calc_entropy(url), 3),
        "host_entropy": round(calc_entropy(h), 3),
        "has_double_slash_path": int("//" in path),
    }


def domain_info(url):
    p = safe_urlparse(url) or urlparse("")
    h = (p.hostname or "").lower()
    return {
        "Hostname": h,
        "Registered domain": registered_from_host(h),
        "Subdomain": subdomain_from_host(h) or "-",
        "Main domain name": main_label(h),
        "Scheme": p.scheme,
        "Path": p.path or "/",
        "Uses IP address": is_ip(h),
    }


def make_realistic_legit_url(domain, rng):
    """Build realistic legitimate URLs from real Tranco domains (mix of homepages, articles, queries, logins)."""
    kind = rng.random()
    sub = rng.choice(["", "", "", "www.", "www.", "app.", "docs.", "blog.", "support.", "m.", "shop."])
    if kind < 0.30:
        return f"https://{sub}{domain}/"
    if kind < 0.55:
        path = rng.choice([
            "/about", "/contact", "/pricing", "/features", "/products",
            "/services", "/blog", "/news", "/help", "/faq", "/careers",
            "/privacy", "/terms", "/download", "/community", "/docs/getting-started",
        ])
        return f"https://{sub}{domain}{path}"
    if kind < 0.78:
        slug = "-".join(rng.choices(TOPICS, k=rng.randint(3, 6)))
        section = rng.choice(["blog", "news", "articles", "guide", "products", "docs"])
        return f"https://{sub}{domain}/{section}/{slug}-{rng.randint(100, 99999)}"
    if kind < 0.90:
        q = "+".join(rng.choices(TOPICS, k=rng.randint(2, 4)))
        return f"https://{sub}{domain}/search?q={q}&page={rng.randint(1, 10)}&lang=en"
    # Legitimate login / account pages on real domains
    auth_path = rng.choice(["/login", "/signin", "/account/settings", "/accounts/login/", "/profile", "/dashboard"])
    return f"https://{sub}{domain}{auth_path}"


def make_lookalike_phish(rng):
    """Small supplement of brand-spoofing lookalikes (like paypa1.com) that are rare in daily feeds."""
    brand = rng.choice(BRANDS)
    swaps = [("o", "0"), ("l", "1"), ("e", "3"), ("a", "4"), ("s", "5"), ("i", "1")]
    fake = brand
    for a, b in swaps:
        if a in fake and rng.random() < 0.6:
            fake = fake.replace(a, b, 1)
            break
    if fake == brand:
        i = rng.randrange(len(brand))
        fake = brand[:i] + brand[i] + brand[i:]  # double letter like gooogle
    tld = rng.choice(["com", "net", "org", "xyz", "top", "site", "online", "co", "io"])
    pattern = rng.randrange(5)
    if pattern == 0:
        return f"https://{fake}.{tld}/"
    if pattern == 1:
        return f"https://{fake}.{tld}/login"
    if pattern == 2:
        return f"https://{brand}-secure.{tld}/verify/account"
    if pattern == 3:
        return f"https://{brand}.verify-account.{fake}.{tld}/signin"
    return f"http://{rng.randint(11, 220)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}/{brand}/login"


def load_real_data():
    rng = random.Random(42)
    phish_urls = []
    phish_domain_counts = Counter()
    phish_sources = []

    def add_phish(u):
        if not is_valid_http_url(u):
            return False
        h = (safe_urlparse(u).hostname or "").lower()
        reg = registered_from_host(h)
        if phish_domain_counts[reg] >= 2:
            return False
        phish_domain_counts[reg] += 1
        phish_urls.append(u)
        return True

    # 1. OpenPhish live verified feed
    openphish_count = 0
    try:
        r = requests.get(OPENPHISH_URL, timeout=12)
        if r.status_code == 200:
            for line in r.text.splitlines():
                if add_phish(line.strip()):
                    openphish_count += 1
            if openphish_count:
                phish_sources.append(f"OpenPhish ({openphish_count:,})")
    except Exception:
        pass

    # 2. Phishing.Database active feed (streamed to stay fast & low-memory)
    phishdb_count = 0
    try:
        with requests.get(PHISHING_DB_URL, stream=True, timeout=20) as r:
            if r.status_code == 200:
                candidates = []
                for idx, raw_line in enumerate(r.iter_lines(decode_unicode=True)):
                    if idx >= 35000:
                        break
                    if raw_line:
                        u = raw_line.strip()
                        if u.startswith("http"):
                            candidates.append(u)
                rng.shuffle(candidates)
                for u in candidates:
                    if len(phish_urls) >= 5200:
                        break
                    if add_phish(u):
                        phishdb_count += 1
                if phishdb_count:
                    phish_sources.append(f"Phishing.Database ({phishdb_count:,})")
    except Exception:
        pass

    # Add targeted lookalike phishing patterns (e.g. paypa1.com) for subtle brand spoofs
    lookalike_target = 500 if len(phish_urls) >= 1000 else 3000
    for _ in range(lookalike_target):
        phish_urls.append(make_lookalike_phish(rng))
    if not phish_sources:
        phish_sources.append(f"Synthetic fallback ({len(phish_urls):,})")
    else:
        phish_sources.append(f"Lookalike patterns ({lookalike_target})")

    # 3. Real legitimate domains from Tranco Top 1M zip
    legit_domains = []
    legit_source = ""
    try:
        r = requests.get(TRANCO_ZIP_URL, timeout=20)
        if r.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                fname = zf.namelist()[0]
                with zf.open(fname) as f:
                    for idx, raw_line in enumerate(f):
                        if idx >= 12000:
                            break
                        parts = raw_line.decode("utf-8", errors="ignore").strip().split(",")
                        if len(parts) >= 2 and "." in parts[1]:
                            legit_domains.append(parts[1].strip().lower())
            if legit_domains:
                legit_source = f"Tranco Top 1M ({len(legit_domains):,} real domains sampled)"
    except Exception:
        pass

    # Fallback legitimate source: OpenDNS Top 10k
    if len(legit_domains) < 500:
        try:
            r = requests.get(OPENDNS_TOP_URL, timeout=12)
            if r.status_code == 200:
                for line in r.text.splitlines():
                    d = line.strip().lower()
                    if d and "." in d:
                        legit_domains.append(d)
                if legit_domains:
                    legit_source = f"OpenDNS Top 10k ({len(legit_domains):,} real domains)"
        except Exception:
            pass

    if len(legit_domains) < 100:
        legit_domains = FALLBACK_DOMAINS.copy()
        legit_source = f"Built-in top domains ({len(legit_domains)})"

    # Match legitimate count to phishing count (balanced classes, diverse real domains)
    rng.shuffle(legit_domains)
    target_legit = len(phish_urls)
    legit_urls = []
    for i in range(target_legit):
        d = legit_domains[i % len(legit_domains)]
        legit_urls.append(make_realistic_legit_url(d, rng))

    source = {
        "phish": " + ".join(phish_sources),
        "legit": legit_source,
    }
    return phish_urls, legit_urls, source


@st.cache_resource(show_spinner="Downloading real feeds and training models... (first load takes ~15 seconds)")
def train_models():
    phish_urls, legit_urls, source = load_real_data()

    urls = legit_urls + phish_urls
    labels = [0] * len(legit_urls) + [1] * len(phish_urls)
    df = pd.DataFrame({"url": urls, "label": labels})
    df["group"] = df["url"].apply(lambda u: registered_from_host((safe_urlparse(u).hostname or "") if safe_urlparse(u) else ""))

    # Domain-separated train/test split (75% train, 25% unseen domains test)
    groups = list(df["group"].unique())
    random.Random(123).shuffle(groups)
    test_groups = set(groups[:max(1, int(len(groups) * 0.25))])
    test = df["group"].isin(test_groups)
    train = ~test

    # Model 1: Character-level TF-IDF + Logistic Regression
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=18000, min_df=2, sublinear_tf=True)
    char_m = LogisticRegression(max_iter=1500, C=2.0, class_weight="balanced")
    char_m.fit(vec.fit_transform(df.loc[train, "url"]), df.loc[train, "label"])

    # Model 2: Structural Random Forest
    x_all = pd.DataFrame([features(u) for u in df["url"]])
    cols = list(x_all.columns)
    struct_m = RandomForestClassifier(
        n_estimators=250, max_depth=22, min_samples_leaf=2,
        random_state=42, n_jobs=-1, class_weight="balanced"
    )
    struct_m.fit(x_all[train], df.loc[train, "label"])

    # Evaluate on unseen domains
    y = df.loc[test, "label"].values
    pc = char_m.predict_proba(vec.transform(df.loc[test, "url"]))[:, 1]
    ps = struct_m.predict_proba(x_all[test])[:, 1]
    ens = 0.5 * pc + 0.5 * ps

    # Threshold table
    thresholds = []
    for t in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        pred = (ens >= t).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        thresholds.append({
            "Threshold": t,
            "Precision": round(float(pr), 3),
            "Recall": round(float(rc), 3),
            "F1": round(float(f1), 3),
            "False Positive Rate": round(fp / max(fp + tn, 1), 3),
            "False Negatives": int(fn),
        })

    # Real failure cases from unseen test domains
    test_df = df.loc[test].copy()
    test_df["score"] = ens
    fp_cases = test_df[(y == 0) & (ens >= 0.5)].nlargest(4, "score")[["url", "score"]].values.tolist()
    fn_cases = test_df[(y == 1) & (ens < 0.5)].nsmallest(4, "score")[["url", "score"]].values.tolist()

    # TLD bias check
    train_df = df.loc[train]
    def top_tlds(series):
        return (
            series.apply(lambda u: (safe_urlparse(u).hostname or "").lower() if safe_urlparse(u) else "")
            .apply(lambda h: h.rsplit(".", 1)[-1] if "." in h else h)
            .value_counts()
            .head(5)
            .to_dict()
        )

    metrics = {
        "train": int(train.sum()),
        "test": int(test.sum()),
        "unique_test_domains": int(df.loc[test, "group"].nunique()),
        "models": {},
        "source": source,
    }
    for name, prob in [("Character Model", pc), ("Structural Model", ps), ("Ensemble (50/50)", ens)]:
        pred = (prob >= 0.5).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        metrics["models"][name] = {
            "Precision": round(float(pr), 3),
            "Recall": round(float(rc), 3),
            "F1": round(float(f1), 3),
            "Accuracy": round(float(accuracy_score(y, pred)), 3),
        }
    cm = confusion_matrix(y, (ens >= 0.5).astype(int), labels=[0, 1]).tolist()
    metrics["cm"] = cm
    metrics["fpr"] = round(cm[0][1] / max(cm[0][0] + cm[0][1], 1), 3)
    metrics["thresholds"] = thresholds
    metrics["fp_cases"] = fp_cases
    metrics["fn_cases"] = fn_cases
    metrics["phish_tlds"] = top_tlds(train_df[train_df["label"] == 1]["url"])
    metrics["legit_tlds"] = top_tlds(train_df[train_df["label"] == 0]["url"])

    return vec, char_m, struct_m, cols, metrics


vec, char_m, struct_m, cols, metrics = train_models()


def indicators(url):
    p = safe_urlparse(url) or urlparse("")
    h = (p.hostname or "").lower()
    f = features(url)
    notes = []
    if f["has_ip"]:
        notes.append("Uses an IP address instead of a domain name")
    if p.scheme != "https":
        notes.append("Does not use HTTPS")
    if "@" in url:
        notes.append("Contains '@' — can hide the real destination")
    if "xn--" in url.lower():
        notes.append("Punycode domain — can imitate a known brand")
    if f["subdomain_count"] >= 2:
        notes.append("Has multiple subdomains")
    if f["hyphens_in_host"] >= 2:
        notes.append("Multiple hyphens in hostname")
    if f["digits_in_host"] >= 4:
        notes.append("Unusual number of digits in hostname")
    if f["host_keywords"] > 0:
        notes.append("Suspicious word (e.g. login/verify/secure) inside the domain name")
    elif f["path_keywords"] > 0:
        notes.append("Contains account/login keyword in path or query")
    if f["risky_tld"]:
        notes.append("Uses a TLD frequently abused in phishing campaigns")
    if f["url_entropy"] > 4.6:
        notes.append("High character randomness (entropy) in URL")

    label = main_label(h)
    sub = subdomain_from_host(h)
    sim, is_near, matched = brand_lookalike_score(label)
    if is_near and matched:
        notes.append(f"Lookalike domain: '{label}' closely resembles '{matched}'")
    if sub:
        for b in BRANDS:
            if b in sub and label != b:
                notes.append(f"Known brand '{b}' appears in subdomain instead of main domain")
                break
    return notes


def analyze(url):
    pc = float(char_m.predict_proba(vec.transform([url]))[0][1])
    ps = float(struct_m.predict_proba(pd.DataFrame([features(url)])[cols])[0][1])
    return pc, ps, 0.5 * pc + 0.5 * ps


def risk_label(url, score):
    h = (safe_urlparse(url).hostname or "").lower() if safe_urlparse(url) else ""
    _, is_near, _ = brand_lookalike_score(main_label(h))
    if score >= 0.70:
        return "High phishing risk"
    if score >= 0.40 or is_near:
        return "Suspicious — investigate carefully"
    return "Lower model-estimated risk"


def set_url(u):
    st.session_state["url_input"] = u


if "url_input" not in st.session_state:
    st.session_state["url_input"] = "https://google.com"

# ── UI Styling ──
st.markdown("""
<style>
.stApp { background: linear-gradient(160deg, #0a0f1e 0%, #111827 50%, #0f172a 100%); }
[data-testid="stHeader"] { background: rgba(10, 15, 30, 0.85); backdrop-filter: blur(12px); }
section[data-testid="stSidebar"] { background: #0c1322; border-right: 1px solid #1e293b; }
section[data-testid="stSidebar"] * { color: #cbd5e1; }
.scan-hero {
    padding: 2.4rem 2.4rem 2rem; margin-bottom: 1.2rem; border: 1px solid #1e3a5f;
    border-radius: 20px; background: linear-gradient(135deg, #0c2340 0%, #0f172a 60%, #1a1040 100%);
    position: relative; overflow: hidden;
}
.scan-hero::before {
    content: ""; position: absolute; top: -50%; right: -20%; width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%); border-radius: 50%;
}
.scan-hero h1 { color: #f1f5f9; font-size: clamp(2rem, 5vw, 3.2rem); line-height: 1.1; margin: 0.4rem 0; padding: 0; font-weight: 800; }
.scan-hero p { color: #94a3b8; max-width: 36rem; margin: 0.5rem 0 0; font-size: 1.05rem; line-height: 1.6; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: transparent; }
.stTabs [data-baseweb="tab"] { background: #1e293b; border-radius: 10px; padding: 10px 20px; color: #94a3b8; font-weight: 600; border: 1px solid #334155; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #0ea5e9, #6366f1); color: white; border-color: transparent; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #0ea5e9, #6366f1) !important; border: none !important; border-radius: 12px !important; font-weight: 700 !important; font-size: 1rem !important; padding: 0.7rem 1.5rem !important; color: white !important; }
.stButton > button:not([kind="primary"]) { background: #1e293b !important; border: 1px solid #334155 !important; border-radius: 10px !important; color: #e2e8f0 !important; }
.stTextInput > div > div > input { background: #0f172a !important; border: 1px solid #334155 !important; border-radius: 12px !important; color: #f1f5f9 !important; padding: 0.8rem 1rem !important; font-size: 1rem !important; }
[data-testid="stMetric"] { background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid #334155; border-radius: 14px; padding: 1.2rem; }
[data-testid="stMetric"] label { color: #94a3b8 !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.8rem !important; }
[data-testid="stExpander"] { background: #111827; border: 1px solid #1e293b; border-radius: 12px; }
.stProgress > div > div > div { background: linear-gradient(90deg, #0ea5e9, #6366f1) !important; border-radius: 999px !important; }
.stDownloadButton > button { background: #1e293b !important; border: 1px solid #334155 !important; border-radius: 10px !important; color: #38bdf8 !important; font-weight: 600 !important; }
.stTextArea > div > div > textarea { background: #0f172a !important; border: 1px solid #334155 !important; border-radius: 12px !important; color: #f1f5f9 !important; }
.stCaption { color: #64748b !important; }
</style>
<section class="scan-hero" aria-labelledby="scan-title">
  <h1 id="scan-title">Inspect the link.<br>Before the click.</h1>
  <p>Paste a URL to check for signs of phishing without visiting the destination.</p>
</section>
""", unsafe_allow_html=True)

src = metrics["source"]
st.caption(
    f"Phishing: {src['phish']} · Legitimate: {src['legit']} · "
    f"Train: {metrics['train']:,} · Test: {metrics['test']:,} ({metrics['unique_test_domains']:,} unseen domains)"
)

with st.sidebar:
    st.header("How to read a result")
    st.markdown(
        "- **High risk:** score of 0.70 or above.\n"
        "- **Suspicious:** score of 0.40 to below 0.70, or a brand lookalike detected.\n"
        "- **Lower risk:** below 0.40 with no lookalike warning."
    )
    st.divider()
    st.caption("Scores are model estimates, not calibrated probabilities.")

tab1, tab2, tab3, tab4 = st.tabs(["Scan a URL", "Batch Check", "Model Performance", "How It Works"])

with tab1:
    with st.container(border=True):
        st.subheader("Check a link")
        examples = [
            ("Regular link", "https://arena.ai"),
            ("Lookalike link", "https://paypa1.com/"),
            ("IP address link", "http://192.0.2.25/bank/login"),
        ]
        for col, (label, example) in zip(st.columns(3), examples):
            col.button(label, on_click=set_url, args=(example,), use_container_width=True)
        url = st.text_input("URL to inspect", key="url_input", max_chars=2048)
        submitted = st.button("Analyze URL", type="primary", use_container_width=True)

    if submitted:
        candidate = url.strip()
        st.session_state.pop("result", None)
        if not is_valid_http_url(candidate):
            st.error("Enter a complete HTTP or HTTPS URL, like https://example.org/.")
        else:
            pc, ps, combined = analyze(candidate)
            st.session_state["result"] = {
                "url": candidate, "char": pc, "struct": ps,
                "combined": combined, "notes": indicators(candidate),
                "info": domain_info(candidate),
            }

    r = st.session_state.get("result")
    if r and r["url"] == url.strip():
        verdict = risk_label(r["url"], r["combined"])
        st.subheader("Assessment")
        if verdict.startswith("High"):
            st.error(verdict)
        elif verdict.startswith("Suspicious"):
            st.warning(verdict)
        else:
            st.success(verdict)
        st.code(r["url"], language=None, wrap_lines=True)

        a, b, c = st.columns(3)
        a.metric("Character model", f"{r['char']:.3f}")
        b.metric("Structural model", f"{r['struct']:.3f}")
        c.metric("Combined score", f"{r['combined']:.3f}")
        st.progress(min(max(r["combined"], 0.0), 1.0))
        st.caption("Combined = 0.5 × Character + 0.5 × Structural. Scores are model estimates, not probabilities.")

        with st.expander("Rule-based indicators", expanded=True):
            for note in r["notes"]:
                st.text(f"• {note}")
            if not r["notes"]:
                st.write("No rule-based indicators triggered.")
        with st.expander("Domain details"):
            st.json(r["info"])

        report = (
            f"LinkGuard Report\n\nURL: {r['url']}\nAssessment: {verdict}\n"
            f"Character: {r['char']:.3f}\nStructural: {r['struct']:.3f}\n"
            f"Combined: {r['combined']:.3f}\n\nIndicators:\n"
            + "\n".join("- " + n for n in (r["notes"] or ["None"]))
        )
        st.download_button("Download report", report, "linkguard_report.txt", "text/plain")

with tab2:
    text = st.text_area("Paste URLs, one per line:", "https://google.com\nhttps://arena.ai\nhttps://paypa1.com/\nhttps://gooogle.xyz/login?verify=1", height=140)
    if st.button("Run Batch Check", type="primary"):
        rows = []
        for u in [x.strip() for x in text.splitlines() if x.strip()]:
            if is_valid_http_url(u):
                pc, ps, cb = analyze(u)
                rows.append({
                    "URL": u,
                    "Character": round(pc, 3),
                    "Structural": round(ps, 3),
                    "Combined": round(cb, 3),
                    "Risk": risk_label(u, cb),
                })
            else:
                rows.append({"URL": u, "Character": "-", "Structural": "-", "Combined": "-", "Risk": "Invalid"})
        out = pd.DataFrame(rows)
        st.dataframe(out, use_container_width=True)
        st.download_button("Download CSV", out.to_csv(index=False), "results.csv", "text/csv")

with tab3:
    st.subheader("Performance on Unseen Domains")
    perf = pd.DataFrame(metrics["models"]).T
    st.dataframe(perf, use_container_width=True)

    st.divider()
    st.subheader("Threshold Analysis")
    st.caption("Precision, recall, and false positive rate on unseen domains across score cutoffs.")
    thresh_df = pd.DataFrame(metrics["thresholds"])
    st.dataframe(thresh_df, use_container_width=True, hide_index=True)

    m1, m2, m3 = st.columns(3)
    t40 = next(t for t in metrics["thresholds"] if t["Threshold"] == 0.40)
    t70 = next(t for t in metrics["thresholds"] if t["Threshold"] == 0.70)
    t50 = next(t for t in metrics["thresholds"] if t["Threshold"] == 0.50)
    m1.metric("FPR at 0.40", f"{t40['False Positive Rate']:.1%}")
    m2.metric("FPR at 0.70", f"{t70['False Positive Rate']:.1%}")
    m3.metric("Recall at 0.50", f"{t50['Recall']:.1%}")

    st.divider()
    st.subheader("Confusion Matrix (Ensemble at 0.50)")
    st.dataframe(pd.DataFrame(metrics["cm"], index=["Actual Legit", "Actual Phish"], columns=["Predicted Legit", "Predicted Phish"]))

    st.divider()
    st.subheader("Dataset Bias Check")
    st.caption("Compares top TLDs in phishing vs. legitimate training data.")
    bc1, bc2 = st.columns(2)
    bc1.write("**Phishing TLDs (top 5):**")
    bc1.json(metrics["phish_tlds"])
    bc2.write("**Legitimate TLDs (top 5):**")
    bc2.json(metrics["legit_tlds"])

    st.divider()
    st.subheader("Failure Cases from Unseen Test Domains")
    st.write("**False Positives** (legitimate URLs wrongly flagged):")
    if metrics["fp_cases"]:
        for url_case, score in metrics["fp_cases"]:
            st.text(f"  Score: {score:.3f}  |  {url_case[:90]}")
    else:
        st.text("  None at 0.50 threshold.")
    st.write("**False Negatives** (phishing URLs missed):")
    if metrics["fn_cases"]:
        for url_case, score in metrics["fn_cases"]:
            st.text(f"  Score: {score:.3f}  |  {url_case[:90]}")
    else:
        st.text("  None at 0.50 threshold.")

    st.divider()
    st.subheader("Feature Importance (Structural Model)")
    imp = pd.DataFrame({"Feature": cols, "Importance": struct_m.feature_importances_}).sort_values("Importance")
    st.bar_chart(imp.set_index("Feature"))

with tab4:
    st.code("URL -> Validate -> Extract features + Character n-grams -> 2 Models -> 50/50 Average -> Risk + Indicators")
    st.markdown("""
**Models used:**
- **Character Model:** TF-IDF (3–5 character n-grams, 18,000 features) + Logistic Regression. Learns suspicious character sequences directly from real phishing URLs.
- **Structural Model:** Random Forest (250 trees) on numerical URL features (lengths, dots, hyphens, digits, entropy, HTTPS, IP check, subdomain depth, keyword location, and brand similarity).
- **Ensemble:** 50/50 average of both model scores.

**Why can a major site like google.com still have a non-zero character score?**
The character model only sees short 3–5 character fragments. Common English letter groups appear in both phishing and legitimate URLs, so the character model alone can be uncertain. The structural model sees a short, clean HTTPS domain with no suspicious indicators and scores it near zero, keeping the combined score low.

**How were the 0.40 and 0.70 thresholds chosen?**
From the Threshold Analysis table in the Performance tab: 0.40 catches most phishing (high recall) while keeping false positives low, and 0.70 achieves high precision for the High Risk label.

**Domain-separated testing:**
All URLs are grouped by registered domain before splitting 75/25 into train and test sets, so the model is always evaluated on domains it has never seen.

**Data sources:**
- Phishing URLs: OpenPhish verified feed (openphish.com) + Phishing.Database active feed + lookalike patterns
- Legitimate URLs: Tranco Top 1M domains (tranco-list.eu) with realistic path variations
""")

st.caption("Scores are model estimates. A low score does not guarantee a site is safe.")
