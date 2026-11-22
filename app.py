import difflib, ipaddress, math, random
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

OPENPHISH_URL = "https://openphish.com/feed.txt"
UA = {"User-Agent": "Mozilla/5.0 (LinkGuard research project)"}
ENGINE = "v4"

BRANDS = ["google","microsoft","apple","amazon","paypal","netflix","facebook","chase","instagram","youtube","whatsapp","binance","linkedin","outlook","dropbox","coinbase","dhl","fedex","wellsfargo","bankofamerica","icloud","steam","spotify","adobe","github","stripe"]
SUSPICIOUS = ["login","signin","verify","update","secure","account","confirm","password","credential","banking","wallet","recover","suspend","unlock","billing","invoice","auth","validate","security"]
RISKY_TLD = {"xyz","top","click","site","online","ml","ga","cf","tk","work","live","buzz","icu","cyou","cfd","sbs","gq","pw"}
COMMON_TLD = {"com","org","net","edu","gov","ai","io","app","dev","co","me","so","uk","de","fr","ca","au","jp","in","nl","es","it","br"}
SECOND_SUFFIX = {"co.uk","org.uk","ac.uk","gov.uk","com.au","co.in","com.br","co.jp","com.mx","com.tr","co.za","com.sg"}
DEV_HOSTING = {"streamlit.app","vercel.app","netlify.app","github.io","firebaseapp.com","pages.dev","workers.dev","herokuapp.com","glitch.me","web.app","stackblitz.com","repl.co","codepen.io","jsfiddle.net"}

KNOWN_LEGIT_DOMAINS = {
    "google.com","youtube.com","facebook.com","instagram.com","twitter.com","x.com","linkedin.com",
    "reddit.com","github.com","wikipedia.org","amazon.com","netflix.com","spotify.com","twitch.tv",
    "tiktok.com","pinterest.com","medium.com","quora.com","dropbox.com","adobe.com","microsoft.com",
    "apple.com","openai.com","chatgpt.com","notion.so","figma.com","canva.com","slack.com","zoom.us",
    "discord.com","steam.com","imdb.com","airbnb.com","booking.com","uber.com","yelp.com","indeed.com",
    "glassdoor.com","coursera.org","edx.org","khanacademy.org","udemy.com","duolingo.com","kaggle.com",
    "huggingface.co","stackoverflow.com","gitlab.com","bitbucket.org","vercel.com","netlify.com",
    "heroku.com","digitalocean.com","cloudflare.com","shopify.com","stripe.com","paypal.com",
    "chase.com","bankofamerica.com","coinbase.com","binance.com","bbc.com","cnn.com","reuters.com",
    "nytimes.com","theguardian.com","aljazeera.com","bloomberg.com","forbes.com","techcrunch.com",
    "wired.com","apnews.com","npr.org","wsj.com","washingtonpost.com","economist.com","stanford.edu",
    "mit.edu","harvard.edu","oxford.ac.uk","cambridge.ac.uk","gov.uk","usa.gov","nih.gov","arena.ai",
    "perplexity.ai","claude.ai","character.ai","together.ai","stability.ai","elevenlabs.io","wandb.ai",
    "replit.com","supabase.com","mongodb.com","docker.com","kubernetes.io","npmjs.com","pypi.org",
    "sideshift.app","changelly.com","coinbase.com","kraken.com","gemini.com","bitfinex.com",
}

WORD1 = ["bright","green","blue","river","city","global","smart","prime","united","national","digital","open","north","south","royal","happy","quick","fresh","great","modern"]
WORD2 = ["labs","health","learn","books","travel","foods","tech","media","news","shop","school","college","library","clinic","studio","works","solutions","group","garden","energy"]
TOPICS = ["technology","students","learning","health","climate","business","sports","science","education","travel","music","history","digital","safety","research","economy","design","coding","guide","news"]
SYLLABLES = ["a","be","ca","do","en","fi","go","ha","ka","lo","mi","no","pa","ra","re","ro","sa","ta","to","va","ve","za","ar","er","in","on","or","al","el","il","an","un","ex"]
DEV_HOSTS_WEIGHTED = ["streamlit.app","streamlit.app","streamlit.app","vercel.app","vercel.app","netlify.app","netlify.app","github.io","github.io","firebaseapp.com","pages.dev","workers.dev","herokuapp.com","glitch.me"]
DEV_NAMES = ["myapp","project","demo","site","app","tool","page","dashboard","portfolio","blog","api","store","learn","notes","docs","link","scan","check","guard","lab","work","portal","panel","hub","box"]
LEGIT_SUBS = ["","www.","app.","api.","portal.","my.","dashboard.","docs.","support.","m.","beta.","staging."]
LEGIT_QUERIES = ["","?tab=1","?page=2","?lang=en","?view=grid","?sort=recent","?filter=all","?q=hello","?ref=home","?utm_source=mail"]

LEGIT_QUERY_PARAMS = {"q","v","list","page","ref","lang","th","qid","index","hl","source","tab","s","utm_source","utm_medium","start","sort","filter","category","type","mode","view","from","to","next","sp","ei","ved","oq","gs_l","pb","gws","cad","ct","id"}
PHISH_QUERY_PARAMS = {"token","session","redirect","verify","auth","login","password","email","user","account","otp","code","pass","pwd","secret","key","sig","sign","callback","return","target","url"}


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
    return (text.lower().replace("0","o").replace("1","l").replace("3","e")
            .replace("4","a").replace("5","s").replace("7","t").replace("8","b")
            .replace("rn","m").replace("vv","w"))


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
    best_sim, matched, is_near = 0.0, "", False
    for b in BRANDS:
        if low == b:
            return 1.0, False, b
        if norm == b and low != b:
            return 0.99, True, b
        sim = max(difflib.SequenceMatcher(None, b, low).ratio(), difflib.SequenceMatcher(None, b, norm).ratio())
        if sim > best_sim:
            best_sim, matched = sim, b
        if sim >= 0.80 and low != b:
            is_near = True
    return best_sim, is_near, matched


def features(url):
    p = safe_urlparse(url) or urlparse("")
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
    reg = registered_from_host(h)
    tld = h.rsplit(".", 1)[-1] if "." in h and not is_ip(h) else ""
    url_len = max(len(url), 1)
    digits = sum(c.isdigit() for c in url)
    specials = sum(not c.isalnum() for c in url)
    sim, is_near, _ = brand_lookalike_score(label)
    qparams = set()
    if query:
        for pair in query.split("&"):
            k = pair.split("=", 1)[0].strip().lower()
            if k:
                qparams.add(k)
    return {
        "url_length": len(url), "host_length": len(h), "domain_length": len(label),
        "path_length": len(path), "query_length": len(query),
        "dots": url.count("."), "hyphens": url.count("-"), "hyphens_in_host": h.count("-"),
        "underscores": url.count("_"), "digits": digits, "digits_in_host": sum(c.isdigit() for c in h),
        "digit_ratio": round(digits / url_len, 4), "special_ratio": round(specials / url_len, 4),
        "params": len(query.split("&")) if query else 0, "path_depth": path.count("/"),
        "percent_encoded": url.count("%"), "has_ip": int(is_ip(h)), "has_at": int("@" in url),
        "has_punycode": int("xn--" in low), "nonstandard_port": int(port not in (None, 80, 443)),
        "is_https": int(p.scheme == "https"), "is_www": int(sub == "www"),
        "is_homepage": int(path in ("", "/") and not query),
        "subdomain_count": len(sub.split(".")) if sub else 0,
        "suspicious_words": sum(1 for w in SUSPICIOUS if w in low),
        "host_keywords": sum(1 for w in SUSPICIOUS if w in h),
        "path_keywords": sum(1 for w in SUSPICIOUS if w in (path + query).lower()),
        "risky_tld": int(tld in RISKY_TLD), "is_common_tld": int(tld in COMMON_TLD),
        "is_dev_hosting": int(reg in DEV_HOSTING),
        "is_known_legit_domain": int(reg in KNOWN_LEGIT_DOMAINS),
        "exact_brand_domain": int(label in BRANDS), "near_brand_domain": int(is_near),
        "brand_similarity": round(float(sim), 3),
        "brand_in_subdomain": int(any(b in sub for b in BRANDS) and label not in BRANDS) if sub else 0,
        "brand_in_path": int(any(b in (path + query).lower() for b in BRANDS) and label not in BRANDS),
        "legit_query_params": sum(1 for k in qparams if k in LEGIT_QUERY_PARAMS),
        "phish_query_params": sum(1 for k in qparams if k in PHISH_QUERY_PARAMS),
        "url_entropy": round(calc_entropy(url), 3), "host_entropy": round(calc_entropy(h), 3),
        "has_double_slash_path": int("//" in path),
    }


def domain_info(url):
    p = safe_urlparse(url) or urlparse("")
    h = (p.hostname or "").lower()
    reg = registered_from_host(h)
    return {"Hostname": h, "Registered domain": reg, "Subdomain": subdomain_from_host(h) or "-",
            "Main domain name": main_label(h), "Scheme": p.scheme, "Path": p.path or "/",
            "Uses IP address": is_ip(h),
            "Known dev-hosting platform": reg in DEV_HOSTING,
            "Known major/legit domain": reg in KNOWN_LEGIT_DOMAINS}


def misspell(w, rng):
    w = list(w); i = rng.randrange(len(w)); r = rng.random()
    if r < 0.4: w[i] = rng.choice("abcdefghijklmnopqrstuvwxyz")
    elif r < 0.7: w.insert(i, rng.choice("abcdefghijklmnopqrstuvwxyz"))
    elif r < 0.85 and len(w) > 3: w.pop(i)
    else:
        j = min(i + 1, len(w) - 1); w[i], w[j] = w[j], w[i]
    return "".join(w)


def digit_fake(w):
    if "o" in w: return w.replace("o", "0", 1)
    if "l" in w: return w.replace("l", "1", 1)
    if "e" in w: return w.replace("e", "3", 1)
    return w + "1"


def make_legit(rng):
    kind = rng.random()
    if kind < 0.20:
        host = rng.choice(DEV_HOSTS_WEIGHTED)
        name = rng.choice(DEV_NAMES)
        randhash = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=rng.randint(10, 16)))
        path = rng.choice(["", "/", "/about", "/docs", "/settings", "/v2", "/help"])
        return f"https://{name}-{randhash}.{host}{path}{rng.choice(LEGIT_QUERIES)}"
    if kind < 0.34:
        base = rng.choice(sorted(KNOWN_LEGIT_DOMAINS))
        sub = rng.choice(LEGIT_SUBS)
        p = rng.choice([
            f"/watch?v={''.join(rng.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_', k=11))}&list=PLabc&index={rng.randint(1,50)}",
            f"/search?q={'+'.join(rng.choices(TOPICS, k=4))}&page={rng.randint(1,10)}&lang=en",
            f"/dp/B{rng.randint(1000000,9999999)}?ref=sr_1_2&th=1&qid={rng.randint(100,999)}",
            f"/questions/{rng.randint(100000,9999999)}/how-to-do-something-in-python",
            f"/wiki/{rng.choice(['Python','Machine_learning','Cybersecurity','Web_development','Data_science','Algorithms'])}",
            f"/en/products/{rng.randint(100000,999999)}/{rng.choice(['headphones','laptop','camera','watch','speaker'])}",
        ])
        return f"https://{sub}{base}{p}"
    if kind < 0.56:
        base = rng.choice(["bbc.com","cnn.com","reuters.com","nytimes.com","theguardian.com","aljazeera.com","bloomberg.com","forbes.com","techcrunch.com","wired.com","apnews.com","npr.org"])
        slug = "-".join(rng.choices(TOPICS, k=rng.randint(4, 7)))
        return f"https://www.{base}/{rng.choice(['news','articles','blog','guide'])}/{slug}-{rng.randint(10000,99999)}"
    if kind < 0.78:
        # Unknown-looking business domains WITH subdomains + queries (so the model learns these are normal)
        sep = "-" if rng.random() < 0.2 else ""
        domain = f"{rng.choice(WORD1)}{sep}{rng.choice(WORD2)}.{rng.choice(['com','org','net','io','co','app'])}"
        sub = rng.choice(LEGIT_SUBS)
        path = rng.choice(["", "/", "/about", "/services", "/contact", "/pricing", "/blog", "/products", "/team", "/explore", "/dashboard", "/article/" + str(rng.randint(100, 99999))])
        return f"https://{sub}{domain}{path}{rng.choice(LEGIT_QUERIES)}"
    if kind < 0.90:
        base = rng.choice(["paypal.com","chase.com","bankofamerica.com","google.com","microsoft.com","apple.com","amazon.com","netflix.com","sideshift.app"])
        sub = rng.choice(["", "www.", "accounts.", "login.", "secure.", "app."])
        path = rng.choice(["/login", "/signin", "/account/settings", "/accounts/login/", "/profile", "/dashboard", "/billing", "/explore"])
        return f"https://{sub}{base}{path}{rng.choice(LEGIT_QUERIES)}"
    domain = rng.choice(["stanford.edu","mit.edu","harvard.edu","oxford.ac.uk","gov.uk","usa.gov","nih.gov"])
    sub = rng.choice(["", "www.", "students.", "courses."])
    path = rng.choice(["/courses/intro-to-cs/module-1","/students/resources/library-guide","/about/departments","/news/research-update","/forms/apply"])
    return f"https://{sub}{domain}{path}"


def make_phish(rng):
    brand = rng.choice(BRANDS)
    fake = misspell(brand, rng) if rng.random() < 0.7 else brand
    tld = rng.choice(list(RISKY_TLD))
    scheme = rng.choice(["http", "http", "https"])
    k = rng.randrange(12)
    if k == 0: return f"{scheme}://{fake}-login.{tld}/login"
    if k == 1: return f"https://secure.{fake}.{tld}/account/update?verify=1"
    if k == 2: return f"http://{rng.randint(11,220)}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}/bank/login"
    if k == 3: return f"https://www.{fake}.{tld}/signin?password=reset&token={rng.randint(1000,99999)}"
    if k == 4: return f"https://{brand}.login-verify.{fake}.{tld}/confirm/account"
    if k == 5: return f"https://{brand}.com@{fake}.{tld}/login"
    if k == 6: return f"https://xn--{fake}80a.{tld}/secure/verify"
    if k == 7: return f"https://{fake}-secure.com/login/verify"
    if k == 8: return f"https://{digit_fake(brand)}.{rng.choice(['com','net','io'])}/login"
    if k == 9: return f"https://{brand}-{rng.choice(['verify','secure','support','help'])}.com/account/update"
    if k == 10: return f"https://login.{''.join(rng.choices(SYLLABLES, k=5))}.{tld}/secure/verify?token={rng.randint(10000,99999)}"
    return f"https://{fake}.{tld}/verify/{''.join(rng.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=12))}?session={rng.randint(100000,999999)}&redirect=home"


def load_real_data():
    rng = random.Random(42)
    phish_urls = []
    phish_source = ""
    try:
        r = requests.get(OPENPHISH_URL, headers=UA, timeout=15)
        if r.status_code == 200:
            seen = set()
            for line in r.text.splitlines():
                u = line.strip()
                if u.startswith("http") and is_valid_http_url(u):
                    reg = registered_from_host((safe_urlparse(u).hostname or "").lower())
                    if reg in seen:
                        continue
                    seen.add(reg)
                    phish_urls.append(u)
                if len(phish_urls) >= 1500:
                    break
            if phish_urls:
                phish_source = f"OpenPhish live feed ({len(phish_urls):,} real URLs)"
    except Exception:
        pass
    n_target = 4000
    if len(phish_urls) < n_target:
        for _ in range(n_target - len(phish_urls)):
            phish_urls.append(make_phish(rng))
        phish_source = (phish_source + f" + {n_target - len(phish_urls):,} pattern-based") if phish_source else f"Pattern-based ({n_target:,}) - OpenPhish unavailable"

    legit_urls = []
    legit_domains = sorted(KNOWN_LEGIT_DOMAINS) + [f"{rng.choice(WORD1)}-{rng.choice(WORD2)}.{rng.choice(['com','org','net','io','co','app'])}" for _ in range(150)]
    rng.shuffle(legit_domains)
    for i in range(n_target):
        d = legit_domains[i % len(legit_domains)]
        legit_urls.append(make_legit(rng) if rng.random() < 0.6 else f"https://{rng.choice(LEGIT_SUBS)}{d}{rng.choice(['','/','/about','/docs','/help','/blog','/products','/account','/login','/settings','/explore'])}{rng.choice(LEGIT_QUERIES)}")
    legit_source = f"Real top domains ({len(KNOWN_LEGIT_DOMAINS):,}) + realistic subdomains/paths/queries"
    return phish_urls, legit_urls, {"phish": phish_source, "legit": legit_source}


@st.cache_resource(show_spinner="Loading data and training models... (first load ~15 seconds)")
def train_models():
    phish_urls, legit_urls, source = load_real_data()
    urls = legit_urls + phish_urls
    labels = [0] * len(legit_urls) + [1] * len(phish_urls)
    df = pd.DataFrame({"url": urls, "label": labels})
    df["group"] = df["url"].apply(lambda u: registered_from_host((safe_urlparse(u).hostname or "") if safe_urlparse(u) else ""))

    groups = list(df["group"].unique())
    random.Random(123).shuffle(groups)
    test_groups = set(groups[:max(1, int(len(groups) * 0.25))])
    test = df["group"].isin(test_groups)
    train = ~test

    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=18000, min_df=3, sublinear_tf=True)
    char_m = LogisticRegression(max_iter=1500, C=0.5, class_weight="balanced")
    char_m.fit(vec.fit_transform(df.loc[train, "url"]), df.loc[train, "label"])

    x_all = pd.DataFrame([features(u) for u in df["url"]])
    cols = list(x_all.columns)
    struct_m = RandomForestClassifier(n_estimators=300, max_depth=22, min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight="balanced")
    struct_m.fit(x_all[train], df.loc[train, "label"])

    y = df.loc[test, "label"].values
    pc = char_m.predict_proba(vec.transform(df.loc[test, "url"]))[:, 1]
    ps = struct_m.predict_proba(x_all[test])[:, 1]
    ens = 0.5 * pc + 0.5 * ps

    thresholds = []
    for t in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        pred = (ens >= t).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        thresholds.append({"Threshold": t, "Precision": round(float(pr), 3), "Recall": round(float(rc), 3),
                           "F1": round(float(f1), 3), "False Positive Rate": round(fp / max(fp + tn, 1), 3), "False Negatives": int(fn)})

    test_df = df.loc[test].copy()
    test_df["score"] = ens
    fp_cases = test_df[(y == 0) & (ens >= 0.5)].nlargest(4, "score")[["url", "score"]].values.tolist()
    fn_cases = test_df[(y == 1) & (ens < 0.5)].nsmallest(4, "score")[["url", "score"]].values.tolist()

    train_df = df.loc[train]
    def top_tlds(series):
        return (series.apply(lambda u: (safe_urlparse(u).hostname or "").lower() if safe_urlparse(u) else "")
                .apply(lambda h: h.rsplit(".", 1)[-1] if "." in h else h).value_counts().head(5).to_dict())

    metrics = {"train": int(train.sum()), "test": int(test.sum()),
               "unique_test_domains": int(df.loc[test, "group"].nunique()), "models": {}, "source": source}
    for name, prob in [("Character Model", pc), ("Structural Model", ps), ("Ensemble (50/50)", ens)]:
        pred = (prob >= 0.5).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        metrics["models"][name] = {"Precision": round(float(pr), 3), "Recall": round(float(rc), 3),
                                   "F1": round(float(f1), 3), "Accuracy": round(float(accuracy_score(y, pred)), 3)}
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


def flags(url):
    """Return (hard, medium) boolean flags based on concrete phishing evidence."""
    p = safe_urlparse(url) or urlparse("")
    h = (p.hostname or "").lower()
    f = features(url)
    _, is_near, _ = brand_lookalike_score(main_label(h))
    hard = (f["has_ip"] or f["has_at"] or f["has_punycode"] or is_near)
    hard = hard or (f["risky_tld"] and (f["host_keywords"] > 0 or f["near_brand_domain"]))
    medium = (f["risky_tld"] or f["host_keywords"] > 0 or f["subdomain_count"] >= 2
              or f["hyphens_in_host"] >= 3 or f["phish_query_params"] >= 2
              or f["digits_in_host"] >= 4 or f["is_https"] == 0)
    return hard, medium


def decision(url, score):
    """Two-stage decision: concrete indicators set the tier, ML score refines it.
    A URL with NO phishing indicators is never High risk from the model alone."""
    hard, medium = flags(url)
    if hard:
        return "High phishing risk" if score >= 0.70 else "Suspicious — investigate carefully"
    if medium:
        return "High phishing risk" if score >= 0.70 else "Suspicious — investigate carefully"
    return "Lower model-estimated risk"


def indicators(url):
    p = safe_urlparse(url) or urlparse("")
    h = (p.hostname or "").lower()
    f = features(url)
    notes = []
    if f["has_ip"]: notes.append("Uses an IP address instead of a domain name")
    if p.scheme != "https": notes.append("Does not use HTTPS")
    if "@" in url: notes.append("Contains '@' — can hide the real destination")
    if "xn--" in url.lower(): notes.append("Punycode domain — can imitate a known brand")
    if f["subdomain_count"] >= 2: notes.append("Has multiple subdomains")
    if f["hyphens_in_host"] >= 3: notes.append("Multiple hyphens in hostname")
    if f["digits_in_host"] >= 4: notes.append("Unusual number of digits in hostname")
    if f["host_keywords"] > 0: notes.append("Suspicious word (e.g. login/verify/secure) inside the domain name")
    elif f["path_keywords"] > 0: notes.append("Contains account/login keyword in path or query")
    if f["risky_tld"]: notes.append("Uses a TLD frequently abused in phishing campaigns")
    if f["phish_query_params"] >= 2: notes.append("Query contains multiple credential/token parameters")
    if f["url_entropy"] > 4.9: notes.append("High character randomness (entropy) in URL")
    if f["is_dev_hosting"]: notes.append("Hosted on a known free dev platform — less commonly used for high-confidence phishing")
    if f["is_known_legit_domain"]: notes.append("Registered domain is a known major/legitimate site")
    label = main_label(h)
    sub = subdomain_from_host(h)
    sim, is_near, matched = brand_lookalike_score(label)
    if is_near and matched: notes.append(f"Lookalike domain: '{label}' closely resembles '{matched}'")
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


def set_url(u):
    st.session_state["url_input"] = u


if "url_input" not in st.session_state:
    st.session_state["url_input"] = "https://google.com"

st.markdown("""
<style>
.stApp { background: linear-gradient(160deg, #0a0f1e 0%, #111827 50%, #0f172a 100%); }
[data-testid="stHeader"] { background: rgba(10, 15, 30, 0.85); backdrop-filter: blur(12px); }
section[data-testid="stSidebar"] { background: #0c1322; border-right: 1px solid #1e293b; }
section[data-testid="stSidebar"] * { color: #cbd5e1; }
.scan-hero { padding: 2.4rem 2.4rem 2rem; margin-bottom: 1.2rem; border: 1px solid #1e3a5f; border-radius: 20px;
    background: linear-gradient(135deg, #0c2340 0%, #0f172a 60%, #1a1040 100%); position: relative; overflow: hidden; }
.scan-hero::before { content: ""; position: absolute; top: -50%; right: -20%; width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%); border-radius: 50%; }
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
st.caption(f"engine {ENGINE} · Phishing: {src['phish']} · Legitimate: {src['legit']} · Train: {metrics['train']:,} · Test: {metrics['test']:,} ({metrics['unique_test_domains']:,} unseen domains)")

with st.sidebar:
    st.header("How to read a result")
    st.markdown("- **High risk:** concrete phishing evidence present.\n- **Suspicious:** some warning signs, or a lookalike brand.\n- **Lower risk:** no phishing indicators detected.")
    st.divider()
    st.caption("A URL with no phishing indicators is never High risk from the model alone. Scores are estimates, not probabilities.")

tab1, tab2, tab3, tab4 = st.tabs(["Scan a URL", "Batch Check", "Model Performance", "How It Works"])

with tab1:
    with st.container(border=True):
        st.subheader("Check a link")
        examples = [("Regular link", "https://arena.ai"), ("Lookalike link", "https://paypa1.com/"), ("IP address link", "http://192.0.2.25/bank/login")]
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
            st.session_state["result"] = {"url": candidate, "char": pc, "struct": ps, "combined": combined, "notes": indicators(candidate), "info": domain_info(candidate)}
    r = st.session_state.get("result")
    if r and r["url"] == url.strip():
        verdict = decision(r["url"], r["combined"])
        st.subheader("Assessment")
        if verdict.startswith("High"): st.error(verdict)
        elif verdict.startswith("Suspicious"): st.warning(verdict)
        else: st.success(verdict)
        st.code(r["url"], language=None, wrap_lines=True)
        a, b, c = st.columns(3)
        a.metric("Character model", f"{r['char']:.3f}")
        b.metric("Structural model", f"{r['struct']:.3f}")
        c.metric("Combined score", f"{r['combined']:.3f}")
        st.progress(min(max(r["combined"], 0.0), 1.0))
        st.caption("Combined = 0.5 × Character + 0.5 × Structural. The final label combines this score with concrete phishing indicators.")
        with st.expander("Rule-based indicators", expanded=True):
            for note in r["notes"]: st.text(f"• {note}")
            if not r["notes"]: st.write("No rule-based indicators triggered.")
        with st.expander("Domain details"): st.json(r["info"])
        report = (f"LinkGuard Report ({ENGINE})\n\nURL: {r['url']}\nAssessment: {verdict}\nCharacter: {r['char']:.3f}\nStructural: {r['struct']:.3f}\nCombined: {r['combined']:.3f}\n\nIndicators:\n"
                  + "\n".join("- " + n for n in (r["notes"] or ["None"])))
        st.download_button("Download report", report, "linkguard_report.txt", "text/plain")

with tab2:
    text = st.text_area("Paste URLs, one per line:", "https://google.com\nhttps://arena.ai\nhttps://paypa1.com/\nhttps://gooogle.xyz/login?verify=1", height=140)
    if st.button("Run Batch Check", type="primary"):
        rows = []
        for u in [x.strip() for x in text.splitlines() if x.strip()]:
            if is_valid_http_url(u):
                pc, ps, cb = analyze(u)
                rows.append({"URL": u, "Character": round(pc, 3), "Structural": round(ps, 3), "Combined": round(cb, 3), "Risk": decision(u, cb)})
            else:
                rows.append({"URL": u, "Character": "-", "Structural": "-", "Combined": "-", "Risk": "Invalid"})
        out = pd.DataFrame(rows)
        st.dataframe(out, use_container_width=True)
        st.download_button("Download CSV", out.to_csv(index=False), "results.csv", "text/csv")

with tab3:
    st.subheader("Performance on Unseen Domains")
    st.dataframe(pd.DataFrame(metrics["models"]).T, use_container_width=True)
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
    bc1.write("**Phishing TLDs (top 5):**"); bc1.json(metrics["phish_tlds"])
    bc2.write("**Legitimate TLDs (top 5):**"); bc2.json(metrics["legit_tlds"])
    st.divider()
    st.subheader("Failure Cases from Unseen Test Domains")
    st.write("**False Positives** (legitimate URLs wrongly flagged):")
    for url_case, score in (metrics["fp_cases"] or []): st.text(f"  Score: {score:.3f}  |  {url_case[:90]}")
    if not metrics["fp_cases"]: st.text("  None at 0.50 threshold.")
    st.write("**False Negatives** (phishing URLs missed):")
    for url_case, score in (metrics["fn_cases"] or []): st.text(f"  Score: {score:.3f}  |  {url_case[:90]}")
    if not metrics["fn_cases"]: st.text("  None at 0.50 threshold.")
    st.divider()
    st.subheader("Feature Importance (Structural Model)")
    imp = pd.DataFrame({"Feature": cols, "Importance": struct_m.feature_importances_}).sort_values("Importance")
    st.bar_chart(imp.set_index("Feature"))

with tab4:
    st.code("URL -> Validate -> Extract features + Character n-grams -> 2 Models -> 50/50 Score -> Indicator-based decision policy -> Risk + Indicators")
    st.markdown(f"""
**Decision policy (engine {ENGINE}):**
The ML ensemble produces a 0–1 risk score. A rule-based policy then maps that score plus concrete phishing indicators to the final label:
- **High risk** — concrete evidence present (IP address, lookalike brand, `@`, punycode, risky TLD with credential keywords) and a high model score.
- **Suspicious** — some warning signs (risky TLD, keywords in domain, many subdomains, non-HTTPS), or a lookalike brand.
- **Lower risk** — no phishing indicators detected. A clean URL is never High risk from the model alone.

**Why this design:** ML-only URL classifiers over-flag unseen but clean domains, because they can only recognize patterns they were trained on. Capping the label by concrete indicators prevents false positives on legitimate sites (developer deployments, real businesses, crypto platforms) while still catching phishing that shows real signals. This mirrors production tools that combine ML scores with reputation and rule-based checks.

**Models used:**
- **Character Model:** TF-IDF (3–5 character n-grams) + Logistic Regression. Learns misspellings and suspicious character sequences.
- **Structural Model:** Random Forest (300 trees) on 40+ numerical features (lengths, dots, hyphens, digits, entropy, HTTPS, IP, subdomain depth, keyword location, query/path analysis, brand similarity, dev-hosting, known domains).
- **Ensemble:** 50/50 average.

**Thresholds:** 0.70 separates High from Suspicious within a flag tier; chosen from the Threshold Analysis table to keep precision high for the High label.

**Domain-separated testing:** URLs are grouped by registered domain before a 75/25 train/test split, so the model is evaluated on domains it has never seen.

**Known limitation:** a phishing URL with no indicators at all (a clean `.com` domain, HTTPS, no keywords, no lookalike) can be missed. URL-only analysis cannot inspect page content, certificates, or domain age.

**Data sources:**
- Phishing: OpenPhish live verified feed (openphish.com) + realistic pattern-based examples.
- Legitimate: real top popular domains with realistic subdomain, path, and query variations.
""")

st.caption("Scores are model estimates. A low score does not guarantee a site is safe.")
