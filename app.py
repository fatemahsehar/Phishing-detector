import base64
import difflib
import ipaddress
import math
import tempfile
import zipfile
from collections import Counter
from urllib.parse import parse_qsl, urlsplit

import numpy as np
import pandas as pd
import requests
import streamlit as st
import tldextract
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline


st.set_page_config(page_title="LinkGuard", page_icon="🛡️", layout="wide")

# UCI PhiUSIIL Phishing URL Dataset. Its labels are:
# 0 = phishing, 1 = legitimate.
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/967/"
    "phiusiil+phishing+url+dataset.zip"
)
SAMPLE_PER_LABEL = 5000
DOMAIN_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())
KEYWORDS = ("login", "signin", "verify", "secure", "update", "password", "account")
BRANDS = ("google", "microsoft", "apple", "amazon", "paypal", "netflix")


def parse_url(url):
    """Parse a URL without visiting its website."""
    if not url or len(url) > 4096 or any(c.isspace() or ord(c) < 32 for c in url):
        return None
    try:
        parts = urlsplit(url)
        if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
            return None
        _ = parts.port  # Reject malformed ports.
        return parts
    except ValueError:
        return None


def registered_domain(host):
    host = host.lower().rstrip(".")
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass

    extracted = DOMAIN_EXTRACTOR(host)
    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}"
    return host


def structural_features(url):
    parts = parse_url(url)
    if parts is None:
        raise ValueError("Invalid HTTP/HTTPS URL.")

    host = parts.hostname.lower()
    extracted = DOMAIN_EXTRACTOR(host)
    text = url.lower()

    try:
        ipaddress.ip_address(host)
        is_ip = 1
    except ValueError:
        is_ip = 0

    frequencies = Counter(url)
    entropy = -sum(
        (count / len(url)) * math.log2(count / len(url))
        for count in frequencies.values()
    )

    return {
        "url_length": len(url),
        "hostname_length": len(host),
        "path_length": len(parts.path),
        "query_length": len(parts.query),
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "digit_count": sum(c.isdigit() for c in url),
        "percent_count": url.count("%"),
        "subdomain_count": (
            len(extracted.subdomain.split(".")) if extracted.subdomain else 0
        ),
        "query_parameter_count": len(parse_qsl(parts.query, keep_blank_values=True)),
        "path_depth": sum(bool(piece) for piece in parts.path.split("/")),
        "keyword_count": sum(word in text for word in KEYWORDS),
        "has_at_symbol": int("@" in parts.netloc),
        "is_https": int(parts.scheme.lower() == "https"),
        "is_ip": is_ip,
        "has_punycode": int("xn--" in host),
        "unusual_port": int(parts.port is not None and parts.port not in (80, 443)),
        "entropy": entropy,
    }


def warning_indicators(url):
    parts = parse_url(url)
    host = parts.hostname.lower()
    extracted = DOMAIN_EXTRACTOR(host)
    warnings = []

    if parts.scheme.lower() == "http":
        warnings.append("The URL does not use HTTPS.")
    if "@" in parts.netloc:
        warnings.append("The address contains an @ symbol.")
    if "xn--" in host:
        warnings.append("The hostname contains punycode.")
    if len(url) > 120:
        warnings.append("The URL is unusually long.")
    if any(word in url.lower() for word in KEYWORDS):
        warnings.append("The URL contains a login or verification-related word.")

    try:
        ipaddress.ip_address(host)
        warnings.append("The hostname is an IP address.")
    except ValueError:
        pass

    for brand in BRANDS:
        name = extracted.domain.lower()
        if name and name != brand:
            similarity = difflib.SequenceMatcher(None, name, brand).ratio()
            if similarity >= 0.80:
                warnings.append(
                    f"The domain name '{name}' resembles the brand '{brand}'."
                )
                break

    if extracted.subdomain:
        tokens = extracted.subdomain.lower().split(".")
        if any(token in BRANDS for token in tokens):
            warnings.append("A brand name appears in the subdomain.")

    return warnings


def load_dataset_sample():
    """Download UCI data and keep a small random sample of each class."""
    rng = np.random.default_rng(42)
    samples = {0: None, 1: None}

    with requests.get(DATA_URL, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()

        with tempfile.TemporaryFile() as downloaded:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    downloaded.write(block)
            downloaded.seek(0)

            with zipfile.ZipFile(downloaded) as archive:
                csv_files = [
                    name for name in archive.namelist()
                    if name.lower().endswith(".csv")
                ]
                if not csv_files:
                    raise ValueError("The UCI download did not contain a CSV file.")

                csv_name = max(
                    csv_files, key=lambda name: archive.getinfo(name).file_size
                )
                with archive.open(csv_name) as csv_file:
                    batches = pd.read_csv(
                        csv_file,
                        usecols=lambda column: column.strip().lower()
                        in {"url", "label"},
                        chunksize=10000,
                        on_bad_lines="skip",
                    )

                    for batch in batches:
                        batch.columns = [col.strip().lower() for col in batch.columns]
                        if not {"url", "label"}.issubset(batch.columns):
                            raise ValueError("Expected URL and label columns.")

                        batch["label"] = pd.to_numeric(
                            batch["label"], errors="coerce"
                        )
                        for label in (0, 1):
                            candidates = batch.loc[
                                batch["label"].eq(label), ["url", "label"]
                            ].dropna(subset=["url"]).copy()

                            if candidates.empty:
                                continue

                            candidates["_random"] = rng.random(len(candidates))
                            previous = samples[label]
                            pool = (
                                candidates
                                if previous is None
                                else pd.concat([previous, candidates], ignore_index=True)
                            )
                            samples[label] = pool.nsmallest(
                                SAMPLE_PER_LABEL, "_random"
                            )

    if any(sample is None for sample in samples.values()):
        raise ValueError("Both phishing and legitimate examples are required.")

    data = pd.concat(
        [samples[0], samples[1]], ignore_index=True
    ).drop(columns=["_random"])

    data["url"] = data["url"].astype(str).str.strip()
    data = data.loc[data["url"].map(lambda url: parse_url(url) is not None)].copy()
    data["label"] = data["label"].astype(int)

    # Remove duplicates and URLs with conflicting labels.
    data = data.drop_duplicates(subset=["url", "label"])
    data = data.loc[
        data.groupby("url")["label"].transform("nunique").eq(1)
    ].drop_duplicates(subset="url").reset_index(drop=True)

    if set(data["label"].unique()) != {0, 1}:
        raise ValueError("The sample needs both classes.")

    # In our models, 1 means phishing.
    data["is_phishing"] = data["label"].eq(0).astype(int)
    data["domain_group"] = data["url"].map(
        lambda url: registered_domain(parse_url(url).hostname)
    )
    return data


def evaluation_row(name, actual, scores):
    predicted = np.asarray(scores) >= 0.50
    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    return {
        "Model": name,
        "Precision": round(precision_score(actual, predicted, zero_division=0), 3),
        "Recall": round(recall_score(actual, predicted, zero_division=0), 3),
        "F1": round(f1_score(actual, predicted, zero_division=0), 3),
        "False-positive rate": round(fp / max(fp + tn, 1), 3),
        "False positives": int(fp),
        "False negatives": int(fn),
    }


@st.cache_resource(show_spinner=False)
def train_models():
    """Train once per running server; reuse models for later scans."""
    data = load_dataset_sample()
    numeric = pd.DataFrame([structural_features(url) for url in data["url"]])
    labels = data["is_phishing"].to_numpy()

    splitter = GroupShuffleSplit(
        n_splits=20, test_size=0.25, random_state=42
    )
    for train_indices, test_indices in splitter.split(
        data, labels, groups=data["domain_group"]
    ):
        if (
            len(set(labels[train_indices])) == 2
            and len(set(labels[test_indices])) == 2
        ):
            break
    else:
        raise ValueError("Could not make a test split containing both classes.")

    character_model = make_pipeline(
        TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5),
            max_features=7000, min_df=2
        ),
        LogisticRegression(max_iter=800, class_weight="balanced"),
    )
    character_model.fit(
        data.iloc[train_indices]["url"], labels[train_indices]
    )

    structural_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    structural_model.fit(
        numeric.iloc[train_indices], labels[train_indices]
    )

    actual = labels[test_indices]
    character_scores = character_model.predict_proba(
        data.iloc[test_indices]["url"]
    )[:, 1]
    structural_scores = structural_model.predict_proba(
        numeric.iloc[test_indices]
    )[:, 1]
    combined_scores = (character_scores + structural_scores) / 2

    evaluation = pd.DataFrame([
        evaluation_row("Character model", actual, character_scores),
        evaluation_row("Structural model", actual, structural_scores),
        evaluation_row("50/50 ensemble", actual, combined_scores),
    ])

    return {
        "character": character_model,
        "structural": structural_model,
        "evaluation": evaluation,
        "training_count": len(train_indices),
        "test_count": len(test_indices),
    }


# -------------------- Website --------------------

st.caption("🛡️ LINKGUARD · PHISHING URL RESEARCH PROTOTYPE")
st.title("Inspect the link. Before the click.")
st.write(
    "Two trained models inspect URL text and structure. "
    "The app does **not** open the website you enter."
)

with st.form("url_form"):
    entered_url = st.text_input(
        "Paste a full URL",
        placeholder="https://example.org/account",
        max_chars=4096,
    )
    analyze_clicked = st.form_submit_button("Analyze URL", type="primary")

st.caption(
    "The first analysis downloads public labelled data and trains the models. "
    "It can take several minutes; later analyses reuse them until the server restarts."
)

if analyze_clicked:
    st.session_state.pop("last_report", None)
    url = entered_url.strip()

    if parse_url(url) is None:
        st.error("Enter a full HTTP or HTTPS URL, such as https://example.org.")
    else:
        try:
            with st.spinner("Preparing the dataset and analyzing the URL..."):
                trained = train_models()

                character_score = float(
                    trained["character"].predict_proba([url])[0, 1]
                )
                numeric_row = pd.DataFrame([structural_features(url)])
                structural_score = float(
                    trained["structural"].predict_proba(numeric_row)[0, 1]
                )

            st.session_state["last_report"] = {
                "url": url,
                "hostname": parse_url(url).hostname,
                "character_score": character_score,
                "structural_score": structural_score,
                "combined_score": (character_score + structural_score) / 2,
                "indicators": warning_indicators(url),
                "evaluation": trained["evaluation"],
                "training_count": trained["training_count"],
                "test_count": trained["test_count"],
            }
        except Exception as error:
            st.error(f"Training or analysis could not finish: {error}")

report = st.session_state.get("last_report")

if report:
    st.divider()
    st.subheader("Analysis report")
    st.code(report["url"], language=None)
    st.write("**Hostname:**", report["hostname"])
    st.write("**Registered domain:**", registered_domain(report["hostname"]))

    left, middle, right = st.columns(3)
    left.metric("Character model score", f"{report['character_score']:.3f}")
    middle.metric("Structural model score", f"{report['structural_score']:.3f}")
    right.metric("Combined score", f"{report['combined_score']:.3f}")

    score = report["combined_score"]
    st.progress(score)

    if score >= 0.70:
        st.error("High model-estimated phishing risk")
    elif score >= 0.40:
        st.warning("Suspicious — investigate further")
    else:
        st.info("Lower model-estimated risk — **not a guarantee of safety**")

    st.caption(
        "Scores range from 0 to 1. They are not calibrated probabilities; "
        "the 0.40 and 0.70 risk cut-offs are provisional."
    )

    st.subheader("URL warning signs")
    if report["indicators"]:
        for item in report["indicators"]:
            st.write("•", item)
    else:
        st.write("No simple warning signs flagged. This does not prove safety.")
    st.caption(
        "These indicators are independent rules, not explanations of "
        "the models' individual decisions."
    )

    with st.expander("See measured model performance"):
        st.write(
            f"Training URLs: **{report['training_count']:,}** · "
            f"Test URLs: **{report['test_count']:,}**"
        )
        st.dataframe(report["evaluation"], hide_index=True)
        st.caption(
            "Test domains were kept separate from training domains. "
            "Metrics use a 0.50 classification threshold on a sampled, "
            "roughly balanced research dataset—not real-world traffic."
        )
        st.markdown(
            "[Dataset source: UCI PhiUSIIL Phishing URL Dataset]"
            "(https://archive.ics.uci.edu/dataset/967/"
            "phiusiil+phishing+url+dataset)"
        )

    with st.expander("Optional VirusTotal lookup"):
        st.warning(
            "Only use this for public URLs. Clicking the button sends the "
            "full URL to VirusTotal; do not submit private links or access tokens."
        )
        api_key = st.text_input(
            "VirusTotal API key", type="password", key="vt_api_key"
        )

        if st.button("Check existing VirusTotal report"):
            if not api_key:
                st.info("Enter your API key first.")
            else:
                try:
                    url_id = base64.urlsafe_b64encode(
                        report["url"].encode()
                    ).decode().rstrip("=")
                    response = requests.get(
                        f"https://www.virustotal.com/api/v3/urls/{url_id}",
                        headers={"x-apikey": api_key},
                        timeout=15,
                    )

                    if response.status_code == 404:
                        st.info("VirusTotal has no existing report for this URL.")
                    elif response.status_code == 200:
                        stats = (
                            response.json()
                            .get("data", {})
                            .get("attributes", {})
                            .get("last_analysis_stats", {})
                        )
                        st.json(stats or {"message": "No analysis statistics found."})
                    else:
                        st.warning(
                            f"VirusTotal returned HTTP {response.status_code}."
                        )
                except requests.RequestException as error:
                    st.warning(f"VirusTotal lookup failed: {error}")

        st.caption("VirusTotal findings do not change the two-model score.")
