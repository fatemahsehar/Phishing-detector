import difflib
import ipaddress
import math
import tempfile
import zipfile

from collections import Counter
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import requests
import streamlit as st
import tldextract

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline


st.set_page_config(
    page_title="LinkGuard",
    page_icon="🛡️",
    layout="wide",
)

# UCI PhiUSIIL labels:
# 0 = phishing, 1 = legitimate.
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/967/"
    "phiusiil+phishing+url+dataset.zip"
)

# Maximum number sampled from each original dataset class.
# Increase only after checking that your hosting has enough memory.
SAMPLE_PER_CLASS = 4000
RANDOM_SEED = 42
MAX_URL_LENGTH = 4096

BRANDS = [
    "google",
    "microsoft",
    "apple",
    "amazon",
    "paypal",
    "netflix",
    "facebook",
    "instagram",
    "youtube",
    "linkedin",
]

KEYWORDS = [
    "login",
    "signin",
    "verify",
    "secure",
    "account",
    "password",
    "update",
    "confirm",
    "billing",
]

# Use the bundled public suffix list, without a network download.
# Private suffixes help distinguish separate hosted-app domains.
extract_domain = tldextract.TLDExtract(
    suffix_list_urls=(),
    include_psl_private_domains=True,
)


def parse_url(value):
    """Parse a URL without visiting the destination."""
    if not isinstance(value, str):
        return None

    if not value or len(value) > MAX_URL_LENGTH:
        return None

    if any(character.isspace() or ord(character) < 32 for character in value):
        return None

    try:
        parsed = urlsplit(value)

        if parsed.scheme.lower() not in ("http", "https"):
            return None

        if not parsed.hostname:
            return None

        # Accessing port checks for malformed or out-of-range values.
        _ = parsed.port

        return parsed

    except ValueError:
        return None


def is_ip(host):
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def domain_parts(host):
    """Return the domain group and some display information."""
    host = host.lower().rstrip(".")

    if is_ip(host):
        return {
            "group": host,
            "subdomain": "",
            "domain": host,
            "suffix": "",
        }

    result = extract_domain(host)

    if result.domain and result.suffix:
        group = f"{result.domain}.{result.suffix}"
    else:
        # Use the whole hostname when no known suffix is found.
        group = host

    return {
        "group": group,
        "subdomain": result.subdomain,
        "domain": result.domain,
        "suffix": result.suffix,
    }


def structural_features(url):
    parsed = parse_url(url)
    if parsed is None:
        raise ValueError("Invalid URL")

    host = parsed.hostname.lower()
    parts = domain_parts(host)
    path = parsed.path or ""
    query = parsed.query or ""

    counts = Counter(url)
    entropy = -sum(
        (count / len(url)) * math.log2(count / len(url))
        for count in counts.values()
    )

    domain_label = parts["domain"].lower()
    lookalike = any(
        domain_label
        and domain_label != brand
        and difflib.SequenceMatcher(
            None, domain_label, brand
        ).ratio() >= 0.80
        for brand in BRANDS
    )

    return {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "dots": url.count("."),
        "hyphens": url.count("-"),
        "digits": sum(character.isdigit() for character in url),
        "percent_signs": url.count("%"),
        "path_segments": len([item for item in path.split("/") if item]),
        "query_parameters": len(query.split("&")) if query else 0,
        "subdomain_count": (
            len(parts["subdomain"].split("."))
            if parts["subdomain"]
            else 0
        ),
        "uses_ip": int(is_ip(host)),
        "uses_https": int(parsed.scheme.lower() == "https"),
        "has_userinfo": int("@" in parsed.netloc),
        "has_punycode": int("xn--" in host),
        "nonstandard_port": int(parsed.port not in (None, 80, 443)),
        "host_keywords": sum(word in host for word in KEYWORDS),
        "path_keywords": sum(
            word in (path + "?" + query).lower()
            for word in KEYWORDS
        ),
        "possible_brand_lookalike": int(lookalike),
        "entropy": entropy,
    }


def indicators(url):
    parsed = parse_url(url)
    host = parsed.hostname.lower()
    parts = domain_parts(host)
    notes = []

    if is_ip(host):
        notes.append("The hostname is an IP address.")

    if parsed.scheme.lower() == "http":
        notes.append("The URL does not use HTTPS.")

    if "@" in parsed.netloc:
        notes.append(
            "The address contains user information before the hostname."
        )

    if "xn--" in host:
        notes.append(
            "The hostname contains punycode; check the displayed domain carefully."
        )

    label = parts["domain"].lower()
    for brand in BRANDS:
        if (
            label
            and label != brand
            and difflib.SequenceMatcher(None, label, brand).ratio() >= 0.80
        ):
            notes.append(
                f"The domain name '{label}' resembles '{brand}'. "
                "This is a similarity warning, not proof of impersonation."
            )
            break

    return notes


def load_dataset():
    """
    Download UCI PhiUSIIL and take a reproducible random sample.
    No generated URLs or fallback data are added.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    samples = {0: None, 1: None}

    with requests.get(
        DATASET_URL,
        stream=True,
        timeout=(20, 120),
    ) as response:
        response.raise_for_status()

        # Store the ZIP temporarily instead of keeping the whole file in RAM.
        with tempfile.TemporaryFile() as downloaded:
            total_bytes = 0

            for block in response.iter_content(chunk_size=1024 * 1024):
                if not block:
                    continue

                total_bytes += len(block)

                if total_bytes > 200 * 1024 * 1024:
                    raise ValueError("Dataset download is larger than expected.")

                downloaded.write(block)

            downloaded.seek(0)

            with zipfile.ZipFile(downloaded) as archive:
                csv_files = [
                    name for name in archive.namelist()
                    if name.lower().endswith(".csv")
                ]

                if not csv_files:
                    raise ValueError("No CSV file found in the dataset archive.")

                csv_name = max(
                    csv_files,
                    key=lambda name: archive.getinfo(name).file_size,
                )

                with archive.open(csv_name) as csv_file:
                    chunks = pd.read_csv(
                        csv_file,
                        usecols=lambda name: name.strip().lower()
                        in {"url", "label"},
                        chunksize=10000,
                        on_bad_lines="skip",
                    )

                    for chunk in chunks:
                        chunk.columns = [
                            column.strip().lower()
                            for column in chunk.columns
                        ]

                        if not {"url", "label"}.issubset(chunk.columns):
                            raise ValueError(
                                "The dataset must contain URL and label columns."
                            )

                        chunk = chunk.dropna(subset=["url", "label"]).copy()
                        chunk["url"] = chunk["url"].astype(str).str.strip()
                        chunk["label"] = pd.to_numeric(
                            chunk["label"],
                            errors="coerce",
                        )

                        chunk = chunk.loc[
                            chunk["url"].map(
                                lambda value: parse_url(value) is not None
                            )
                        ].copy()

                        for original_label in (0, 1):
                            candidates = chunk.loc[
                                chunk["label"].eq(original_label),
                                ["url", "label"],
                            ].copy()

                            if candidates.empty:
                                continue

                            # Random priorities give a reproducible sample
                            # across the whole file, not only the first rows.
                            candidates["_priority"] = rng.random(
                                len(candidates)
                            )

                            previous = samples[original_label]

                            if previous is None:
                                pool = candidates
                            else:
                                pool = pd.concat(
                                    [previous, candidates],
                                    ignore_index=True,
                                )

                            samples[original_label] = pool.nsmallest(
                                SAMPLE_PER_CLASS,
                                "_priority",
                            )

    if samples[0] is None or samples[1] is None:
        raise ValueError(
            "The dataset did not provide usable URLs for both classes."
        )

    data = pd.concat(
        [samples[0], samples[1]],
        ignore_index=True,
    ).drop(columns=["_priority"])

    # Remove duplicate URLs and conflicting labels.
    data = data.drop_duplicates(subset=["url", "label"])

    data = data.loc[
        data.groupby("url")["label"].transform("nunique").eq(1)
    ].copy()

    data = data.drop_duplicates(subset="url").reset_index(drop=True)

    # Convert to our model convention:
    # 1 = phishing, 0 = legitimate.
    data["is_phishing"] = data["label"].eq(0).astype(int)

    data["domain_group"] = data["url"].map(
        lambda value: domain_parts(
            parse_url(value).hostname
        )["group"]
    )

    if set(data["is_phishing"].unique()) != {0, 1}:
        raise ValueError("Both classes are needed for training.")

    return data


def split_by_domain(data):
    """
    Split into train, validation, and test groups.
    Never fall back to splitting individual rows.
    """
    labels = data["is_phishing"].to_numpy()
    groups = data["domain_group"]

    outer = GroupShuffleSplit(
        n_splits=20,
        test_size=0.20,
        random_state=RANDOM_SEED,
    )

    for remaining_idx, test_idx in outer.split(data, labels, groups):
        if len(set(labels[test_idx])) < 2:
            continue

        remaining = data.iloc[remaining_idx]

        inner = GroupShuffleSplit(
            n_splits=20,
            test_size=0.25,
            random_state=RANDOM_SEED + 1,
        )

        for train_local, val_local in inner.split(
            remaining,
            remaining["is_phishing"],
            remaining["domain_group"],
        ):
            train_idx = remaining_idx[train_local]
            val_idx = remaining_idx[val_local]

            if (
                len(set(labels[train_idx])) == 2
                and len(set(labels[val_idx])) == 2
            ):
                return train_idx, val_idx, test_idx

    raise ValueError(
        "Could not create three domain-separated splits containing "
        "both classes. Try a larger sample."
    )


def threshold_row(actual, scores, threshold):
    predicted = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    ).ravel()

    return {
        "Threshold": threshold,
        "Precision": round(
            precision_score(actual, predicted, zero_division=0), 3
        ),
        "Recall": round(
            recall_score(actual, predicted, zero_division=0), 3
        ),
        "F1": round(f1_score(actual, predicted, zero_division=0), 3),
        "False-positive rate": round(fp / (fp + tn), 3),
        "Legitimate URLs": int(fp + tn),
        "False positives": int(fp),
        "Phishing URLs": int(tp + fn),
        "Missed phishing": int(fn),
    }


def model_row(name, actual, scores):
    predicted = (scores >= 0.50).astype(int)

    return {
        "Model": name,
        "Accuracy": round(accuracy_score(actual, predicted), 3),
        "Precision": round(
            precision_score(actual, predicted, zero_division=0), 3
        ),
        "Recall": round(
            recall_score(actual, predicted, zero_division=0), 3
        ),
        "F1": round(f1_score(actual, predicted, zero_division=0), 3),
    }


@st.cache_resource(show_spinner=False)
def train_models():
    data = load_dataset()

    train_idx, val_idx, test_idx = split_by_domain(data)
    labels = data["is_phishing"].to_numpy()

    numeric = pd.DataFrame(
        [structural_features(url) for url in data["url"]]
    )

    character_model = make_pipeline(
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=20000,
            min_df=2,
            sublinear_tf=True,
        ),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
    )

    character_model.fit(
        data.iloc[train_idx]["url"],
        labels[train_idx],
    )

    structural_model = RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )

    structural_model.fit(
        numeric.iloc[train_idx],
        labels[train_idx],
    )

    def scores_for(indices):
        char_scores = character_model.predict_proba(
            data.iloc[indices]["url"]
        )[:, 1]

        structural_scores = structural_model.predict_proba(
            numeric.iloc[indices]
        )[:, 1]

        combined_scores = (char_scores + structural_scores) / 2

        return char_scores, structural_scores, combined_scores

    val_char, val_struct, val_combined = scores_for(val_idx)
    test_char, test_struct, test_combined = scores_for(test_idx)

    val_actual = labels[val_idx]
    test_actual = labels[test_idx]

    validation_table = pd.DataFrame([
        threshold_row(val_actual, val_combined, threshold)
        for threshold in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
    ])

    test_threshold_table = pd.DataFrame([
        threshold_row(test_actual, test_combined, threshold)
        for threshold in (0.40, 0.50, 0.70)
    ])

    model_table = pd.DataFrame([
        model_row("Character model", test_actual, test_char),
        model_row("Structural model", test_actual, test_struct),
        model_row("50/50 ensemble", test_actual, test_combined),
    ])

    matrix = confusion_matrix(
        test_actual,
        (test_combined >= 0.50).astype(int),
        labels=[0, 1],
    )

    # Simple checks for differences between classes in the training set.
    training_data = data.iloc[train_idx]

    bias_rows = []
    for label_value, name in [(0, "Legitimate"), (1, "Phishing")]:
        subset = training_data.loc[
            training_data["is_phishing"].eq(label_value)
        ]

        parsed_urls = subset["url"].map(parse_url)

        bias_rows.append({
            "Class": name,
            "URLs": len(subset),
            "Domains": subset["domain_group"].nunique(),
            "Median URL length": round(
                subset["url"].str.len().median(), 1
            ),
            "URLs with a query": round(
                parsed_urls.map(lambda item: bool(item.query)).mean(),
                3,
            ),
            "HTTPS URLs": round(
                parsed_urls.map(
                    lambda item: item.scheme.lower() == "https"
                ).mean(),
                3,
            ),
        })

    return {
        "character_model": character_model,
        "structural_model": structural_model,
        "feature_columns": list(numeric.columns),
        "train_count": len(train_idx),
        "validation_count": len(val_idx),
        "test_count": len(test_idx),
        "train_domains": data.iloc[train_idx]["domain_group"].nunique(),
        "validation_domains": data.iloc[val_idx]["domain_group"].nunique(),
        "test_domains": data.iloc[test_idx]["domain_group"].nunique(),
        "model_table": model_table,
        "validation_table": validation_table,
        "test_threshold_table": test_threshold_table,
        "matrix": matrix,
        "bias_table": pd.DataFrame(bias_rows),
    }


def analyze_url(url, trained):
    character_score = float(
        trained["character_model"].predict_proba([url])[0, 1]
    )

    numeric = pd.DataFrame([structural_features(url)])
    numeric = numeric[trained["feature_columns"]]

    structural_score = float(
        trained["structural_model"].predict_proba(numeric)[0, 1]
    )

    return {
        "character": character_score,
        "structural": structural_score,
        "combined": (character_score + structural_score) / 2,
        "indicators": indicators(url),
        "domain": domain_parts(parse_url(url).hostname)["group"],
    }


def risk_label(score):
    if score >= 0.70:
        return "High model-estimated risk"
    if score >= 0.40:
        return "Suspicious — investigate"
    return "Lower model-estimated risk"


# -------------------- Interface --------------------

st.markdown(
    """
    <style>
    .hero {
        padding: 2rem;
        border-radius: 18px;
        border: 1px solid #315469;
        background: radial-gradient(
            ellipse at top right, #1e504d, transparent 65%
        ), #102233;
        margin-bottom: 1rem;
    }
    .hero h1 {
        color: #f4faff;
        margin: 0 0 0.5rem;
        padding: 0;
    }
    .hero p {
        color: #cfdee8;
        margin: 0;
    }
    </style>

    <section class="hero">
      <h1>🛡️ LinkGuard</h1>
      <p>Inspect the link before the click.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Analyze URL text without opening the destination website."
)

st.caption(
    "The first analysis downloads the public dataset and trains the models. "
    "It may take several minutes. The models are reused until the server restarts."
)

with st.form("scan_form"):
    submitted_url = st.text_input(
        "Paste a full URL",
        placeholder="https://example.org/",
        max_chars=MAX_URL_LENGTH,
    )
    submitted = st.form_submit_button(
        "Analyze URL",
        type="primary",
    )

if submitted:
    st.session_state.pop("last_result", None)
    candidate = submitted_url.strip()

    if parse_url(candidate) is None:
        st.error(
            "Enter a valid HTTP or HTTPS URL, such as https://example.org/."
        )
    else:
        try:
            with st.spinner(
                "Loading the dataset, training, and analyzing..."
            ):
                trained = train_models()
                result = analyze_url(candidate, trained)

            st.session_state["last_result"] = {
                "url": candidate,
                **result,
            }

        except Exception as error:
            st.error(
                "Training or analysis failed. The app did not switch "
                "to generated data."
            )
            st.code(str(error), language=None)

report = st.session_state.get("last_result")

if report:
    st.divider()
    st.subheader("Assessment")

    verdict = risk_label(report["combined"])

    if report["combined"] >= 0.70:
        st.error(verdict)
    elif report["combined"] >= 0.40:
        st.warning(verdict)
    else:
        st.info(verdict)

    st.code(report["url"], language=None)
    st.write("**Domain group:**", report["domain"])

    left, middle, right = st.columns(3)
    left.metric("Character model", f"{report['character']:.3f}")
    middle.metric("Structural model", f"{report['structural']:.3f}")
    right.metric("Combined score", f"{report['combined']:.3f}")

    st.progress(report["combined"])

    st.caption(
        "The combined score is a 50/50 average. Scores are not calibrated "
        "probabilities. The 0.40 and 0.70 boundaries are provisional."
    )

    st.subheader("Rule-based indicators")

    if report["indicators"]:
        for note in report["indicators"]:
            st.write("•", note)
    else:
        st.write(
            "No rule-based indicators triggered. "
            "This does not prove that the URL is safe."
        )

    st.caption(
        "Indicators are separate checks; they do not explain every "
        "reason behind a model's score."
    )

    st.divider()

    with st.expander("Model performance and dataset details"):
        # The models have already been cached by the scan.
        trained = train_models()

        st.write(
            f"**Training:** {trained['train_count']:,} URLs "
            f"from {trained['train_domains']:,} domain groups  \n"
            f"**Validation:** {trained['validation_count']:,} URLs "
            f"from {trained['validation_domains']:,} domain groups  \n"
            f"**Test:** {trained['test_count']:,} URLs "
            f"from {trained['test_domains']:,} domain groups"
        )

        st.subheader("Model comparison at 0.50 — test set")
        st.dataframe(
            trained["model_table"],
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("Thresholds — validation set")
        st.dataframe(
            trained["validation_table"],
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("False alarms and missed phishing — test set")
        st.dataframe(
            trained["test_threshold_table"],
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "False-positive rate means the fraction of labelled legitimate "
            "test URLs whose scores reach the listed threshold. "
            "Recall means the fraction of labelled phishing URLs caught."
        )

        st.subheader("Confusion matrix — ensemble at 0.50")
        st.dataframe(
            pd.DataFrame(
                trained["matrix"],
                index=["Labelled legitimate", "Labelled phishing"],
                columns=["Predicted legitimate", "Predicted phishing"],
            ),
            use_container_width=True,
        )

        st.subheader("Dataset bias check — training set")
        st.dataframe(
            trained["bias_table"],
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "Differences in length, HTTPS, or query usage between classes "
            "may let models learn dataset shortcuts. These checks do not "
            "prove that the dataset is unbiased."
        )

        st.markdown(
            "[Dataset source: UCI PhiUSIIL Phishing URL Dataset]"
            "(https://archive.ics.uci.edu/dataset/967/"
            "phiusiil+phishing+url+dataset)"
        )

st.divider()
st.caption(
    "A lower score does not guarantee safety. Do not use this tool "
    "as the only basis for entering credentials or opening a link."
)
