import ipaddress
from urllib.parse import urlsplit

import streamlit as st

st.set_page_config(page_title="LinkGuard", page_icon="🛡️")

st.caption("LINKGUARD · RESEARCH STARTER")
st.title("Inspect the link. Before the click.")
st.write("Check basic URL warning signs without opening the website.")

url = st.text_input(
    "Paste a full URL",
    placeholder="https://example.org/login",
)

if st.button("Inspect URL", type="primary"):
    try:
        text = url.strip()
        if not text or any(character.isspace() for character in text):
            raise ValueError

        parsed = urlsplit(text)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError

        _ = parsed.port  # Reject invalid ports.
        host = parsed.hostname
        warnings = []

        if parsed.scheme == "http":
            warnings.append("The URL does not use HTTPS.")
        if "@" in parsed.netloc:
            warnings.append("The address contains an @ symbol.")
        if "xn--" in host:
            warnings.append("The hostname contains punycode.")
        if len(text) > 120:
            warnings.append("The URL is unusually long.")

        try:
            ipaddress.ip_address(host)
            warnings.append("The hostname is an IP address.")
        except ValueError:
            pass

        st.write("**Hostname:**", host)
        if warnings:
            st.warning("Warning signs to review — not a phishing verdict:")
            for item in warnings:
                st.write("•", item)
        else:
            st.info("No basic warning signs found. This does not mean the URL is safe.")

    except ValueError:
        st.error("Enter a full HTTP or HTTPS URL, such as https://example.org.")

st.divider()
st.subheader("Next: the full detector")
st.write("The character model, structural model, and VirusTotal reports can be added after the starter is online.")
st.caption("This version does not run machine learning or visit the submitted website.")
