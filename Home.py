import streamlit as st


st.set_page_config(
    page_title="CrossBorder Intelligence | Home",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 82% 4%, rgba(32, 113, 120, .22), transparent 34%),
            #0b1117;
        color: #e6eef1;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: #101920;
        border-right: 1px solid #26343d;
    }
    [data-testid="stSidebar"] * { color: #e6eef1; }
    [data-testid="stMainBlockContainer"] { max-width: 1180px; padding-top: 2rem; }
    .brand {
        color: #75d1c5;
        font-size: .82rem;
        font-weight: 750;
        letter-spacing: .16em;
        text-transform: uppercase;
    }
    .hero {
        background: linear-gradient(120deg, #102d3a 0%, #174e59 68%, #167a77 100%);
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 22px;
        color: #f5fbfb;
        margin: 1.1rem 0 1.4rem;
        overflow: hidden;
        padding: clamp(2rem, 5vw, 4.2rem);
        position: relative;
    }
    .hero:after {
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 50%;
        content: "";
        height: 330px;
        position: absolute;
        right: -80px;
        top: -115px;
        width: 330px;
    }
    .eyebrow {
        color: #a7e4dc;
        font-size: .75rem;
        font-weight: 700;
        letter-spacing: .13em;
        text-transform: uppercase;
    }
    .hero h1 {
        color: #fff;
        font-size: clamp(2.2rem, 5vw, 3.8rem);
        letter-spacing: -.045em;
        line-height: 1.06;
        margin: .8rem 0 1rem;
        max-width: 760px;
    }
    .hero p {
        color: #d3e2e5;
        font-size: 1.08rem;
        line-height: 1.7;
        margin: 0;
        max-width: 720px;
    }
    .route-pill {
        background: rgba(255,255,255,.1);
        border: 1px solid rgba(255,255,255,.18);
        border-radius: 999px;
        color: #e7f4f3;
        display: inline-block;
        font-size: .85rem;
        margin-top: 1.5rem;
        padding: .55rem .9rem;
    }
    .section-label {
        color: #75d1c5;
        font-size: .73rem;
        font-weight: 750;
        letter-spacing: .13em;
        margin: 1rem 0 .45rem;
        text-transform: uppercase;
    }
    .section-title {
        color: #edf4f6;
        font-size: clamp(1.55rem, 3vw, 2.15rem);
        font-weight: 750;
        letter-spacing: -.035em;
        line-height: 1.2;
        margin: 0 0 .55rem;
    }
    .section-copy {
        color: #a7b7be;
        font-size: 1rem;
        line-height: 1.65;
        margin: 0 0 1.25rem;
        max-width: 800px;
    }
    .info-card {
        background: #141e27;
        border: 1px solid #293943;
        border-radius: 15px;
        box-shadow: 0 5px 18px rgba(0, 0, 0, .16);
        height: 100%;
        padding: 1.15rem 1.2rem;
    }
    .info-card h3 {
        color: #e8f0f2;
        font-size: 1rem;
        margin: 0 0 .45rem;
    }
    .info-card p {
        color: #a9bac1;
        font-size: .91rem;
        line-height: 1.55;
        margin: 0;
    }
    .step-number {
        align-items: center;
        background: #173a3b;
        border-radius: 9px;
        color: #8be0d4;
        display: inline-flex;
        font-size: .76rem;
        font-weight: 750;
        height: 29px;
        justify-content: center;
        margin-bottom: .8rem;
        width: 29px;
    }
    .use-case {
        background: #141e27;
        border: 1px solid #293943;
        border-left: 3px solid #16817c;
        border-radius: 12px;
        height: 100%;
        padding: 1rem 1.05rem;
    }
    .use-case h3 {
        color: #e8f0f2;
        font-size: .98rem;
        margin: 0 0 .4rem;
    }
    .use-case p {
        color: #a9bac1;
        font-size: .88rem;
        line-height: 1.5;
        margin: 0;
    }
    .approach {
        background: #132b30;
        border: 1px solid #24494a;
        border-radius: 16px;
        margin: 1.5rem 0 .6rem;
        padding: 1.4rem 1.5rem;
    }
    .approach strong { color: #8be0d4; }
    .approach p {
        color: #b4c7ca;
        line-height: 1.65;
        margin: .35rem 0 0;
    }
    .footer {
        border-top: 1px solid #293943;
        color: #94a5ad;
        font-size: .82rem;
        margin-top: 2.5rem;
        padding: 1rem 0 .5rem;
    }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; }
        .hero { border-radius: 16px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



st.markdown(
    """
    <div class="brand">CrossBorder Intelligence</div>
    <section class="hero">
      <div class="eyebrow">Smarter international payments</div>
      <h1>Make every payment journey more predictable.</h1>
      <p>
        An intelligent analytics platform to anticipate payment failures,
        spot settlement delays and unusual activity, and help teams choose
        better actions across the cross-border payment lifecycle.
      </p>
      <span class="route-pill">India &nbsp;→&nbsp; UK · USA · Europe · Singapore</span>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-label">The project</div>
    <h2 class="section-title">From payment events to better decisions</h2>
    <p class="section-copy">
      International transfers pass through multiple institutions, currencies
      and messaging networks. This project applies data science to improve
      visibility and help operations teams act before small issues become
      failed or late payments.
    </p>
    """,
    unsafe_allow_html=True,
)

journey = [
    ("01", "Initiate", "Customer submits an international payment."),
    ("02", "Validate", "The source bank checks payment details and eligibility."),
    ("03", "Convert", "Funds are converted into the destination currency."),
    ("04", "Route", "Correspondent banks and payment networks carry the transfer."),
    ("05", "Settle", "The beneficiary bank processes, settles or rejects payment."),
]
journey_columns = st.columns(len(journey), gap="small")
for column, (number, title, description) in zip(journey_columns, journey):
    with column:
        st.markdown(
            f"""
            <div class="info-card">
              <span class="step-number">{number}</span>
              <h3>{title}</h3>
              <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="section-label">Operational focus</div>
    <h2 class="section-title">Challenges worth solving</h2>
    <p class="section-copy">
      The platform focuses on the friction that makes cross-border payments
      costly, difficult to predict and hard to track.
    </p>
    """,
    unsafe_allow_html=True,
)

challenges = [
    ("Payment failures", "Understand and reduce avoidable rejections."),
    ("Settlement delays", "Identify transfers at risk of missing their SLA."),
    ("Intermediary charges", "Make routing and cost trade-offs more visible."),
    ("FX fluctuations", "Bring exchange-rate movement into payment decisions."),
    ("Suspicious patterns", "Surface unusual activity for further review."),
    ("Retries and tracking", "Reduce repeated attempts and improve status visibility."),
]
challenge_columns = st.columns(3, gap="medium")
for index, (title, description) in enumerate(challenges):
    with challenge_columns[index % 3]:
        st.markdown(
            f"""
            <div class="info-card" style="margin-bottom: 1rem;">
              <h3>{title}</h3>
              <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="section-label">Data science use cases</div>
    <h2 class="section-title">Five ways to improve payment outcomes</h2>
    <p class="section-copy">
      Each use case connects an analytical method to a practical operational
      decision. Predictions support teams; they do not replace appropriate
      review or controls.
    </p>
    """,
    unsafe_allow_html=True,
)

use_cases = [
    ("Payment success prediction", "Classification", "Flag payments with elevated failure risk so teams can investigate or intervene earlier."),
    ("Settlement delay prediction", "Regression / classification", "Estimate settlement time or identify transactions likely to breach their SLA."),
    ("Payment anomaly detection", "Unsupervised learning", "Surface unusual transaction patterns for investigation."),
    ("Payment routing optimisation", "Optimisation / machine learning", "Compare routing choices to help balance cost, speed and reliability."),
    ("FX rate prediction", "Time-series forecasting", "Forecast exchange-rate trends to inform FX decisions."),
]
use_case_columns = st.columns(2, gap="medium")
for index, (title, method, value) in enumerate(use_cases):
    with use_case_columns[index % 2]:
        st.markdown(
            f"""
            <div class="use-case" style="margin-bottom: 1rem;">
              <h3>{title}</h3>
              <p><strong>{method}</strong><br>{value}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="approach">
      <strong>Intended business value</strong>
      <p>
        Reduce preventable failures, improve SLA performance, increase
        visibility into payment risk, and support more informed routing and
        FX decisions. Actual impact depends on model validation, operational
        adoption and ongoing monitoring.
      </p>
    </div>
    <div class="footer">
      CrossBorder Intelligence · Data science for safer, more predictable payments
    </div>
    """,
    unsafe_allow_html=True,
)