import os
import pickle
import warnings
 
import pandas as pd
from flask import Flask, render_template_string, request
 
warnings.filterwarnings("ignore")
 
app = Flask(__name__)
 
# --------------------------------------------------------------------------
# Load model
# --------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logistic.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
 
FEATURES = list(model.feature_names_in_)    # exact column order the model expects
CLASSES = [int(c) for c in model.classes_]  # [0, 1]  ->  0 = rejected, 1 = approved
 
# --------------------------------------------------------------------------
# Label-encoded categories (alphabetical, like sklearn's LabelEncoder).
# Position in each list = the number the model receives. Rename if yours differ.
# --------------------------------------------------------------------------
GENDER     = ["Female", "Male"]
YES_NO     = ["No", "Yes"]
DEPENDENTS = ["0", "1", "2", "3+"]
EDUCATION  = ["Graduate", "Not Graduate"]
AREA       = ["Rural", "Semiurban", "Urban"]
TERMS      = [12, 36, 60, 84, 120, 180, 240, 300, 360, 480]   # months
 
 
def opts(lst):
    return list(enumerate(lst))
 
 
# --------------------------------------------------------------------------
# Form definition
# --------------------------------------------------------------------------
SECTIONS = [
    {"title": "👤 Applicant Details", "fields": [
        {"name": "Gender", "label": "Gender", "kind": "select", "options": opts(GENDER), "default": 1},
        {"name": "Married", "label": "Married", "kind": "select", "options": opts(YES_NO), "default": 1},
        {"name": "Dependents", "label": "Dependents", "kind": "select", "options": opts(DEPENDENTS), "default": 0},
        {"name": "Education", "label": "Education", "kind": "select", "options": opts(EDUCATION), "default": 0},
        {"name": "Self_Employed", "label": "Self Employed", "kind": "select", "options": opts(YES_NO), "default": 0},
    ]},
    {"title": "💰 Income", "fields": [
        {"name": "ApplicantIncome", "label": "Applicant Income (monthly)", "kind": "number",
         "min": 0, "max": 100000, "step": 100, "default": 5000},
        {"name": "CoapplicantIncome", "label": "Co-applicant Income (monthly)", "kind": "number",
         "min": 0, "max": 50000, "step": 100, "default": 1500},
        {"name": "Credit_History", "label": "Credit History", "kind": "select",
         "options": [(1, "Good (meets guidelines)"), (0, "Poor (does not meet guidelines)")], "default": 1},
    ]},
    {"title": "🏠 Loan Request", "fields": [
        {"name": "LoanAmount", "label": "Loan Amount (in thousands)", "kind": "range",
         "min": 10, "max": 700, "step": 1, "default": 130},
        {"name": "Loan_Amount_Term", "label": "Loan Term (months)", "kind": "select",
         "options": [(t, f"{t} months ({t // 12} yr)") for t in TERMS], "default": 360},
        {"name": "Property_Area", "label": "Property Area", "kind": "select",
         "options": opts(AREA), "default": 1},
    ]},
]
 
 
# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def verdict_info(p: float) -> dict:
    """p = approval probability in 0-100."""
    if p >= 65:
        return {"level": "High Chance", "verdict": "Likely Approved", "icon": "✅", "color": "#10b981",
                "msg": "This application matches the profile of approved loans."}
    if p >= 40:
        return {"level": "Borderline", "verdict": "Needs Review", "icon": "⚖️", "color": "#fbbf24",
                "msg": "Signals are mixed. A manual review is recommended."}
    return {"level": "Low Chance", "verdict": "Likely Rejected", "icon": "❌", "color": "#f43f5e",
            "msg": "This application resembles loans that were not approved."}
 
 
def build_tips(v: dict) -> list:
    tips = []
    if int(v["Credit_History"]) == 0:
        tips.append("📉 Poor credit history is the biggest factor, so clearing past dues would help most.")
    else:
        tips.append("💳 A good credit history is the strongest positive signal here.")
    total_income = v["ApplicantIncome"] + v["CoapplicantIncome"]
    if total_income < 4000:
        tips.append("💼 Household income is on the lower side; adding a co-applicant may strengthen the case.")
    if v["CoapplicantIncome"] == 0:
        tips.append("🤝 No co-applicant income is listed. A co-applicant can improve repayment capacity.")
    if int(v["Self_Employed"]) == 1:
        tips.append("📄 Self-employed applicants should keep income proof and tax returns ready.")
    return tips[:4]
 
 
# --------------------------------------------------------------------------
# HTML template
# --------------------------------------------------------------------------
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Loan Approval Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root{
    --bg1:#071226; --bg2:#0f1f45; --card:rgba(255,255,255,.07); --border:rgba(255,255,255,.14);
    --text:#eff4ff; --muted:#9fb0d0; --gold:#fbbf24; --green:#10b981; --sky:#38bdf8;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{
    font-family:'Plus Jakarta Sans',system-ui,sans-serif;color:var(--text);min-height:100vh;padding:34px 16px 60px;
    background:
      radial-gradient(800px 480px at 0% 0%, rgba(251,191,36,.20), transparent 60%),
      radial-gradient(800px 520px at 100% 5%, rgba(56,189,248,.24), transparent 60%),
      radial-gradient(700px 500px at 50% 115%, rgba(16,185,129,.20), transparent 60%),
      linear-gradient(160deg,var(--bg1),var(--bg2));
    background-attachment:fixed;
  }
  .wrap{max-width:1080px;margin:0 auto}
  header{text-align:center;margin-bottom:30px}
  .badge{display:inline-block;padding:6px 16px;border-radius:999px;font-size:12px;font-weight:600;
    letter-spacing:.08em;text-transform:uppercase;color:var(--gold);
    background:rgba(251,191,36,.12);border:1px solid rgba(251,191,36,.38)}
  h1{font-size:clamp(28px,5vw,46px);font-weight:800;margin:14px 0 8px;
    background:linear-gradient(90deg,#fff,var(--gold) 45%,var(--sky));
    -webkit-background-clip:text;background-clip:text;color:transparent}
  header p{color:var(--muted);max-width:560px;margin:0 auto;font-size:15px}
 
  /* result */
  .result{margin-bottom:28px;padding:30px;border-radius:24px;display:flex;gap:34px;align-items:center;
    justify-content:center;flex-wrap:wrap;
    background:linear-gradient(135deg,color-mix(in srgb,var(--rc) 20%,transparent),rgba(255,255,255,.04));
    border:1px solid color-mix(in srgb,var(--rc) 55%,transparent);
    box-shadow:0 0 60px color-mix(in srgb,var(--rc) 22%,transparent);animation:pop .5s ease}
  @keyframes pop{from{opacity:0;transform:translateY(14px) scale(.98)}to{opacity:1;transform:none}}
  .ring{position:relative;width:190px;height:190px}
  .ring svg{transform:rotate(-90deg)}
  .ring .bg{stroke:rgba(255,255,255,.12)}
  .ring .fg{stroke:var(--rc);stroke-linecap:round;stroke-dasharray:502;stroke-dashoffset:502;
    animation:fill 1.2s ease forwards;filter:drop-shadow(0 0 8px var(--rc))}
  @keyframes fill{to{stroke-dashoffset:calc(502 - 502 * var(--pct) / 100)}}
  .ring .center{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
  .ring .num{font-size:42px;font-weight:800;line-height:1}
  .ring .of{font-size:12px;color:var(--muted);margin-top:4px}
  .info{max-width:440px}
  .level{display:inline-block;font-size:13px;font-weight:700;padding:4px 14px;border-radius:999px;
    background:var(--rc);color:#061225;margin-bottom:10px}
  .info h2{font-size:28px;margin-bottom:6px}
  .info p.msg{color:#d3def5;font-size:14px;margin-bottom:14px}
  .bars{margin-bottom:14px}
  .bar-row{display:flex;align-items:center;gap:10px;margin:7px 0;font-size:13px}
  .bar-row .n{width:62px;color:var(--muted)}
  .track{flex:1;height:10px;border-radius:99px;background:rgba(255,255,255,.1);overflow:hidden}
  .fill{height:100%;border-radius:99px}
  .bar-row .v{width:48px;text-align:right;font-weight:600}
  .tips{list-style:none;display:grid;gap:8px}
  .tips li{font-size:13px;padding:9px 12px;border-radius:12px;background:rgba(255,255,255,.07);
    border:1px solid var(--border)}
 
  .error{background:rgba(244,63,94,.15);border:1px solid rgba(244,63,94,.5);
    padding:14px 18px;border-radius:14px;margin-bottom:20px}
 
  /* form */
  form{display:grid;gap:20px}
  .grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}
  .card{background:var(--card);border:1px solid var(--border);border-radius:20px;padding:22px;
    backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px)}
  .card h2{font-size:17px;margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid var(--border)}
  .field{margin-bottom:16px}.field:last-child{margin-bottom:0}
  label{display:flex;justify-content:space-between;font-size:13px;font-weight:500;color:var(--muted);margin-bottom:7px}
  label output{color:var(--gold);font-weight:700}
  select,input[type=number]{width:100%;padding:11px 12px;border-radius:12px;font:inherit;color:var(--text);
    background:rgba(6,14,32,.6);border:1px solid var(--border);outline:none;transition:.2s}
  select option{background:#0f1f45}
  select:focus,input[type=number]:focus{border-color:var(--sky);box-shadow:0 0 0 3px rgba(56,189,248,.25)}
  input[type=range]{width:100%;-webkit-appearance:none;height:6px;border-radius:99px;outline:none;
    background:linear-gradient(90deg,var(--gold),var(--sky))}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:20px;height:20px;border-radius:50%;
    background:#fff;border:3px solid var(--gold);cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.4)}
  input[type=range]::-moz-range-thumb{width:16px;height:16px;border-radius:50%;background:#fff;
    border:3px solid var(--gold);cursor:pointer}
 
  .actions{display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:6px}
  button,.reset{padding:15px 38px;border-radius:14px;border:0;font:inherit;font-weight:700;font-size:16px;
    cursor:pointer;transition:.2s;text-decoration:none;display:inline-block}
  button{color:#061225;background:linear-gradient(90deg,var(--gold),#34d399);
    box-shadow:0 10px 30px rgba(251,191,36,.3)}
  button:hover{transform:translateY(-2px);box-shadow:0 14px 36px rgba(52,211,153,.4)}
  .reset{color:var(--text);background:var(--card);border:1px solid var(--border)}
  .reset:hover{background:rgba(255,255,255,.12)}
  footer{text-align:center;color:var(--muted);font-size:12px;margin-top:34px;max-width:620px;
    margin-left:auto;margin-right:auto}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <span class="badge">Logistic Regression · FinTech</span>
    <h1>Loan Approval Predictor</h1>
    <p>Enter the applicant's details and loan request to estimate the chance of approval.</p>
  </header>
 
  {% if error %}<div class="error">⚠️ {{ error }}</div>{% endif %}
 
  {% if result %}
  <div class="result" style="--rc: {{ result.color }}; --pct: {{ result.p_yes }}">
    <div class="ring">
      <svg width="190" height="190" viewBox="0 0 190 190">
        <circle class="bg" cx="95" cy="95" r="80" fill="none" stroke-width="14"/>
        <circle class="fg" cx="95" cy="95" r="80" fill="none" stroke-width="14"/>
      </svg>
      <div class="center"><span class="num">{{ result.p_yes }}%</span><span class="of">approval chance</span></div>
    </div>
    <div class="info">
      <span class="level">{{ result.icon }} {{ result.level }}</span>
      <h2>{{ result.verdict }}</h2>
      <p class="msg">{{ result.msg }}</p>
      <div class="bars">
        <div class="bar-row"><span class="n">Approved</span>
          <div class="track"><div class="fill" style="width:{{ result.p_yes }}%;background:#10b981"></div></div>
          <span class="v">{{ result.p_yes }}%</span></div>
        <div class="bar-row"><span class="n">Rejected</span>
          <div class="track"><div class="fill" style="width:{{ result.p_no }}%;background:#f43f5e"></div></div>
          <span class="v">{{ result.p_no }}%</span></div>
      </div>
      <ul class="tips">{% for t in result.tips %}<li>{{ t }}</li>{% endfor %}</ul>
    </div>
  </div>
  {% endif %}
 
  <form method="POST" action="/">
    <div class="grid3">
    {% for s in sections %}
      <div class="card">
        <h2>{{ s.title }}</h2>
        {% for f in s.fields %}
        <div class="field">
          {% set val = values.get(f.name, f.default) %}
          {% if f.kind == 'select' %}
            <label for="{{ f.name }}">{{ f.label }}</label>
            <select id="{{ f.name }}" name="{{ f.name }}">
              {% for code, text in f.options %}
              <option value="{{ code }}" {{ 'selected' if code|string == val|string }}>{{ text }}</option>
              {% endfor %}
            </select>
          {% elif f.kind == 'range' %}
            <label for="{{ f.name }}">{{ f.label }}<output id="o_{{ f.name }}">{{ val }}</output></label>
            <input type="range" id="{{ f.name }}" name="{{ f.name }}" min="{{ f.min }}" max="{{ f.max }}"
                   step="{{ f.step }}" value="{{ val }}"
                   oninput="document.getElementById('o_{{ f.name }}').textContent=this.value">
          {% else %}
            <label for="{{ f.name }}">{{ f.label }}</label>
            <input type="number" id="{{ f.name }}" name="{{ f.name }}" min="{{ f.min }}" max="{{ f.max }}"
                   step="{{ f.step }}" value="{{ val }}" required>
          {% endif %}
        </div>
        {% endfor %}
      </div>
    {% endfor %}
    </div>
    <div class="actions">
      <button type="submit">🏦 Check Loan Approval</button>
      <a class="reset" href="/">↺ Reset</a>
    </div>
  </form>
  <footer>Built with Flask &amp; scikit-learn. This is a statistical estimate only and does not
    represent a lender's final decision.</footer>
</div>
</body>
</html>
"""
 
 
# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result, error, values = None, None, {}
 
    if request.method == "POST":
        values = request.form.to_dict()
        try:
            v = {name: float(values[name]) for name in FEATURES}
            X = pd.DataFrame([v])[FEATURES]
 
            proba = model.predict_proba(X)[0]
            p_yes = round(float(proba[CLASSES.index(1)]) * 100, 1)
            p_no = round(100 - p_yes, 1)
 
            result = {"p_yes": p_yes, "p_no": p_no, "tips": build_tips(v), **verdict_info(p_yes)}
        except (KeyError, ValueError) as e:
            error = f"Please fill every field with valid values ({e})."
        except Exception as e:  # noqa: BLE001
            error = f"Prediction failed: {e}"
 
    return render_template_string(
        TEMPLATE, sections=SECTIONS, values=values, result=result, error=error
    )
 
 
if __name__ == "__main__":
    app.run(debug=True)
 
