# ═══════════════════════════════════════════════════════════════
# gradio_ui.py — واجهة Gradio الاحترافية الكاملة
# ═══════════════════════════════════════════════════════════════

import random, time, tempfile
import pandas as pd
import gradio as gr

from config import MODELS, CITIES, FEATURES_POOL, CONDITIONS, AGENT_NAMES, AGENCIES
from models_logic import (
    generate_batch, bulk_generate, SYSTEM_PROMPT,
    make_prompt, call_model, safe_parse,
)

# ── تخزين بيانات الجلسة ──
_session: list[dict] = []

MODEL_CHOICES = [m["display_name"] for m in MODELS.values()]
CITIES_LIST   = list(CITIES.keys())


def _model_key(display_name: str) -> str:
    return next((k for k, v in MODELS.items() if v["display_name"] == display_name), "gpt-4o-mini")


# ══════════════════════════════════════════════════════════════
# دوال المنطق
# ══════════════════════════════════════════════════════════════

def ui_generate(
    model_choice,
    cities_sel, prop_types_sel, listing_types_sel,
    total,
    price_min, price_max,
    area_min, area_max,
    rooms_min, rooms_max,
    date_start, date_end,
    output_name,
    tier_choice, extra, retry_on_fail,
):
    global _session

    model_key = _model_key(model_choice)
    m         = MODELS[model_key]
    batch_sz  = m["batch_size"]

    target_cities   = cities_sel        if cities_sel        else list(CITIES.keys())
    target_ptypes   = prop_types_sel    if prop_types_sel    else ["شقة", "منزل"]
    target_listings = listing_types_sel if listing_types_sel else ["بيع", "إيجار"]
    total = int(total)

    tier_map = {
        "متوازن":    [0.20, 0.50, 0.30],
        "راقي فقط":  [1.00, 0.00, 0.00],
        "متوسط فقط": [0.00, 1.00, 0.00],
        "شعبي فقط":  [0.00, 0.00, 1.00],
    }
    tier_w = tier_map.get(tier_choice, [0.20, 0.50, 0.30])

    # بناء قيود السعر/المساحة/الغرف
    constraints = []
    if price_min: constraints.append(f"السعر لا يقل عن {int(price_min):,} دولار")
    if price_max: constraints.append(f"السعر لا يزيد عن {int(price_max):,} دولار")
    if area_min:  constraints.append(f"المساحة لا تقل عن {int(area_min)} م²")
    if area_max:  constraints.append(f"المساحة لا تزيد عن {int(area_max)} م²")
    if rooms_min: constraints.append(f"غرف النوم لا تقل عن {int(rooms_min)}")
    if rooms_max: constraints.append(f"غرف النوم لا تزيد عن {int(rooms_max)}")

    extra_full = extra.strip()
    if constraints:
        extra_full = "قيود صارمة:\n" + "\n".join(f"• {c}" for c in constraints) + "\n\n" + extra_full

    all_records, errors, generated = [], 0, 0
    log_lines = [f"🚀 بدأ التوليد | {m['display_name']} | هدف: {total} سجل"]

    while generated < total:
        batch   = min(batch_sz, total - generated)
        city    = random.choice(target_cities)
        ptype   = random.choices(target_ptypes,   weights=([0.75, 0.25] if len(target_ptypes)   == 2 else [1])[:len(target_ptypes)])[0]
        listing = random.choices(target_listings, weights=([0.60, 0.40] if len(target_listings) == 2 else [1])[:len(target_listings)])[0]

        try:
            city_data = CITIES[city]
            tier      = random.choices(["راقي", "متوسط", "شعبي"], weights=tier_w)[0]
            tier_key  = {"راقي": "أحياء_راقية", "متوسط": "أحياء_متوسطة", "شعبي": "أحياء_شعبية"}[tier]
            nbr       = random.choice(city_data[tier_key])

            prompt = make_prompt(city, nbr, tier, ptype, listing, batch, (date_start, date_end))
            if extra_full:
                prompt += f"\n\nتعليمات إضافية يجب الالتزام بها:\n{extra_full}"

            raw  = call_model(model_key, SYSTEM_PROMPT, prompt)
            recs = safe_parse(raw)
            all_records.extend(recs)
            generated += len(recs)
            log_lines.append(f"✅ +{len(recs)} | {city}/{nbr} | {ptype}/{listing} → إجمالي: {generated}")

        except Exception as e:
            errors += 1
            log_lines.append(f"⚠️  خطأ [{city}/{ptype}]: {str(e)[:60]}")
            if not retry_on_fail or errors > 10:
                log_lines.append("❌ تجاوز الحد الأقصى للأخطاء. توقف.")
                break
            time.sleep(2)

    _session.extend(all_records)

    if not all_records:
        return "❌ فشل التوليد — تحقق من مفاتيح API", pd.DataFrame(), None, None, None, "\n".join(log_lines)

    df = pd.DataFrame(all_records)
    for col in ["السعر_دولار", "المساحة_م2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    safe_name = (output_name.strip() or "عقارات_سوريا").replace(" ", "_")
    csv_path  = f"/tmp/{safe_name}.csv"
    json_path = f"/tmp/{safe_name}.json"
    xlsx_path = f"/tmp/{safe_name}.xlsx"
    df.to_csv(csv_path,   index=False, encoding="utf-8-sig")
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    df.to_excel(xlsx_path, index=False)

    avg_price = df["السعر_دولار"].mean() if "السعر_دولار" in df.columns else 0
    avg_area  = df["المساحة_م2"].mean()  if "المساحة_م2"  in df.columns else 0
    city_dist = df["المدينة"].value_counts().to_dict()     if "المدينة"    in df.columns else {}
    ptype_dist= df["نوع_العقار"].value_counts().to_dict()  if "نوع_العقار" in df.columns else {}

    city_rows  = "\n".join(f"| {c} | {n} |" for c, n in city_dist.items())
    ptype_rows = "\n".join(f"| {t} | {n} |" for t, n in ptype_dist.items())

    summary = f"""## ✅ اكتمل التوليد!

| المعلومة | القيمة |
|---|---|
| النموذج | {m['display_name']} |
| السجلات المولّدة | **{len(df)}** |
| إجمالي الجلسة | **{len(_session)}** |
| متوسط السعر | **${avg_price:,.0f}** |
| متوسط المساحة | **{avg_area:.0f} م²** |
| الأخطاء | {errors} |

**توزيع المدن:**
| المدينة | العدد |
|---|---|
{city_rows}

**توزيع أنواع العقار:**
| النوع | العدد |
|---|---|
{ptype_rows}

> 💡 *{m['تعليق']}*
"""
    return summary, df, csv_path, json_path, xlsx_path, "\n".join(log_lines[-25:])


def ui_export_session():
    if not _session:
        return None, None, None
    df = pd.DataFrame(_session)
    df.to_csv("/tmp/session_all.csv",   index=False, encoding="utf-8-sig")
    df.to_json("/tmp/session_all.json", orient="records", force_ascii=False, indent=2)
    df.to_excel("/tmp/session_all.xlsx", index=False)
    return "/tmp/session_all.csv", "/tmp/session_all.json", "/tmp/session_all.xlsx"


def ui_clear():
    global _session
    _session = []
    return "✅ تم مسح بيانات الجلسة", pd.DataFrame(), None, None, None, ""


def ui_compare(m_a, m_b, city, ptype, listing, cnt):
    k_a, k_b = _model_key(m_a), _model_key(m_b)
    cnt = int(cnt)
    results = {}
    for key in [k_a, k_b]:
        try:
            recs = generate_batch(key, city, ptype, listing, cnt, ("2024-01-01", "2025-12-31"))
            df_r = pd.DataFrame(recs)
            if "السعر_دولار" in df_r.columns:
                df_r["السعر_دولار"] = pd.to_numeric(df_r["السعر_دولار"], errors="coerce")
            avg = df_r["السعر_دولار"].mean() if "السعر_دولار" in df_r.columns else 0
            msg = f"✅ **{len(recs)} سجل** | متوسط السعر: ${avg:,.0f}\n\n{MODELS[key]['توصية']}"
            results[key] = (msg, df_r)
        except Exception as e:
            results[key] = (f"❌ **خطأ:** {e}", pd.DataFrame())
    return results[k_a][0], results[k_a][1], results[k_b][0], results[k_b][1]


def colab_download(csv_p, json_p, xlsx_p):
    try:
        from google.colab import files
        for p in [csv_p, json_p, xlsx_p]:
            if p:
                files.download(p)
        return "✅ بدأ التحميل!"
    except Exception as e:
        return f"⚠️ {e}"


# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&display=swap');
* { font-family: 'Cairo', sans-serif !important; }
body, .gradio-container { background: #080c18 !important; direction: rtl; }

.hero {
    background: linear-gradient(135deg, #0d1f3c 0%, #1a3a5c 50%, #0d2d4a 100%);
    border: 1px solid rgba(56,189,248,0.25); border-radius: 20px;
    padding: 36px 44px; margin-bottom: 20px; position: relative; overflow: hidden;
    box-shadow: 0 8px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
}
.hero::before {
    content:''; position:absolute; top:-80px; right:-60px; width:320px; height:320px;
    background:radial-gradient(circle,rgba(56,189,248,0.15) 0%,transparent 70%); border-radius:50%;
}
.hero h1 { color:#f0f9ff !important; font-size:2.1em !important; font-weight:900 !important; margin:0 0 6px 0 !important; }
.hero .sub { color:#7dd3fc !important; font-size:1.05em; }
.hero .badges { margin-top:16px; display:flex; flex-wrap:wrap; gap:8px; }
.hero .badge { background:rgba(56,189,248,0.12); border:1px solid rgba(56,189,248,0.3); color:#bae6fd; padding:4px 14px; border-radius:20px; font-size:0.82em; }

.section-card { background:rgba(255,255,255,0.025) !important; border:1px solid rgba(56,189,248,0.12) !important; border-radius:14px !important; padding:18px 20px !important; margin-bottom:14px !important; }
.section-title { color:#7dd3fc !important; font-size:0.95em !important; font-weight:700 !important; margin:0 0 12px 0 !important; }

.btn-generate { background:linear-gradient(135deg,#1e40af,#3b82f6) !important; border:1px solid rgba(96,165,250,0.5) !important; color:#fff !important; font-size:1.15em !important; font-weight:800 !important; border-radius:12px !important; padding:15px !important; box-shadow:0 4px 24px rgba(59,130,246,0.4) !important; transition:all 0.2s !important; }
.btn-generate:hover { box-shadow:0 8px 32px rgba(59,130,246,0.6) !important; transform:translateY(-2px) !important; }
.btn-clear { background:rgba(239,68,68,0.12) !important; border:1px solid rgba(239,68,68,0.3) !important; color:#fca5a5 !important; border-radius:10px !important; }
.btn-dl { background:rgba(34,197,94,0.1) !important; border:1px solid rgba(34,197,94,0.3) !important; color:#86efac !important; border-radius:10px !important; font-size:0.88em !important; }
.btn-session { background:rgba(168,85,247,0.12) !important; border:1px solid rgba(168,85,247,0.3) !important; color:#d8b4fe !important; border-radius:10px !important; }

label { color:#93c5fd !important; font-weight:600 !important; font-size:0.88em !important; }
.gradio-textbox input, .gradio-textbox textarea { background:rgba(15,23,42,0.8) !important; color:#e2e8f0 !important; border-color:rgba(56,189,248,0.2) !important; }
.log-box > label + div, .log-box textarea { background:#050a14 !important; color:#4ade80 !important; font-family:'Courier New',monospace !important; font-size:0.80em !important; border-color:rgba(56,189,248,0.1) !important; }
.model-card { background:rgba(30,58,138,0.15); border:1px solid rgba(59,130,246,0.2); border-radius:10px; padding:12px 16px; font-size:0.86em; color:#94a3b8; line-height:1.8; }
"""

# ══════════════════════════════════════════════════════════════
# بناء demo
# ══════════════════════════════════════════════════════════════

def build_demo():
    with gr.Blocks(
        css=CSS,
        title="🏠 مولّد البيانات العقارية السورية",
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
    ) as demo:

        gr.HTML("""
        <div class='hero'>
          <h1>🏠 مولّد البيانات العقارية السورية</h1>
          <p class='sub'>Syrian Real Estate Synthetic Data Generator · 2024 – 2025</p>
          <div class='badges'>
            <span class='badge'>🏙️ 7 مدن</span><span class='badge'>🏠 شقق ومنازل</span>
            <span class='badge'>⚡ 50–500 سجل</span><span class='badge'>🤖 5 نماذج AI</span>
            <span class='badge'>💾 CSV · JSON · XLSX</span><span class='badge'>🆓 Llama محلي</span>
          </div>
        </div>
        """)

        with gr.Tabs():

            # ── تبويب 1: التوليد ──
            with gr.TabItem("⚡ توليد البيانات"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=320):

                        with gr.Group(elem_classes=["section-card"]):
                            gr.HTML("<p class='section-title'>🤖 النموذج</p>")
                            model_dd = gr.Dropdown(choices=MODEL_CHOICES, value=MODEL_CHOICES[0], label="اختر نموذج AI")
                            gr.HTML("""<div class='model-card'>
                              <b>⚡ GPT-4o-mini</b> — الأسرع والأرخص · مثالي للإنتاج الضخم<br>
                              <b>🏆 GPT-4o</b> — الأقوى · أوصاف أغنى وأدق<br>
                              <b>🐦 Claude Haiku</b> — JSON نظيف بلا أخطاء<br>
                              <b>🎯 Claude Sonnet</b> — الأفضل للعربية والسياق الثقافي<br>
                              <b>🆓 Llama 3B</b> — مجاني · للاختبار فقط
                            </div>""")

                        with gr.Group(elem_classes=["section-card"]):
                            gr.HTML("<p class='section-title'>📍 الموقع الجغرافي</p>")
                            cities_cb = gr.CheckboxGroup(choices=CITIES_LIST, value=["دمشق", "حلب", "اللاذقية", "طرطوس"], label="المدن (فارغ = كل المدن)")
                            tier_dd   = gr.Dropdown(choices=["متوازن", "راقي فقط", "متوسط فقط", "شعبي فقط"], value="متوازن", label="توزيع الأحياء")

                        with gr.Group(elem_classes=["section-card"]):
                            gr.HTML("<p class='section-title'>🏠 نوع العقار والإعلان</p>")
                            with gr.Row():
                                prop_cb    = gr.CheckboxGroup(choices=["شقة", "منزل"], value=["شقة", "منزل"], label="العقار")
                                listing_cb = gr.CheckboxGroup(choices=["بيع", "إيجار"], value=["بيع", "إيجار"], label="الإعلان")

                        with gr.Group(elem_classes=["section-card"]):
                            gr.HTML("<p class='section-title'>💰 فلاتر السعر والمساحة</p>")
                            with gr.Row():
                                price_min = gr.Number(label="أدنى سعر ($)", value=None, precision=0)
                                price_max = gr.Number(label="أعلى سعر ($)", value=None, precision=0)
                            with gr.Row():
                                area_min  = gr.Number(label="أدنى مساحة (م²)", value=None, precision=0)
                                area_max  = gr.Number(label="أعلى مساحة (م²)", value=None, precision=0)
                            with gr.Row():
                                rooms_min = gr.Number(label="أدنى غرف نوم", value=None, precision=0)
                                rooms_max = gr.Number(label="أعلى غرف نوم", value=None, precision=0)

                        with gr.Group(elem_classes=["section-card"]):
                            gr.HTML("<p class='section-title'>📊 الكمية والفترة الزمنية</p>")
                            total_sl = gr.Slider(50, 500, value=200, step=50, label="عدد السجلات")
                            with gr.Row():
                                date_s = gr.Textbox(value="2024-01-01", label="من")
                                date_e = gr.Textbox(value="2025-12-31", label="إلى")
                            output_name = gr.Textbox(value="عقارات_سوريا", label="اسم ملف الإخراج")

                        with gr.Group(elem_classes=["section-card"]):
                            gr.HTML("<p class='section-title'>✏️ تعليمات إضافية (اختياري)</p>")
                            extra     = gr.Textbox(placeholder="مثال: ركّز على الشقق فوق الطابق الخامس...", lines=3, label="")
                            retry_chk = gr.Checkbox(value=True, label="إعادة المحاولة تلقائياً عند الفشل")

                        with gr.Row():
                            gen_btn   = gr.Button("🚀  ولّد البيانات الآن", variant="primary", elem_classes=["btn-generate"], scale=3)
                            clear_btn = gr.Button("🗑️ مسح", scale=1, elem_classes=["btn-clear"])

                    with gr.Column(scale=2):
                        summary_md = gr.Markdown("_اضغط **ولّد البيانات** للبدء..._")
                        data_df    = gr.Dataframe(label="📋 البيانات المولّدة", interactive=False, wrap=True)

                        gr.HTML("<p style='color:#7dd3fc;font-weight:700;margin:12px 0 6px;'>💾 تحميل الدفعة الحالية</p>")
                        with gr.Row():
                            dl_csv  = gr.File(label="⬇️ CSV")
                            dl_json = gr.File(label="⬇️ JSON")
                            dl_xlsx = gr.File(label="⬇️ XLSX")

                        gr.HTML("<p style='color:#d8b4fe;font-weight:700;margin:12px 0 6px;'>📦 تصدير كامل الجلسة</p>")
                        exp_btn = gr.Button("📦 اجمع كل الجلسة وصدّرها", elem_classes=["btn-session"])
                        with gr.Row():
                            exp_csv  = gr.File(label="📦 CSV الجلسة")
                            exp_json = gr.File(label="📦 JSON الجلسة")
                            exp_xlsx = gr.File(label="📦 XLSX الجلسة")

                        gr.HTML("<p style='color:#86efac;font-weight:700;margin:12px 0 6px;'>⬇️ تحميل مباشر (Colab)</p>")
                        with gr.Row():
                            colab_dl_btn = gr.Button("⬇️ حمّل للجهاز الآن", elem_classes=["btn-dl"])
                            colab_status = gr.Textbox(label="", interactive=False, lines=1)

                        log_box = gr.Textbox(label="📋 سجل التوليد", lines=8, interactive=False, elem_classes=["log-box"])

                # ربط الأحداث
                gen_btn.click(
                    fn=ui_generate,
                    inputs=[model_dd, cities_cb, prop_cb, listing_cb, total_sl,
                            price_min, price_max, area_min, area_max, rooms_min, rooms_max,
                            date_s, date_e, output_name, tier_dd, extra, retry_chk],
                    outputs=[summary_md, data_df, dl_csv, dl_json, dl_xlsx, log_box],
                )
                clear_btn.click(fn=ui_clear, outputs=[summary_md, data_df, dl_csv, dl_json, dl_xlsx, log_box])
                exp_btn.click(fn=ui_export_session, outputs=[exp_csv, exp_json, exp_xlsx])
                colab_dl_btn.click(fn=colab_download, inputs=[dl_csv, dl_json, dl_xlsx], outputs=[colab_status])

            # ── تبويب 2: مقارنة النماذج ──
            with gr.TabItem("🔬 مقارنة النماذج"):
                gr.Markdown("### جرّب نفس الطلب على نموذجين وقارن الجودة")
                with gr.Row():
                    cmp_city    = gr.Dropdown(choices=CITIES_LIST, value="دمشق", label="المدينة")
                    cmp_ptype   = gr.Dropdown(choices=["شقة", "منزل"], value="شقة", label="نوع العقار")
                    cmp_listing = gr.Dropdown(choices=["بيع", "إيجار"], value="بيع", label="الإعلان")
                    cmp_cnt     = gr.Slider(1, 5, value=2, step=1, label="سجلات لكل نموذج")
                cmp_btn = gr.Button("⚡ قارن الآن", variant="primary", elem_classes=["btn-generate"])
                with gr.Row():
                    with gr.Column():
                        ma_dd  = gr.Dropdown(choices=MODEL_CHOICES, value=MODEL_CHOICES[0], label="النموذج A")
                        ma_out = gr.Markdown("_..._")
                        ma_df  = gr.Dataframe(interactive=False, wrap=True)
                    with gr.Column():
                        mb_dd  = gr.Dropdown(choices=MODEL_CHOICES, value=MODEL_CHOICES[-1], label="النموذج B")
                        mb_out = gr.Markdown("_..._")
                        mb_df  = gr.Dataframe(interactive=False, wrap=True)
                cmp_btn.click(fn=ui_compare, inputs=[ma_dd, mb_dd, cmp_city, cmp_ptype, cmp_listing, cmp_cnt], outputs=[ma_out, ma_df, mb_out, mb_df])

            # ── تبويب 3: دليل ──
            with gr.TabItem("📚 دليل الأسعار والنماذج"):
                gr.Markdown("""
## 🤖 النماذج الخمسة

| النموذج | السرعة | العربية | التكلفة | Batch | التوصية |
|---------|--------|---------|---------|-------|---------|
| ⚡ GPT-4o-mini | ⚡⚡⚡ | ⭐⭐⭐⭐ | 💰 | 15 | ✅ الإنتاج الضخم |
| 🏆 GPT-4o | ⚡⚡ | ⭐⭐⭐⭐⭐ | 💰💰💰 | 10 | ✅ الجودة القصوى |
| 🐦 Claude Haiku | ⚡⚡⚡ | ⭐⭐⭐⭐ | 💰 | 15 | ✅ JSON دقيق |
| 🎯 Claude Sonnet | ⚡⚡ | ⭐⭐⭐⭐⭐ | 💰💰 | 10 | ✅ المشاريع التجارية |
| 🆓 Llama 3B | ⚡ | ⭐⭐⭐ | 🆓 | 3 | ⚠️ للتجارب فقط |

---

## 📊 أسعار السوق السوري 2024–2025

| المدينة | فاخر ($/م²) | متوسط ($/م²) | اقتصادي ($/م²) | اتجاه |
|---------|------------|-------------|----------------|------|
| دمشق | 1,200 | 650 | 300 | ↑ +15% |
| اللاذقية | 1,050 | 560 | 260 | ↑ +18% |
| طرطوس | 950 | 520 | 240 | ↑ +12% |
| حلب | 900 | 480 | 220 | ↑ +22% |
| السويداء | 800 | 430 | 200 | ↑ +8% |
| حمص | 750 | 400 | 180 | ↑ +30% |
| ريف دمشق | 700 | 350 | 150 | ↑ +25% |

> 💡 **معدل الصرف:** 13,500–15,200 ل.س/$ (2024) · 14,500–15,000 (2025)
> ⚠️ جميع البيانات **اصطناعية 100%** — للبحث والتطوير فقط
""")

        gr.HTML("""<div style='text-align:center;padding:20px;color:#4a5568;font-size:0.82em;
            border-top:1px solid rgba(56,189,248,0.08);margin-top:20px;'>
            🏠 مولّد البيانات العقارية السورية — البيانات اصطناعية للأغراض البحثية فقط
        </div>""")

    return demo
