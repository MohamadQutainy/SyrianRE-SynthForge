# ═══════════════════════════════════════════════════════════════
# models_logic.py — منطق النماذج والتوليد
# ═══════════════════════════════════════════════════════════════

import re, json, random, time
import pandas as pd
from tqdm.auto import tqdm

from config import (
    CITIES, FEATURES_POOL, AGENCIES, CONDITIONS, AGENT_NAMES,
    MODELS, OPENAI_API_KEY, ANTHROPIC_API_KEY, HF_TOKEN,
)

# ── تهيئة العملاء ──
openai_client  = None
claude_client  = None
_llama_model   = None
_llama_tok     = None

try:
    from openai import OpenAI
    if OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI متصل")
    else:
        print("⚠️  OPENAI_API_KEY غير موجود في .env")
except ImportError:
    print("⚠️  مكتبة openai غير مثبّتة")

try:
    import anthropic
    if ANTHROPIC_API_KEY:
        claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        print("✅ Anthropic متصل")
    else:
        print("⚠️  ANTHROPIC_API_KEY غير موجود في .env")
except ImportError:
    print("⚠️  مكتبة anthropic غير مثبّتة")


# ═══ System Prompt ═══
SYSTEM_PROMPT = """أنت خبير عقاري سوري متخصص في سوق الشقق والمنازل خلال 2024–2025.
مهمتك توليد بيانات اصطناعية واقعية ودقيقة للغاية.

قواعد صارمة:
• الأسعار منطقية ومتوافقة مع الواقع السوري — لا مبالغة ولا تهوين
• معدل الصرف 2024: 13,500–15,200 ليرة/دولار | 2025: 14,500–15,000
• الأحياء الراقية أسعارها أعلى 2–3 أضعاف من الشعبية
• الأوصاف التسويقية باللغة العربية الفصيحة الطبيعية
• أرجع JSON array نظيفاً فقط — لا نص قبله ولا بعده"""


# ═══ بناء الـ Prompt ═══
def make_prompt(city, neighborhood, neighborhood_class, prop_type,
                listing_type, count, date_range):
    city_data = CITIES[city]
    price_key = {
        "راقي":   "سعر_فاخر_usd",
        "متوسط":  "سعر_متوسط_usd",
        "شعبي":   "سعر_اقتصادي_usd",
    }
    ref_price = city_data[price_key.get(neighborhood_class, "سعر_متوسط_usd")]

    if prop_type == "شقة":
        area_hint  = "50–250 م² للشقق"
        rooms_hint = "1–5 غرف نوم"
    else:
        area_hint  = "150–500 م² للمنازل والفيلات"
        rooms_hint = "3–7 غرف نوم"

    features_sample = random.sample(FEATURES_POOL, 8)
    agencies_sample = random.sample(AGENCIES, 5)

    return f"""ولّد {count} إعلان عقاري اصطناعي واقعي للسوق السوري 2024–2025.

المواصفات:
• المدينة: {city} | الحي: {neighborhood} ({neighborhood_class})
• نوع العقار: {prop_type} | نوع الإعلان: {listing_type}
• المساحات: {area_hint} | {rooms_hint}
• السعر المرجعي: ~{ref_price}$/م² لهذا المستوى في {city}
• اتجاه السوق: {city_data['اتجاه_2024_2025']} | {city_data['ملاحظة_سوق']}
• تاريخ الإدراج: بين {date_range[0]} و {date_range[1]}

لكل عقار أنشئ هذا الـ JSON بالضبط:
{{
  "رقم_الإعلان": "SY-XXXX-XXXXX",
  "نوع_الإعلان": "{listing_type}",
  "نوع_العقار": "{prop_type}",
  "المدينة": "{city}",
  "الحي": "{neighborhood}",
  "تفاصيل_الموقع": "وصف دقيق: قرب/شارع/طريق ...",
  "المساحة_م2": 120,
  "غرف_النوم": 3,
  "الحمامات": 2,
  "الطابق": 4,
  "عدد_الطوابق_الكلي": 8,
  "السعر_دولار": 85000,
  "السعر_ليرة": 1275000000,
  "سعر_المتر_دولار": 708,
  "نوع_السعر": "ثابت",
  "تاريخ_الإدراج": "2024-06-15",
  "سنة_البناء": 2018,
  "سنة_التجديد": null,
  "الحالة": "اختر من: {', '.join(CONDITIONS)}",
  "المميزات": ["اختر 3-6 من: {', '.join(features_sample)}"],
  "الوصف_التسويقي": "وصف جذاب 2–3 جمل بالعربية الفصيحة",
  "اسم_الوسيط": "اختر من: {', '.join(random.sample(AGENT_NAMES, 4))}",
  "مكتب_العقارات": "اختر من: {', '.join(agencies_sample)}",
  "رقم_الهاتف": "+963-944-123456",
  "مشاهدات_الإعلان": 1250,
  "موثق": true,
  "ملاحظة_السوق": "ملاحظة مختصرة عن وضع السوق"
}}

⚠️ أرجع JSON array فقط بدون أي نص إضافي:
[{{...}}, {{...}}]"""


# ═══ safe_parse — تحليل JSON آمن ═══
def safe_parse(text: str) -> list:
    original = text
    text = text.strip()

    # إزالة markdown code blocks
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = text.strip('`').strip()

    # استخراج JSON array
    s, e = text.find('['), text.rfind(']') + 1
    if s != -1 and e > s:
        text = text[s:e]

    def _variants(t):
        yield t
        t1 = re.sub(r',\s*([}\]])', r'\1', t)
        yield t1
        t2 = re.sub(r':\s*رقم أو null\b', ': null', t)
        t2 = re.sub(r':\s*true أو false\b', ': true', t2)
        t2 = re.sub(r':\s*رقم\b[^,}\]\n"]{0,60}', ': 0', t2)
        yield t2
        yield re.sub(r',\s*([}\]])', r'\1', t2)

    for v in _variants(text):
        try:
            return json.loads(v)
        except Exception:
            pass

    # fallback: كائنات منفردة
    objects = re.findall(r'\{(?:[^{}]|\{[^{}]*\})*\}', text, re.DOTALL)
    results = []
    for obj in objects:
        for v in _variants(obj):
            try:
                results.append(json.loads(v))
                break
            except Exception:
                pass
    if results:
        return results

    print(f"  [DEBUG] فشل التحليل:\n{original[:400]}\n{'─'*40}")
    raise ValueError('فشل في تحليل JSON')


# ═══ _load_llama ═══
def _load_llama():
    global _llama_model, _llama_tok
    if _llama_model is not None:
        return
    print('⏳ تحميل Llama (3–5 دقائق)...')
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from huggingface_hub import login
        if HF_TOKEN:
            login(HF_TOKEN)
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type='nf4',
        )
        mid = MODELS['llama-3b']['id']
        _llama_tok   = AutoTokenizer.from_pretrained(mid)
        _llama_tok.pad_token = _llama_tok.eos_token
        _llama_model = AutoModelForCausalLM.from_pretrained(
            mid, device_map='auto', quantization_config=qcfg)
        print('✅ Llama جاهز!')
    except Exception as e:
        raise RuntimeError(f"فشل تحميل Llama: {e}")


# ═══ call_model ═══
def call_model(model_key: str, system: str, user: str) -> str:
    m = MODELS[model_key]

    if m['type'] == 'openai':
        if not openai_client:
            raise ValueError('OpenAI غير متصل — تحقق من OPENAI_API_KEY في .env')
        r = openai_client.chat.completions.create(
            model=m['id'],
            messages=[{'role': 'system', 'content': system},
                      {'role': 'user',   'content': user}],
            max_tokens=m['max_tokens'], temperature=0.85,
        )
        return r.choices[0].message.content

    elif m['type'] == 'anthropic':
        if not claude_client:
            raise ValueError('Anthropic غير متصل — تحقق من ANTHROPIC_API_KEY في .env')
        r = claude_client.messages.create(
            model=m['id'], max_tokens=m['max_tokens'],
            system=system,
            messages=[{'role': 'user', 'content': user}],
        )
        return r.content[0].text

    elif m['type'] == 'local':
        import torch
        _load_llama()
        msgs = [{'role': 'system', 'content': system},
                {'role': 'user',   'content': user}]
        ids = _llama_tok.apply_chat_template(
            msgs, return_tensors='pt', add_generation_prompt=True
        ).to('cuda')
        with torch.no_grad():
            out = _llama_model.generate(
                ids, max_new_tokens=m['max_tokens'],
                temperature=0.85, do_sample=True,
            )
        return _llama_tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    raise ValueError(f"نوع نموذج غير معروف: {m['type']}")


# ═══ generate_batch ═══
def generate_batch(model_key, city, prop_type, listing_type, batch_size, date_range,
                   tier_weights=None, extra_instructions=""):
    city_data = CITIES[city]
    w = tier_weights or [0.20, 0.50, 0.30]
    tier     = random.choices(['راقي', 'متوسط', 'شعبي'], weights=w)[0]
    tier_key = {'راقي': 'أحياء_راقية', 'متوسط': 'أحياء_متوسطة', 'شعبي': 'أحياء_شعبية'}[tier]
    neighborhood = random.choice(city_data[tier_key])

    prompt = make_prompt(city, neighborhood, tier, prop_type, listing_type,
                         batch_size, date_range)
    if extra_instructions.strip():
        prompt += f"\n\nتعليمات إضافية:\n{extra_instructions}"

    raw = call_model(model_key, SYSTEM_PROMPT, prompt)
    return safe_parse(raw)


# ═══ bulk_generate ═══
def bulk_generate(
    model_key="gpt-4o-mini",
    total=300,
    cities=None,
    prop_types=None,
    listing_types=None,
    date_start="2024-01-01",
    date_end="2025-12-31",
    output_name="عقارات_سوريا",
    retry_on_fail=True,
    tier_weights=None,
    extra_instructions="",
):
    """توليد ضخم بتوزيع متوازن عبر المدن وأنواع العقارات."""
    target_cities   = cities       or list(CITIES.keys())
    target_ptypes   = prop_types   or ['شقة', 'منزل']
    target_listings = listing_types or ['بيع', 'إيجار']
    batch_sz  = MODELS[model_key]['batch_size']
    date_range = (date_start, date_end)
    w = tier_weights or [0.20, 0.50, 0.30]

    all_records, errors, generated = [], 0, 0

    print('═' * 60)
    print(f"🏭 بدء التوليد | {MODELS[model_key]['display_name']}")
    print(f"📊 الهدف: {total} سجل | Batch: {batch_sz}")
    print(f"🏙️  المدن: {target_cities}")
    print('═' * 60)

    pbar = tqdm(total=total, desc='⚡ التوليد', unit='سجل')

    while generated < total:
        batch   = min(batch_sz, total - generated)
        city    = random.choice(target_cities)
        ptype   = random.choices(target_ptypes,   weights=([0.75, 0.25] if len(target_ptypes)   == 2 else [1])[:len(target_ptypes)])[0]
        listing = random.choices(target_listings, weights=([0.60, 0.40] if len(target_listings) == 2 else [1])[:len(target_listings)])[0]

        try:
            recs = generate_batch(model_key, city, ptype, listing, batch,
                                  date_range, w, extra_instructions)
            all_records.extend(recs)
            generated += len(recs)
            pbar.update(len(recs))
            pbar.set_postfix({'مدينة': city, 'نوع': ptype, 'إجمالي': generated})

            if generated % 50 < batch_sz and generated >= 50:
                pd.DataFrame(all_records).to_csv(
                    f'temp_{output_name}.csv', index=False, encoding='utf-8-sig')

        except Exception as e:
            errors += 1
            tqdm.write(f'  ⚠️  خطأ [{city}/{ptype}]: {str(e)[:60]}')
            if not retry_on_fail or errors > 10:
                tqdm.write('❌ تجاوز الحد الأقصى للأخطاء. توقف.')
                break
            time.sleep(2)

    pbar.close()

    if not all_records:
        print('❌ لم يُولَّد أي سجل!')
        return None

    df = pd.DataFrame(all_records)
    for col in ['السعر_دولار', 'المساحة_م2']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    csv_path  = f'{output_name}.csv'
    json_path = f'{output_name}.json'
    xlsx_path = f'{output_name}.xlsx'

    df.to_csv(csv_path,   index=False, encoding='utf-8-sig')
    df.to_json(json_path, orient='records', force_ascii=False, indent=2)
    df.to_excel(xlsx_path, index=False)

    print('\n' + '═' * 60)
    print(f'🎉 اكتمل التوليد! ✅ {len(df)} سجل | ❌ {errors} أخطاء')
    print(f'💾 {csv_path} | {json_path} | {xlsx_path}')
    print('═' * 60)

    if 'السعر_دولار' in df.columns:
        print('\n📊 إحصائيات الأسعار (دولار):')
        print(df['السعر_دولار'].describe().apply(lambda x: f'{x:,.0f}'))
    if 'المدينة' in df.columns:
        print('\n🏙️  توزيع المدن:')
        print(df['المدينة'].value_counts())

    return df
