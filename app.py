# ═══════════════════════════════════════════════════════════════
# app.py — نقطة الدخول الرئيسية
# ═══════════════════════════════════════════════════════════════
# تشغيل:  python app.py

from gradio_ui import build_demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        share=True,       # رابط مؤقت عام عبر ngrok
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
