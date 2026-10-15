"""Gradio web app: enter text → dominant emotion + confidence bar chart over the 6 classes.

Usage:  python app.py      (then open http://127.0.0.1:7860)
"""
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C
from src.predict import RNNPredictor, TransformerPredictor

EMOJI = {"joy": "😄", "sadness": "😢", "anger": "😠", "fear": "😨", "surprise": "😲", "disgust": "🤢"}
COLORS = {"joy": "#F2B701", "sadness": "#3969AC", "anger": "#E73F74", "fear": "#7F3C8D",
          "surprise": "#11A579", "disgust": "#80BA5A"}

PREDICTORS = {}
if (C.MODELS_DIR / "distilbert" / "config.json").exists():
    PREDICTORS["DistilBERT (fine-tuned)"] = TransformerPredictor()
if (C.MODELS_DIR / "bilstm_attention.pt").exists():
    PREDICTORS["BiLSTM + Attention"] = RNNPredictor("bilstm_attention")
if not PREDICTORS:
    raise SystemExit("No trained model found in models/. Run the training scripts or "
                     "`python scripts/download_models.py` first.")
ATTN = RNNPredictor("bilstm_attention") if (C.MODELS_DIR / "bilstm_attention.pt").exists() else None


def bar_chart(probs):
    order = probs.argsort()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    labels = [C.EMOTIONS[i] for i in order]   # matplotlib's default font has no emoji glyphs
    bars = ax.barh(labels, probs[order], color=[COLORS[C.EMOTIONS[i]] for i in order])
    ax.bar_label(bars, labels=[f"{p:.1%}" for p in probs[order]], padding=3, fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("confidence")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def classify(text, model_name):
    if not text or not text.strip():
        return "Please enter some text.", None, None, []
    pred = PREDICTORS[model_name]
    probs = pred.predict([text])[0] if isinstance(pred, TransformerPredictor) else pred.predict([text])[0][0]
    top = C.EMOTIONS[probs.argmax()]
    headline = f"## {EMOJI[top]} {top.capitalize()}  —  {probs.max():.1%} confidence"
    highlights = []
    if ATTN is not None:
        _, toks, attn = ATTN.predict([text])
        a = attn[0] / max(attn[0].max(), 1e-9)
        highlights = [(t + " ", float(w)) for t, w in zip(toks[0], a)]
    return headline, {e: float(p) for e, p in zip(C.EMOTIONS, probs)}, bar_chart(probs), highlights


EXAMPLES = [
    ["Thank you so much, this honestly made my whole week!"],
    ["I miss my grandma so much, it still hurts."],
    ["Stop lying to me. I'm sick of your excuses."],
    ["I'm terrified that I'll fail the exam tomorrow."],
    ["Wait, you're telling me he was the killer all along?!"],
    ["Ew, that is absolutely revolting. Who would eat that?"],
]

with gr.Blocks(title="Emotion Classifier") as demo:
    gr.Markdown("# Emotion Classification with Attention\n"
                "Type any sentence. The model predicts one of **joy, sadness, anger, fear, surprise, "
                "disgust** (trained on Google's GoEmotions mapped to Ekman's 6 emotions).")
    with gr.Row():
        with gr.Column(scale=1):
            text = gr.Textbox(label="Your text", lines=4, placeholder="How are you feeling?")
            model = gr.Radio(list(PREDICTORS), value=list(PREDICTORS)[0], label="Model")
            btn = gr.Button("Classify", variant="primary")
            gr.Examples(EXAMPLES, inputs=text)
        with gr.Column(scale=1):
            headline = gr.Markdown()
            plot = gr.Plot(label="Confidence across the 6 emotions")
            label = gr.Label(num_top_classes=6, label="Probabilities")
    attn = gr.HighlightedText(label="BiLSTM attention — which words drive the prediction",
                              combine_adjacent=False, show_legend=False)
    outputs = [headline, label, plot, attn]
    btn.click(classify, [text, model], outputs)
    text.submit(classify, [text, model], outputs)

if __name__ == "__main__":
    demo.launch()
