from typing import Any, Dict

import gradio as gr
from fastai.vision.all import *

learn_inf = load_learner('export.pkl')

categories = learn_inf.dls.vocab

def classify_digit(digit_img: Any) -> Dict[str, float]:
    _, _, probs = learn_inf.predict(digit_img)

    return dict(zip(categories, map(float, probs)))

image = gr.Image(width=192, height=192)
label = gr.Label()
examples = [f"{digit}.jpeg" for digit in learn_inf.dls.vocab]
intf = gr.Interface(fn=classify_digit, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)