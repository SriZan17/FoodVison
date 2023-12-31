import gradio as gr


def greet(name):
    print("Hello " + name + "!")
    return "Hello " + name + "!"


demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch(share=True)
