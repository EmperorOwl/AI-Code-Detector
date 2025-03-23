from flask import Flask, request, render_template

from model import Model

app = Flask(__name__)
model = Model()


@app.route('/', methods=['GET', 'POST'])
def classify_code():
    result = None
    if request.method == 'POST':
        code_snippet = request.form['code_snippet']
        label = model.classify_code(code_snippet)
        result = 'AI' if label == 0 else 'Human'
    return render_template('index.html', result=result)
