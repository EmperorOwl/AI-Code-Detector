from flask import Flask, request, render_template

from src.models.codebert_model import CodeBertModel


app = Flask(__name__)
model = CodeBertModel(use_saved=True)


@app.route('/', methods=['GET', 'POST'])
def classify_code():
    result = None
    code_snippet = ''
    if request.method == 'POST':
        code_snippet = request.form['code_snippet']
        ai_probability = model.classify_code(code_snippet)
        result = f'{ai_probability:.2f}% AI-written'
    return render_template('index.html',
                           code_snippet=code_snippet,
                           result=result)


if __name__ == '__main__':
    app.run(debug=True)
