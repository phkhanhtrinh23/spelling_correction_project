from flask import Flask, render_template, request, url_for, redirect
from base import SpellingCorrection

app = Flask(__name__)
obj = SpellingCorrection()

# Example
messages = [
        {
            'text': 'This is the text.',
            'answer': 'This is the answer.',
        },
            ]

@app.route('/')
def index():
    return render_template('index.html', messages=messages)

@app.route('/create/', methods=['GET', 'POST'])
def create():
    if request.method == 'POST':
        text = request.form['text']
        answer = []
        for sentence in text.split("."):
            sentence = sentence.strip()
            if len(sentence) > 1:
                answer.append(obj(sentence))
        
        answer = (". ").join(answer)

        if answer is None:
            return render_template('create.html')
        else:
            messages.append({'text': text, 'answer': answer})
            return redirect(url_for('index'))

    return render_template('create.html')
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8016)