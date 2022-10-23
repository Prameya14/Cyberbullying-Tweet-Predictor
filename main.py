from flask import Flask, redirect, render_template, request
from functions import *

app = Flask(__name__, template_folder="Templates")


@app.route('/', methods=['GET', 'POST'])
def interpret():
    if request.method == 'POST':
      tweet = request.form['tweet']
      if tweet != "":
        category = custom_input_prediction(tweet)
        return render_template('index.html', text=category)
      else:
        return redirect('/')
    return render_template('index.html', text="")


if __name__ == "__main__":
    app.run(debug=True)
