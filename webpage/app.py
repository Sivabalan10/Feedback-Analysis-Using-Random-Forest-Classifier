# Web page integration code
from flask import Flask, render_template, jsonify, redirect, send_file

app = Flask(__name__)

@app.route('/')
def homepage():
    return "Homepage"

@app.route('/result')
def result():
    return "result"

if __name__ == "__main__":
    app.run(debug = True)