from flask import Flask, render_template, request
import pandas as pd
import module as md

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/cluster", methods=["POST"])
def cluster():
    if request.method == "POST":
        df1 = pd.read_excel(request.files["file1"])
        df2 = pd.read_excel(request.files["file2"])
        df3 = pd.read_excel(request.files["file3"])
        result = md.preprocess_df(df1, df2, df3)
        print(result)
        viz = md.visualization()
        return render_template("cluster.html", viz=viz)
    else:
        return


if __name__ == "__main__":
    app.run()
