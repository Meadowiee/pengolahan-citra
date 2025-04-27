from flask import Flask, render_template

# Create a Flask instance
app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
