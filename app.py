from flask import Flask
from component.create import create_bp
from component.query import query_bp

app = Flask(__name__)

# Blueprintを登録
app.register_blueprint(create_bp)
app.register_blueprint(query_bp)

if __name__ == '__main__':
    app.run()