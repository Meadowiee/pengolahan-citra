from flask import Flask
import os
from app.controllers.main import main_bp

def create_app():
    app = Flask(__name__)
    app.config.from_object('config.Config')

    app.register_blueprint(main_bp)
    return app