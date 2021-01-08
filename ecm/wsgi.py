__author__ = "Lukas Merkle"
__copyright__ = "Copyright 2020, 31.07.20"
__email__ = 'lukas.merkle@tum.de'


from rc_model_flask_entry import app


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)