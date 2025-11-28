from celery import Celery

app = Celery("teledash-processing-asr")
app.config_from_object("config")

if __name__ == "__main__":
    app.start()
