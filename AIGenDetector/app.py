from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

from models import get_ai_probability

app = FastAPI()


templates = Jinja2Templates(directory="templates")


app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.post("/process_text/")
async def upload_image(text_input: str = Form(), classification_model: str = Form()):

    if len(text_input) > 512:
        return templates.TemplateResponse("incorrect_input.html", {"request": {}})

    else:
        predictions_results = get_ai_probability(text_input, classification_model)

        return templates.TemplateResponse("result.html", {"request":{}, "predictions_results": predictions_results})


@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request":request})