from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
from pin_detection.router import router as pin_detection_router
import plotly.io as pio

app = FastAPI()

pio.templates.default = "plotly_white"

app.mount("/styles", StaticFiles(directory="styles"), name="styles")

app.include_router(pin_detection_router)


@app.get("/")
async def root():
    response = RedirectResponse(url="/docs")
    return response
