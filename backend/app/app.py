from fastapi import FastAPI, File, Form, UploadFile

app = FastAPI()

@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    caption: str = Form(""),
):
    return {"message": "Upload endpoint working"}

@app.get("/get_test")
async def get_test():
    return {"message": "Get endpoint working"}