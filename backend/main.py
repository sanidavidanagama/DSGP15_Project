
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import image
from app.routers import job
from app.core.config import settings


app = FastAPI(
    title="INKIND API",
    description="API for INKIND project - DSGP group 15 of Informatics Institute of Technology affiliated with Robert Gordon University, Abeerdeen",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# Use settings for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.parse_allowed_origins(settings.ALLOWED_ORIGINS),
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


app.include_router(image.router)
app.include_router(job.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

