from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from app.routers import image
from app.routers import job
from app.routers import class_router
from app.routers import student
from app.routers import auth
from app.routers import dashboard
from app.core.config import settings

from app.database.database import engine, Base
import app.models.class_model
import app.models.student
import app.models.student_saved_analysis
import app.models.user


app = FastAPI(
    title="INKIND API",
    description="API for INKIND project",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


def _ensure_user_username_column() -> None:
    # Keep existing SQLite databases compatible by adding the new column if missing.
    with engine.begin() as connection:
        row = connection.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        ).fetchone()
        if not row:
            return

        columns = connection.execute(text("PRAGMA table_info(users)")).fetchall()
        column_names = {column[1] for column in columns}
        if "username" not in column_names:
            connection.execute(text("ALTER TABLE users ADD COLUMN username VARCHAR"))
            connection.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_username ON users (username)"))

# Create Database Tables
Base.metadata.create_all(bind=engine)
_ensure_user_username_column()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(image.router)
app.include_router(job.router)
app.include_router(class_router.router)
app.include_router(student.router)
app.include_router(auth.router)
app.include_router(dashboard.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

