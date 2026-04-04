from datetime import datetime

from pydantic import BaseModel


class DashboardRecentActivityItem(BaseModel):
    class_id: int
    student_id: int
    student_name: str
    emotion: str
    time_ago: str
    saved_at: datetime


class DashboardOverviewResponse(BaseModel):
    total_students: int
    total_analyses: int
    active_classes: int
    analyses_this_week: int
    recent_activity: list[DashboardRecentActivityItem]
