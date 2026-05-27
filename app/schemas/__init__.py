from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone

class DataSourceBase(BaseModel):
    name: str
    type: str

class DataSourceCreate(BaseModel):
    name: str
    type: str
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    connection_string: Optional[str] = None
    container_name: Optional[str] = None
    ssl_mode: Optional[str] = None  # disable, allow, prefer, require, verify-ca, verify-full

class DataSourceTest(BaseModel):
    type: str
    connection_string: Optional[str] = None
    container_name: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    ssl_mode: Optional[str] = None

class DataSourceResponse(BaseModel):
    id: int
    name: str
    type: str
    status: str
    lastSync: Optional[str] = None
    owner: str
    created_at: datetime

    class Config:
        from_attributes = True

    @classmethod
    def from_orm(cls, obj):
        created = obj.created_at
        if created and created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return cls(
            id=obj.id,
            name=obj.name,
            type=obj.type,
            status=obj.status,
            lastSync=None,
            owner=obj.owner,
            created_at=created,
        )

class DatasetBase(BaseModel):
    datasource_id: int
    physical_name: str
    display_name: str

class DatasetResponse(BaseModel):
    id: int
    name: str  # display_name
    sourceId: int  # datasource_id
    path: str  # physical_name
    createdAt: datetime  # created_at

    class Config:
        from_attributes = True

    @classmethod
    def from_orm(cls, obj):
        created = obj.created_at
        if created and created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return cls(
            id=obj.id,
            name=obj.display_name,
            sourceId=obj.datasource_id,
            path=obj.physical_name,
            createdAt=created
        )

class GlobalContextBase(BaseModel):
    active_datasource_id: Optional[int] = None
    active_dataset_id: Optional[int] = None

class GlobalContextResponse(BaseModel):
    active_datasource_id: Optional[int] = None
    active_dataset_id: Optional[int] = None

    class Config:
        from_attributes = True