from pydantic import BaseModel, Field, field_validator
from typing import List

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1) # Ensure query is non-empty
    top_k: int = Field(default=3, gt=0) # Ensure top_k is positive, default to 3

    @field_validator('query')
    def query_must_not_be_whitespace(cls, value):
        if not value.strip():
            raise ValueError("Query must not be empty or only whitespace")
        return value

class ContextResponse(BaseModel):
    context: List[str]
    # Add metadata/distances later if needed
    # metadata: List[dict] = None
    # distances: List[float] = None