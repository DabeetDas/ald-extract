from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    conversation: list[ChatTurn] = Field(default_factory=list)
    scope_paper_id: str | None = None


class SourceChunk(BaseModel):
    source_id: str
    source_type: Literal["rag", "wikipedia"] = "rag"
    title: str | None = None
    url: str | None = None
    paper_id: str | None = None
    target_material: str | None = None
    process_type: str | None = None
    excerpt: str
    retrieval_score: float | None = None
    rerank_score: float | None = None


class RetrievalDiagnostics(BaseModel):
    scope: str
    hyde_enabled: bool
    retrieved_count: int
    reranked_count: int
    hyde_preview: str | None = None
    planner_used: bool = False
    executed_steps: int = 0
    tool_calls: int = 0
    validation_status: Literal["pass", "warning", "fail"] = "warning"
    validation_queries: list[str] = Field(default_factory=list)


class PlanStep(BaseModel):
    step_id: str = Field(min_length=1, max_length=20)
    step_type: Literal["analysis", "tool"]
    title: str = Field(min_length=1, max_length=160)
    objective: str = Field(min_length=1, max_length=600)
    depends_on: list[str] = Field(default_factory=list)
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    expected_output: str = Field(min_length=1, max_length=400)


class AgentPlan(BaseModel):
    planner_summary: str = Field(min_length=1, max_length=1200)
    analysis: str = Field(min_length=1, max_length=2000)
    synthesis_goal: str = Field(min_length=1, max_length=800)
    validation_focus: list[str] = Field(default_factory=list)
    steps: list[PlanStep] = Field(default_factory=list)


class ExecutionArtifact(BaseModel):
    step_id: str
    step_type: Literal["analysis", "tool"]
    title: str
    status: Literal["completed", "failed", "skipped"]
    tool_name: str | None = None
    resolved_arguments: dict[str, Any] = Field(default_factory=dict)
    output_summary: str
    raw_output: Any = None


class ValidationReport(BaseModel):
    factual_grounding: Literal["pass", "warning", "fail"]
    logical_consistency: Literal["pass", "warning", "fail"]
    cross_verification: Literal["pass", "warning", "fail"]
    issues: list[str] = Field(default_factory=list)
    summary: str
    verdict: Literal["pass", "warning", "fail"]
    revised_answer: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    diagnostics: RetrievalDiagnostics
    plan: AgentPlan | None = None
    execution: list[ExecutionArtifact] = Field(default_factory=list)
    validation: ValidationReport | None = None
