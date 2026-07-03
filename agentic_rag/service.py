from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request as urllib_request

from agentic_rag_pipeline.llm import GeminiLLM, LLMUnavailableError
from agentic_rag_pipeline.schemas import (
    AgentPlan,
    ChatRequest,
    ChatResponse,
    ExecutionArtifact,
    PlanStep,
    RetrievalDiagnostics,
    SourceChunk,
    ValidationReport,
)
from agentic_rag_pipeline.settings import Settings


class ConfigurationError(RuntimeError):
    """Raised when a required backend dependency or env var is missing."""


@dataclass
class RetrievedChunk:
    source_id: str
    text: str
    paper_id: str | None
    target_material: str | None
    process_type: str | None
    original_score: float | None = None
    hyde_score: float | None = None
    fusion_score: float = 0.0
    rerank_score: float | None = None


@dataclass
class ToolExecutionResult:
    tool_name: str
    summary: str
    payload: dict[str, Any]
    sources: list[SourceChunk]


def _safe_attr(item, name: str, default=None):
    if hasattr(item, name):
        return getattr(item, name)
    if isinstance(item, dict):
        return item.get(name, default)
    return default


class AgenticRAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._pc = None
        self._index = None
        self._embedder = None
        self._llm: GeminiLLM | None = None
        self._llm_error: str | None = None
        self._tools = {
            "rag_search": self._tool_rag_search,
        }
        if self.settings.wikipedia_enabled:
            self._tools["wikipedia_lookup"] = self._tool_wikipedia_lookup

    def health(self) -> dict:
        warnings: list[str] = []

        if not self.settings.pinecone_api_key:
            warnings.append("PINECONE_API_KEY is missing.")
        if not self.settings.gemini_api_key:
            warnings.append("GEMINI_API_KEY or GOOGLE_API_KEY is missing; planner, synthesis, and validation will fall back.")

        llm_enabled = self.llm_available
        if self._llm_error:
            warnings.append(self._llm_error)

        return {
            "status": "ok" if not warnings else "degraded",
            "index_name": self.settings.pinecone_index_name,
            "scope": "paper-aware",
            "mode": "agentic",
            "hyde_configured": self.settings.hyde_enabled and llm_enabled,
            "llm_enabled": llm_enabled,
            "validation_enabled": self.settings.validation_enabled,
            "available_tools": sorted(self._tools),
            "warnings": warnings,
        }

    @property
    def llm_available(self) -> bool:
        if self._llm is not None:
            return True
        if self._llm_error:
            return False
        try:
            self._llm = GeminiLLM(self.settings)
            return True
        except LLMUnavailableError as exc:
            self._llm_error = str(exc)
            return False

    def _get_llm(self) -> GeminiLLM:
        if not self.llm_available or self._llm is None:
            raise LLMUnavailableError(self._llm_error or "LLM is unavailable.")
        return self._llm

    def _get_pinecone(self):
        if self._pc is not None:
            return self._pc

        if not self.settings.pinecone_api_key:
            raise ConfigurationError("PINECONE_API_KEY is required for retrieval.")

        try:
            from pinecone import Pinecone
        except Exception as exc:
            raise ConfigurationError(
                "Pinecone SDK is unavailable. Install `pinecone` and remove the deprecated "
                "`pinecone-client` package if it is still present."
            ) from exc

        self._pc = Pinecone(api_key=self.settings.pinecone_api_key)
        return self._pc

    def _get_index(self):
        if self._index is not None:
            return self._index

        pc = self._get_pinecone()
        if self.settings.pinecone_index_host:
            self._index = pc.Index(host=self.settings.pinecone_index_host)
        else:
            self._index = pc.Index(self.settings.pinecone_index_name)
        return self._index

    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise ConfigurationError(
                "Local embedding requires `sentence-transformers`. Install it with "
                "`pip install sentence-transformers` or update agentic_rag_pipeline/requirements.txt."
            ) from exc

        self._embedder = SentenceTransformer(self.settings.local_embed_model)
        return self._embedder

    def _embed(self, text: str, input_type: str) -> list[float]:
        embedder = self._get_embedder()
        embed_text = text
        if input_type == "query" and self.settings.local_embed_query_instruction:
            embed_text = f"{self.settings.local_embed_query_instruction}{text}"

        values = embedder.encode(
            embed_text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return values.astype(float).tolist()

    def _query_index(
        self,
        vector: list[float],
        top_k: int,
        scope_paper_id: str | None,
        source: str,
    ) -> list[RetrievedChunk]:
        index = self._get_index()
        metadata_filter = {"paper_id": {"$eq": scope_paper_id}} if scope_paper_id else None

        result = index.query(
            vector=vector,
            top_k=top_k,
            namespace=self.settings.pinecone_namespace,
            include_values=False,
            include_metadata=True,
            filter=metadata_filter,
        )

        matches = _safe_attr(result, "matches", []) or []
        chunks: list[RetrievedChunk] = []
        for match in matches:
            metadata = _safe_attr(match, "metadata", {}) or {}
            text = metadata.get("text", "").strip()
            if not text:
                continue
            chunk = RetrievedChunk(
                source_id=str(_safe_attr(match, "id", "unknown")),
                text=text,
                paper_id=metadata.get("paper_id"),
                target_material=metadata.get("target_material"),
                process_type=metadata.get("process_type"),
            )
            score = float(_safe_attr(match, "score", 0.0) or 0.0)
            if source == "original":
                chunk.original_score = score
            else:
                chunk.hyde_score = score
            chunks.append(chunk)
        return chunks

    def _fuse_results(
        self,
        original_chunks: list[RetrievedChunk],
        hyde_chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        combined: dict[str, RetrievedChunk] = {}

        def merge(chunks: list[RetrievedChunk], source: str) -> None:
            for rank, chunk in enumerate(chunks, start=1):
                existing = combined.get(chunk.source_id)
                rrf_bonus = 1.0 / (self.settings.rrf_k + rank)
                if existing is None:
                    chunk.fusion_score = rrf_bonus
                    combined[chunk.source_id] = chunk
                    continue

                existing.fusion_score += rrf_bonus
                if source == "original" and chunk.original_score is not None:
                    existing.original_score = chunk.original_score
                if source == "hyde" and chunk.hyde_score is not None:
                    existing.hyde_score = chunk.hyde_score

        merge(original_chunks, "original")
        merge(hyde_chunks, "hyde")

        ranked = sorted(
            combined.values(),
            key=lambda item: (
                item.fusion_score,
                item.original_score or 0.0,
                item.hyde_score or 0.0,
            ),
            reverse=True,
        )
        return ranked[: self.settings.rerank_input_k]

    def _rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return []

        pc = self._get_pinecone()
        documents = [
            {
                "id": chunk.source_id,
                "text": chunk.text,
                "paper_id": chunk.paper_id or "unknown",
                "target_material": chunk.target_material or "Unknown",
                "process_type": chunk.process_type or "Unknown",
            }
            for chunk in candidates
        ]

        reranked = pc.inference.rerank(
            model=self.settings.pinecone_rerank_model,
            query=query,
            documents=documents,
            top_n=min(self.settings.final_top_k, len(documents)),
            return_documents=True,
            parameters={"truncate": "END"},
        )

        index_lookup = {chunk.source_id: chunk for chunk in candidates}
        ordered: list[RetrievedChunk] = []

        for item in _safe_attr(reranked, "data", []) or []:
            document = _safe_attr(item, "document", {}) or {}
            source_id = document.get("id") or candidates[_safe_attr(item, "index", 0)].source_id
            chunk = index_lookup.get(source_id)
            if chunk is None:
                continue
            chunk.rerank_score = float(_safe_attr(item, "score", 0.0) or 0.0)
            ordered.append(chunk)

        return ordered

    def _run_rag_retrieval(
        self,
        query: str,
        conversation,
        scope_paper_id: str | None,
    ) -> tuple[list[RetrievedChunk], str | None, RetrievalDiagnostics]:
        hyde_document: str | None = None
        hyde_chunks: list[RetrievedChunk] = []

        original_vector = self._embed(query, input_type="query")
        original_chunks = self._query_index(
            vector=original_vector,
            top_k=self.settings.initial_top_k,
            scope_paper_id=scope_paper_id,
            source="original",
        )

        if self.settings.hyde_enabled and self.llm_available:
            hyde_document = self._get_llm().generate_hyde_document(
                query=query,
                conversation=conversation,
                scope_paper_id=scope_paper_id,
            )
            if hyde_document:
                hyde_vector = self._embed(hyde_document, input_type="passage")
                hyde_chunks = self._query_index(
                    vector=hyde_vector,
                    top_k=self.settings.initial_top_k,
                    scope_paper_id=scope_paper_id,
                    source="hyde",
                )

        fused_chunks = self._fuse_results(original_chunks, hyde_chunks)
        reranked_chunks = self._rerank(query, fused_chunks)

        diagnostics = RetrievalDiagnostics(
            scope=scope_paper_id or "global-catalog",
            hyde_enabled=bool(hyde_document),
            retrieved_count=len(fused_chunks),
            reranked_count=len(reranked_chunks),
            hyde_preview=hyde_document[:220] if hyde_document else None,
        )
        return reranked_chunks, hyde_document, diagnostics

    def _tool_rag_search(
        self,
        query: str,
        conversation,
        scope_paper_id: str | None = None,
        query_style: str = "standard",
    ) -> ToolExecutionResult:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("rag_search requires a non-empty query.")

        chunks, _hyde_document, diagnostics = self._run_rag_retrieval(
            query=cleaned_query,
            conversation=conversation,
            scope_paper_id=scope_paper_id,
        )

        sources = [
            SourceChunk(
                source_id=chunk.source_id,
                source_type="rag",
                paper_id=chunk.paper_id,
                target_material=chunk.target_material,
                process_type=chunk.process_type,
                excerpt=chunk.text,
                retrieval_score=chunk.fusion_score,
                rerank_score=chunk.rerank_score,
            )
            for chunk in chunks
        ]

        summary_bits = [f"Retrieved {len(sources)} grounded ALD chunks for query `{cleaned_query}`."]
        if sources:
            top = sources[0]
            summary_bits.append(
                f"Top evidence came from {top.paper_id or 'an unscoped paper'}"
                f"{f' on {top.target_material}' if top.target_material else ''}."
            )
        evidence_summary = " ".join(summary_bits)

        payload = {
            "query": cleaned_query,
            "query_style": query_style,
            "scope": scope_paper_id or "global-catalog",
            "evidence_summary": evidence_summary,
            "retrieval_diagnostics": diagnostics.model_dump(),
            "source_count": len(sources),
            "sources": [source.model_dump() for source in sources],
        }

        return ToolExecutionResult(
            tool_name="rag_search",
            summary=evidence_summary,
            payload=payload,
            sources=sources,
        )

    def _fetch_json(self, url: str) -> Any:
        req = urllib_request.Request(
            url,
            headers={
                "User-Agent": "ALD-LLaMat-Agentic-RAG/1.0 (research assistant)",
                "Accept": "application/json",
            },
        )
        with urllib_request.urlopen(req, timeout=8) as response:
            return json.loads(response.read().decode("utf-8"))

    def _wikipedia_query_candidates(self, query: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", query).strip()
        if not normalized:
            return []

        candidates: list[str] = [normalized]
        tokens = [token for token in re.split(r"\s+", normalized) if token]
        if len(tokens) <= 1:
            return candidates

        seen = {normalized.lower()}
        for width in range(len(tokens) - 1, 0, -1):
            for start in range(0, len(tokens) - width + 1):
                candidate = " ".join(tokens[start : start + width]).strip()
                if len(candidate) < 3:
                    continue
                key = candidate.lower()
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
                if len(candidates) >= 20:
                    return candidates
        return candidates

    def _search_wikipedia_title(self, query: str) -> tuple[str | None, str | None]:
        for candidate in self._wikipedia_query_candidates(query):
            search_url = (
                "https://en.wikipedia.org/w/api.php?"
                + parse.urlencode(
                    {
                        "action": "opensearch",
                        "search": candidate,
                        "limit": 1,
                        "namespace": 0,
                        "format": "json",
                    }
                )
            )
            try:
                search_payload = self._fetch_json(search_url)
            except error.URLError as exc:
                raise RuntimeError(f"Wikipedia search failed: {exc}") from exc

            titles = search_payload[1] if isinstance(search_payload, list) and len(search_payload) > 1 else []
            if titles:
                return str(titles[0]), candidate

        return None, None

    def _tool_wikipedia_lookup(self, query: str) -> ToolExecutionResult:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("wikipedia_lookup requires a non-empty query.")

        title, matched_query = self._search_wikipedia_title(cleaned_query)
        if not title or not matched_query:
            raise RuntimeError(f"No Wikipedia page found for query: {cleaned_query}")

        summary_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + parse.quote(title, safe="")
        try:
            page_payload = self._fetch_json(summary_url)
        except error.URLError as exc:
            raise RuntimeError(f"Wikipedia summary fetch failed: {exc}") from exc

        extract = (page_payload.get("extract") or "").strip()
        if not extract:
            raise RuntimeError(f"Wikipedia page `{title}` did not include a summary extract.")

        url = (
            page_payload.get("content_urls", {})
            .get("desktop", {})
            .get("page")
            or f"https://en.wikipedia.org/wiki/{parse.quote(title.replace(' ', '_'))}"
        )
        source = SourceChunk(
            source_id=f"wiki:{title}",
            source_type="wikipedia",
            title=title,
            url=url,
            excerpt=extract,
        )

        payload = {
            "query": cleaned_query,
            "matched_query": matched_query,
            "title": title,
            "url": url,
            "evidence_summary": extract[:320],
            "sources": [source.model_dump()],
        }
        fallback_note = "" if matched_query == cleaned_query else f" via fallback query `{matched_query}`"
        return ToolExecutionResult(
            tool_name="wikipedia_lookup",
            summary=f"Wikipedia returned background context for `{title}`{fallback_note}.",
            payload=payload,
            sources=[source],
        )

    def _available_tools(self) -> list[dict[str, Any]]:
        tools = [
            {
                "name": "rag_search",
                "description": "Runs the existing HyDE + reciprocal-rank-fusion + Pinecone rerank ALD retrieval pipeline.",
                "arguments": {
                    "query": "str",
                    "scope_paper_id": "str | null",
                    "query_style": "str",
                },
            }
        ]
        if self.settings.wikipedia_enabled:
            tools.append(
                {
                    "name": "wikipedia_lookup",
                    "description": "Fetches a concise Wikipedia summary for broad background context.",
                    "arguments": {
                        "query": "str",
                    },
                }
            )
        return tools

    def _should_use_wikipedia(self, query: str, scope_paper_id: str | None) -> bool:
        if not self.settings.wikipedia_enabled:
            return False
        if scope_paper_id:
            return False

        lowered = query.lower()
        triggers = (
            "what is ",
            "who is ",
            "history of ",
            "define ",
            "overview of ",
            "background on ",
            "wikipedia",
        )
        return any(token in lowered for token in triggers)

    def _fallback_plan(self, request: ChatRequest) -> AgentPlan:
        steps: list[PlanStep] = [
            PlanStep(
                step_id="A1",
                step_type="analysis",
                title="Analyze query scope",
                objective=(
                    "Determine the ALD entities, process details, comparisons, and evidence needed to answer "
                    "the user query directly from grounded sources."
                ),
                expected_output="A concise evidence checklist for synthesis.",
            ),
            PlanStep(
                step_id="E1",
                step_type="tool",
                title="Retrieve corpus evidence",
                objective="Use the existing HyDE + rerank RAG pipeline to gather grounded ALD evidence.",
                tool_name="rag_search",
                arguments={
                    "query": request.query,
                    "scope_paper_id": request.scope_paper_id,
                    "query_style": "standard",
                },
                expected_output="Reranked ALD evidence chunks from Pinecone.",
            ),
        ]

        if self._should_use_wikipedia(request.query, request.scope_paper_id):
            steps.append(
                PlanStep(
                    step_id="E2",
                    step_type="tool",
                    title="Fetch background context",
                    objective="Use Wikipedia for broad background or definitional context.",
                    depends_on=["E1"],
                    tool_name="wikipedia_lookup",
                    arguments={"query": request.query},
                    expected_output="A concise encyclopedic background summary.",
                )
            )

        return AgentPlan(
            planner_summary="Fallback plan: analyze the request, retrieve corpus evidence, and optionally add Wikipedia background context.",
            analysis="The answer should prioritize ALD-specific experimental evidence, then supplement with general background only if useful.",
            synthesis_goal="Answer the user directly using grounded evidence, highlight the best-supported conclusion, and surface caveats clearly.",
            validation_focus=[
                "Check that every strong claim maps to retrieved evidence.",
                "Flag missing or conflicting experimental support.",
            ],
            steps=steps,
        )

    def _normalize_plan(self, raw_plan: dict[str, Any], request: ChatRequest) -> AgentPlan:
        plan = AgentPlan.model_validate(raw_plan)
        normalized_steps: list[PlanStep] = []
        seen_ids: set[str] = set()
        tool_steps = 0
        analysis_steps = 0

        for index, step in enumerate(plan.steps, start=1):
            step_id = step.step_id
            if step_id in seen_ids:
                prefix = "A" if step.step_type == "analysis" else "E"
                step_id = f"{prefix}{index}"
            seen_ids.add(step_id)

            if step.step_type == "analysis":
                analysis_steps += 1
                step = step.model_copy(update={"step_id": step_id, "tool_name": None, "arguments": {}})
            else:
                if not step.tool_name or step.tool_name not in self._tools:
                    raise ValueError(f"Unknown tool in plan: {step.tool_name}")
                tool_steps += 1
                step = step.model_copy(update={"step_id": step_id})
            normalized_steps.append(step)

        if analysis_steps == 0:
            normalized_steps.insert(
                0,
                PlanStep(
                    step_id="A1",
                    step_type="analysis",
                    title="Analyze query scope",
                    objective="Identify the evidence needed and the minimum tool sequence required to answer the query.",
                    expected_output="An execution-ready evidence checklist.",
                ),
            )
        if tool_steps == 0:
            raise ValueError("Planner returned no executable tool steps.")

        normalized_steps = normalized_steps[: self.settings.agentic_max_plan_steps]
        return plan.model_copy(update={"steps": normalized_steps})

    def _plan_request(self, request: ChatRequest) -> tuple[AgentPlan, bool]:
        if self.llm_available:
            try:
                raw_plan = self._get_llm().generate_plan(
                    query=request.query,
                    conversation=request.conversation,
                    scope_paper_id=request.scope_paper_id,
                    available_tools=self._available_tools(),
                    max_steps=self.settings.agentic_max_plan_steps,
                )
                return self._normalize_plan(raw_plan, request), True
            except Exception:
                pass
        return self._fallback_plan(request), False

    def _resolve_reference(self, store: dict[str, Any], step_id: str, path: str | None) -> Any:
        value = store.get(step_id)
        if path is None or path == "":
            return value

        current = value
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and part.isdigit():
                idx = int(part)
                current = current[idx] if 0 <= idx < len(current) else None
            else:
                return None
        return current

    def _resolve_templates(self, value: Any, store: dict[str, Any]) -> Any:
        if isinstance(value, str):
            exact_match = re.fullmatch(r"#([A-Za-z0-9_]+)(?:\.([A-Za-z0-9_.-]+))?", value.strip())
            if exact_match:
                return self._resolve_reference(store, exact_match.group(1), exact_match.group(2))

            def replace(match: re.Match[str]) -> str:
                resolved = self._resolve_reference(store, match.group(1), match.group(2))
                if resolved is None:
                    return ""
                if isinstance(resolved, (dict, list)):
                    return json.dumps(resolved)
                return str(resolved)

            return re.sub(r"#([A-Za-z0-9_]+)(?:\.([A-Za-z0-9_.-]+))?", replace, value)

        if isinstance(value, list):
            return [self._resolve_templates(item, store) for item in value]
        if isinstance(value, dict):
            return {key: self._resolve_templates(item, store) for key, item in value.items()}
        return value

    def _execute_tool(
        self,
        tool_name: str,
        resolved_arguments: dict[str, Any],
        request: ChatRequest,
    ) -> ToolExecutionResult:
        if tool_name == "rag_search":
            return self._tool_rag_search(
                query=str(resolved_arguments.get("query") or request.query),
                conversation=request.conversation,
                scope_paper_id=resolved_arguments.get("scope_paper_id", request.scope_paper_id),
                query_style=str(resolved_arguments.get("query_style") or "standard"),
            )
        if tool_name == "wikipedia_lookup":
            return self._tool_wikipedia_lookup(
                query=str(resolved_arguments.get("query") or request.query),
            )
        raise ValueError(f"Unknown tool: {tool_name}")

    def _execute_plan(self, plan: AgentPlan, request: ChatRequest) -> list[ExecutionArtifact]:
        artifacts: list[ExecutionArtifact] = []
        store: dict[str, Any] = {}

        for step in plan.steps:
            if step.step_type == "analysis":
                raw_output = {
                    "analysis": step.objective,
                    "planner_summary": plan.planner_summary,
                    "synthesis_goal": plan.synthesis_goal,
                }
                artifacts.append(
                    ExecutionArtifact(
                        step_id=step.step_id,
                        step_type=step.step_type,
                        title=step.title,
                        status="completed",
                        output_summary=step.expected_output,
                        raw_output=raw_output,
                    )
                )
                store[step.step_id] = raw_output
                continue

            missing_dependency = next((dep for dep in step.depends_on if dep not in store), None)
            if missing_dependency:
                artifacts.append(
                    ExecutionArtifact(
                        step_id=step.step_id,
                        step_type=step.step_type,
                        title=step.title,
                        tool_name=step.tool_name,
                        status="skipped",
                        output_summary=f"Skipped because dependency `{missing_dependency}` was unavailable.",
                        raw_output={"error": f"Missing dependency: {missing_dependency}"},
                    )
                )
                continue

            resolved_arguments = self._resolve_templates(step.arguments, store)
            try:
                result = self._execute_tool(step.tool_name or "", resolved_arguments, request)
                payload = {
                    **result.payload,
                    "tool_summary": result.summary,
                    "sources": [source.model_dump() for source in result.sources],
                }
                artifacts.append(
                    ExecutionArtifact(
                        step_id=step.step_id,
                        step_type=step.step_type,
                        title=step.title,
                        tool_name=step.tool_name,
                        status="completed",
                        resolved_arguments=resolved_arguments,
                        output_summary=result.summary,
                        raw_output=payload,
                    )
                )
                store[step.step_id] = payload
            except Exception as exc:
                payload = {
                    "error": str(exc),
                    "resolved_arguments": resolved_arguments,
                }
                artifacts.append(
                    ExecutionArtifact(
                        step_id=step.step_id,
                        step_type=step.step_type,
                        title=step.title,
                        tool_name=step.tool_name,
                        status="failed",
                        resolved_arguments=resolved_arguments,
                        output_summary=f"{step.tool_name or 'tool'} failed: {exc}",
                        raw_output=payload,
                    )
                )
                store[step.step_id] = payload

        return artifacts

    def _collect_sources(self, artifacts: list[ExecutionArtifact]) -> list[SourceChunk]:
        ordered: list[SourceChunk] = []
        seen: set[tuple[str, str]] = set()

        for artifact in artifacts:
            raw_output = artifact.raw_output if isinstance(artifact.raw_output, dict) else {}
            for raw_source in raw_output.get("sources", []):
                try:
                    source = SourceChunk.model_validate(raw_source)
                except Exception:
                    continue
                key = (source.source_type, source.source_id)
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(source)
        return ordered

    def _build_context(self, sources: list[SourceChunk]) -> str:
        blocks: list[str] = []
        total_chars = 0

        for idx, source in enumerate(sources, start=1):
            if source.source_type == "wikipedia":
                block = (
                    f"[S{idx}] source=wikipedia | title={source.title or source.source_id} | "
                    f"url={source.url or 'n/a'}\n"
                    f"{source.excerpt.strip()}"
                )
            else:
                block = (
                    f"[S{idx}] source=rag | paper_id={source.paper_id or 'unknown'} | "
                    f"material={source.target_material or 'Unknown'} | "
                    f"process={source.process_type or 'Unknown'}\n"
                    f"{source.excerpt.strip()}"
                )

            if total_chars + len(block) > self.settings.answer_context_chars:
                break
            blocks.append(block)
            total_chars += len(block)

        return "\n\n".join(blocks)

    def _summarize_execution(self, artifacts: list[ExecutionArtifact]) -> str:
        lines: list[str] = []
        for artifact in artifacts:
            prefix = artifact.step_id
            if artifact.tool_name:
                lines.append(f"{prefix} [{artifact.status}] {artifact.tool_name}: {artifact.output_summary}")
            else:
                lines.append(f"{prefix} [{artifact.status}] {artifact.title}: {artifact.output_summary}")
        return "\n".join(lines)

    def _fallback_answer(
        self,
        query: str,
        sources: list[SourceChunk],
        execution_summary: str,
    ) -> str:
        if not sources:
            return (
                "Answer:\n"
                "I could not assemble grounded evidence strong enough to answer that query.\n\n"
                "Recommended approach:\n"
                "- Narrow the question to a material, precursor set, process type, or specific paper.\n"
                "- Retry with a more explicit experimental target such as temperature window, oxidant choice, or film property.\n\n"
                "Evidence:\n"
                f"- Execution summary: {execution_summary or 'No tool outputs were available.'}\n\n"
                "Caveats:\n"
                "- This response used fallback synthesis because planner or LLM synthesis support was unavailable."
            )

        lines = [
            "Answer:",
            f"I gathered {len(sources)} grounded source blocks for: {query}",
            "",
            "Recommended approach:",
            "- Use the evidence cards below as the grounding layer for the final interpretation.",
            "",
            "Evidence:",
        ]
        for idx, source in enumerate(sources, start=1):
            descriptor = source.title or source.paper_id or source.source_id
            lines.append(f"- [S{idx}] {descriptor}: {source.excerpt[:220].strip()}...")
        lines.extend(
            [
                "",
                "Caveats:",
                "- This is a fallback synthesis path, so the answer is a compact evidence summary rather than a fully planned narrative.",
            ]
        )
        return "\n".join(lines)

    def _synthesize_answer(
        self,
        request: ChatRequest,
        plan: AgentPlan,
        artifacts: list[ExecutionArtifact],
        sources: list[SourceChunk],
    ) -> str:
        execution_summary = self._summarize_execution(artifacts)
        if self.llm_available and sources:
            context_blocks = self._build_context(sources)
            return self._get_llm().generate_answer(
                query=request.query,
                conversation=request.conversation,
                scope_paper_id=request.scope_paper_id,
                plan_summary=plan.planner_summary,
                execution_summary=execution_summary,
                context_blocks=context_blocks,
            )
        return self._fallback_answer(request.query, sources, execution_summary)

    def _build_validation_queries(
        self,
        request: ChatRequest,
        plan: AgentPlan,
        draft_answer: str,
    ) -> list[str]:
        if not self.settings.validation_enabled:
            return []

        queries: list[str] = []
        if self.llm_available:
            try:
                queries.extend(
                    self._get_llm().generate_validation_queries(
                        query=request.query,
                        draft_answer=draft_answer,
                        validation_focus=plan.validation_focus,
                        limit=self.settings.validation_query_limit,
                    )
                )
            except Exception:
                pass

        if not queries:
            queries.append(f"{request.query} experimental evidence and counterexamples")

        deduped: list[str] = []
        seen: set[str] = set()
        for item in queries:
            cleaned = item.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(cleaned)
        return deduped[: self.settings.validation_query_limit]

    def _run_validation_retrievals(
        self,
        request: ChatRequest,
        queries: list[str],
    ) -> list[ExecutionArtifact]:
        artifacts: list[ExecutionArtifact] = []
        for index, query in enumerate(queries, start=1):
            step_id = f"V{index}"
            try:
                result = self._tool_rag_search(
                    query=query,
                    conversation=request.conversation,
                    scope_paper_id=request.scope_paper_id,
                    query_style="verification",
                )
                payload = {
                    **result.payload,
                    "tool_summary": result.summary,
                    "sources": [source.model_dump() for source in result.sources],
                }
                artifacts.append(
                    ExecutionArtifact(
                        step_id=step_id,
                        step_type="tool",
                        title="Cross-verify evidence",
                        tool_name="rag_search",
                        status="completed",
                        resolved_arguments={
                            "query": query,
                            "scope_paper_id": request.scope_paper_id,
                            "query_style": "verification",
                        },
                        output_summary=result.summary,
                        raw_output=payload,
                    )
                )
            except Exception as exc:
                artifacts.append(
                    ExecutionArtifact(
                        step_id=step_id,
                        step_type="tool",
                        title="Cross-verify evidence",
                        tool_name="rag_search",
                        status="failed",
                        resolved_arguments={
                            "query": query,
                            "scope_paper_id": request.scope_paper_id,
                            "query_style": "verification",
                        },
                        output_summary=f"Validation retrieval failed: {exc}",
                        raw_output={"error": str(exc), "sources": []},
                    )
                )
        return artifacts

    def _fallback_validation(
        self,
        sources: list[SourceChunk],
        validation_queries: list[str],
        validation_artifacts: list[ExecutionArtifact],
    ) -> ValidationReport:
        factual = "pass" if sources else "fail"
        cross = "pass" if any(self._collect_sources(validation_artifacts)) else "warning"
        logical = "pass" if sources else "warning"
        issues: list[str] = []
        if not sources:
            issues.append("No grounded sources were available for the final answer.")
        if validation_queries and cross != "pass":
            issues.append("Cross-verification queries did not return usable evidence.")

        verdict = "pass" if factual == "pass" and logical == "pass" and cross in {"pass", "warning"} else "warning"
        if factual == "fail":
            verdict = "fail"

        return ValidationReport(
            factual_grounding=factual,
            logical_consistency=logical,
            cross_verification=cross,
            issues=issues,
            summary="Fallback validation assessed source coverage and whether redundant retrieval produced usable evidence.",
            verdict=verdict,
            revised_answer=None,
        )

    def _validate_answer(
        self,
        request: ChatRequest,
        plan: AgentPlan,
        draft_answer: str,
        sources: list[SourceChunk],
        validation_queries: list[str],
        validation_artifacts: list[ExecutionArtifact],
    ) -> ValidationReport:
        if not self.settings.validation_enabled:
            return ValidationReport(
                factual_grounding="warning",
                logical_consistency="warning",
                cross_verification="warning",
                issues=["Validation was disabled in configuration."],
                summary="Validation agent was skipped.",
                verdict="warning",
                revised_answer=None,
            )

        if self.llm_available and sources:
            try:
                context_blocks = self._build_context(sources)
                raw_report = self._get_llm().validate_answer(
                    query=request.query,
                    draft_answer=draft_answer,
                    validation_focus=plan.validation_focus,
                    context_blocks=context_blocks,
                )
                if not raw_report.get("revised_answer"):
                    raw_report["revised_answer"] = None
                return ValidationReport.model_validate(raw_report)
            except Exception:
                pass

        return self._fallback_validation(sources, validation_queries, validation_artifacts)

    def chat(self, request: ChatRequest) -> ChatResponse:
        query = request.query.strip()
        if not query:
            raise ValueError("Query must not be empty.")

        normalized_request = request.model_copy(update={"query": query})
        plan, planner_used = self._plan_request(normalized_request)
        execution = self._execute_plan(plan, normalized_request)
        primary_sources = self._collect_sources(execution)
        draft_answer = self._synthesize_answer(normalized_request, plan, execution, primary_sources)

        validation_queries = self._build_validation_queries(normalized_request, plan, draft_answer)
        validation_execution = self._run_validation_retrievals(normalized_request, validation_queries)
        full_execution = [*execution, *validation_execution]
        all_sources = self._collect_sources(full_execution)
        validation = self._validate_answer(
            request=normalized_request,
            plan=plan,
            draft_answer=draft_answer,
            sources=all_sources,
            validation_queries=validation_queries,
            validation_artifacts=validation_execution,
        )

        final_answer = validation.revised_answer or draft_answer
        tool_calls = sum(1 for artifact in full_execution if artifact.step_type == "tool")
        diagnostics = RetrievalDiagnostics(
            scope=normalized_request.scope_paper_id or "global-catalog",
            hyde_enabled=any(
                bool((artifact.raw_output or {}).get("retrieval_diagnostics", {}).get("hyde_enabled"))
                for artifact in full_execution
                if isinstance(artifact.raw_output, dict)
            ),
            retrieved_count=len(all_sources),
            reranked_count=sum(
                1
                for source in all_sources
                if source.source_type == "rag"
            ),
            hyde_preview=next(
                (
                    (artifact.raw_output or {}).get("retrieval_diagnostics", {}).get("hyde_preview")
                    for artifact in full_execution
                    if isinstance(artifact.raw_output, dict)
                    and (artifact.raw_output or {}).get("retrieval_diagnostics", {}).get("hyde_preview")
                ),
                None,
            ),
            planner_used=planner_used,
            executed_steps=len(full_execution),
            tool_calls=tool_calls,
            validation_status=validation.verdict,
            validation_queries=validation_queries,
        )

        return ChatResponse(
            answer=final_answer,
            sources=all_sources,
            diagnostics=diagnostics,
            plan=plan,
            execution=full_execution,
            validation=validation,
        )
