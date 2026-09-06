"""FastAPI 应用入口.

启动顺序:
  1. 初始化日志
  2. 校验配置
  3. 创建 FastAPI 实例
  4. 注册中间件
  5. 注册全局异常处理器
  6. 注册路由
  7. 挂载静态文件 (前端)
  8. lifespan 钩子 (启动时连 Milvus, 关闭时断开)

启动方式:
  开发: uvicorn app.main:app --reload
  生产: uvicorn app.main:app --workers 4
"""

import asyncio
import contextlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.api.middleware import setup_middlewares
from app.api.v1 import aiops, chat, documents, eval as eval_api, health, incidents, queue, skills, webhook, wiki, approvals
from app.config import settings
from app.core.mcp_client import mcp_client_manager
from app.core.milvus import milvus_manager
from app.core.redis_client import close_redis_client
from app.db.postgres import close_postgres, connect_postgres, init_incident_schema
from app.exceptions import AppException
from app.logging_config import setup_logging
from app.queue.kafka import incident_queue
from app.schemas.common import ApiResponse
from app.observability.metrics import render_metrics
from app.observability.tracing import setup_tracing
from app.observability.langfuse_client import flush_langfuse


# ============================================================
# Lifespan: 启动/关闭钩子
# ============================================================
async def _refresh_queue_metrics() -> None:
    """Refresh Kafka lag/DLQ gauges; the exporter must not depend on UI polling."""
    interval = max(5, int(settings.queue_metrics_refresh_sec))
    while True:
        try:
            await incident_queue.status()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                f"[observability] queue metrics refresh failed: {type(exc).__name__}: {exc}"
            )
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期管理.

    yield 之前: 启动时执行 (连数据库、初始化资源)
    yield 之后: 关闭时执行 (清理资源)
    """
    # ==================== 启动 ====================
    logger.info("=" * 60)
    logger.info(f"启动 {settings.app_name} v{settings.app_version}")
    logger.info(f"运行模式: {'DEBUG' if settings.debug else 'PRODUCTION'}")
    logger.info("=" * 60)

    # 1. 校验配置 (无效时直接抛错, 让 uvicorn 退出)
    settings.validate_runtime()

    # 2. 连接 Milvus (必需依赖, 失败则启动失败)
    milvus_manager.connect()

    # 3. 初始化 Incident Pipeline (Postgres 事实库 + Kafka 队列)
    queue_metrics_task: asyncio.Task[None] | None = None
    if settings.incident_pipeline_enabled:
        await connect_postgres()
        await init_incident_schema()
        await incident_queue.connect()
        if (
            settings.observability_enabled
            and settings.metrics_enabled
            and settings.queue_metrics_refresh_sec > 0
        ):
            queue_metrics_task = asyncio.create_task(
                _refresh_queue_metrics(),
                name="kafka-metrics-refresh",
            )

    # 4. 加载 MCP 工具 (可选依赖, 失败仅 warning)
    await mcp_client_manager.connect(fail_silently=True)

    logger.info("应用就绪, 等待请求...")
    yield
    # ==================== 关闭 ====================
    logger.info("应用正在关闭...")
    if queue_metrics_task is not None:
        queue_metrics_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await queue_metrics_task
    await mcp_client_manager.close()
    if settings.incident_pipeline_enabled:
        await incident_queue.close()
        await close_postgres()
    await close_redis_client()
    milvus_manager.disconnect()
    flush_langfuse()
    logger.info("应用已关闭")


# ============================================================
# 创建 FastAPI 实例
# ============================================================
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="企业级多智能体智能运维诊断平台 - 基于 LangGraph + RAG + MCP",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ============================================================
# 中间件
# ============================================================
setup_middlewares(app)
setup_tracing("api", fastapi_app=app)


# ============================================================
# 全局异常处理器
# ============================================================
@app.exception_handler(AppException)
async def handle_app_exception(request: Request, exc: AppException) -> JSONResponse:
    """处理业务异常."""
    logger.warning(f"业务异常: {exc.code} - {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ApiResponse.error(
            code=exc.code, message=exc.message, detail=exc.detail
        ).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """处理 Pydantic 校验失败."""
    logger.warning(f"参数校验失败: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content=ApiResponse.error(
            code="VALIDATION_ERROR",
            message="请求参数校验失败",
            detail=exc.errors(),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def handle_unexpected_exception(
    request: Request, exc: Exception
) -> JSONResponse:
    """兜底: 处理所有未捕获异常."""
    logger.exception(f"未预期的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content=ApiResponse.error(
            code="INTERNAL_ERROR",
            message="服务内部错误, 请稍后重试或联系管理员",
            detail=str(exc) if settings.debug else None,
        ).model_dump(),
    )


# ============================================================
# 路由注册
# ============================================================
API_PREFIX = "/api/v1"


@app.get(settings.metrics_path, include_in_schema=False)
async def prometheus_metrics() -> Response:
    """Expose platform and AgentOps metrics for Prometheus scraping."""
    if not settings.observability_enabled or not settings.metrics_enabled:
        return Response(status_code=404)
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)

app.include_router(health.router, prefix=API_PREFIX)
app.include_router(chat.router, prefix=API_PREFIX)
app.include_router(aiops.router, prefix=API_PREFIX)
app.include_router(documents.router, prefix=API_PREFIX)
app.include_router(incidents.router, prefix=API_PREFIX)
app.include_router(skills.router, prefix=API_PREFIX)
app.include_router(webhook.router, prefix=API_PREFIX)
app.include_router(queue.router, prefix=API_PREFIX)
app.include_router(eval_api.router, prefix=API_PREFIX)
app.include_router(wiki.router, prefix=API_PREFIX)
app.include_router(approvals.router, prefix=API_PREFIX)


# ============================================================
# 静态文件 (前端)
# ============================================================
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(FRONTEND_DIR), html=True),
        name="frontend",
    )
else:
    logger.warning(f"前端目录不存在: {FRONTEND_DIR} (将在阶段 5 创建)")

    @app.get("/", include_in_schema=False)
    async def root() -> ApiResponse[dict]:
        """临时根路由 (前端目录还没创建时使用)."""
        return ApiResponse.success(
            data={
                "name": settings.app_name,
                "version": settings.app_version,
                "docs": "/docs",
                "health": f"{API_PREFIX}/health",
            },
            message="服务运行中, 前端目录将在阶段 5 创建",
        )


# ============================================================
# 启动时初始化日志
# ============================================================
setup_logging()
