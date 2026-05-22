"""Graphiti / Neo4j 分页读取工具。

使用 Cypher SKIP/LIMIT 对 Neo4j 中的节点和边进行分页迭代，
封装自动翻页逻辑（含单页重试），对调用方透明地返回完整列表。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from graphiti_core import Graphiti

from .async_runner import run_async
from .logger import get_logger

logger = get_logger('mirofish.zep_paging')

_DEFAULT_PAGE_SIZE = 100
_MAX_NODES = 2000
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_DELAY = 2.0  # seconds, doubles each retry




def _fetch_page_with_retry(
    api_call: Callable[..., Any],
    *args: Any,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    retry_delay: float = _DEFAULT_RETRY_DELAY,
    page_description: str = "page",
    **kwargs: Any,
) -> list[Any]:
    """Single-page request with exponential-backoff retry on transient errors."""
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")

    last_exception: Exception | None = None
    delay = retry_delay

    for attempt in range(max_retries):
        try:
            return api_call(*args, **kwargs)
        except (ConnectionError, TimeoutError, OSError, Exception) as e:
            last_exception = e
            if attempt < max_retries - 1:
                logger.warning(
                    f"Graphiti {page_description} attempt {attempt + 1} failed: "
                    f"{str(e)[:100]}, retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= 2
            else:
                logger.error(
                    f"Graphiti {page_description} failed after {max_retries} attempts: {str(e)}"
                )

    assert last_exception is not None
    raise last_exception


async def _fetch_nodes_page(
    graphiti: Graphiti,
    graph_id: str,
    skip: int,
    limit: int,
) -> list[Any]:
    """Fetch a single page of Entity nodes ordered by uuid."""
    async with graphiti.driver.session() as session:
        result = await session.run(
            "MATCH (n:Entity {group_id: $gid}) "
            "RETURN n ORDER BY n.uuid SKIP $skip LIMIT $limit",
            gid=graph_id,
            skip=skip,
            limit=limit,
        )
        return [rec["n"] async for rec in result]


async def _fetch_edges_page(
    graphiti: Graphiti,
    graph_id: str,
    skip: int,
    limit: int,
) -> list[Any]:
    """Fetch a single page of RELATES_TO edges ordered by uuid."""
    async with graphiti.driver.session() as session:
        result = await session.run(
            "MATCH ()-[e:RELATES_TO {group_id: $gid}]->() "
            "RETURN e ORDER BY e.uuid SKIP $skip LIMIT $limit",
            gid=graph_id,
            skip=skip,
            limit=limit,
        )
        return [rec["e"] async for rec in result]


async def _fetch_node_by_uuid(graphiti: Graphiti, node_uuid: str) -> Any | None:
    """Fetch a single Entity node by its uuid."""
    async with graphiti.driver.session() as session:
        result = await session.run(
            "MATCH (n:Entity {uuid: $uuid}) RETURN n",
            uuid=node_uuid,
        )
        rec = await result.single()
        return rec["n"] if rec else None


def fetch_all_nodes(
    graphiti: Graphiti,
    graph_id: str,
    page_size: int = _DEFAULT_PAGE_SIZE,
    max_items: int = _MAX_NODES,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    retry_delay: float = _DEFAULT_RETRY_DELAY,
) -> list[Any]:
    """分页获取图谱 Entity 节点，最多返回 max_items 条（默认 2000）。每页请求自带重试。

    Args:
        graphiti: 已初始化的 Graphiti 客户端实例。
        graph_id: 图谱分组 ID（Neo4j 中的 group_id 属性）。
        page_size: 每页节点数，默认 100。
        max_items: 总节点数上限，默认 2000。
        max_retries: 每页最大重试次数，默认 3。
        retry_delay: 首次重试延迟秒数，默认 2.0（指数退避）。

    Returns:
        Neo4j 节点对象列表（neo4j.graph.Node）。
    """
    all_nodes: list[Any] = []
    skip = 0
    page_num = 0

    while len(all_nodes) < max_items:
        page_num += 1

        batch = _fetch_page_with_retry(
            run_async,
            _fetch_nodes_page(graphiti, graph_id, skip, page_size),
            max_retries=max_retries,
            retry_delay=retry_delay,
            page_description=f"fetch nodes page {page_num} (graph={graph_id})",
        )

        if not batch:
            break

        all_nodes.extend(batch)

        if len(all_nodes) >= max_items:
            all_nodes = all_nodes[:max_items]
            logger.warning(
                f"Node count reached limit ({max_items}), "
                f"stopping pagination for graph {graph_id}"
            )
            break

        if len(batch) < page_size:
            break

        skip += page_size

    logger.debug(f"fetch_all_nodes: retrieved {len(all_nodes)} nodes for graph {graph_id}")
    return all_nodes


def fetch_all_edges(
    graphiti: Graphiti,
    graph_id: str,
    page_size: int = _DEFAULT_PAGE_SIZE,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    retry_delay: float = _DEFAULT_RETRY_DELAY,
) -> list[Any]:
    """分页获取图谱所有 RELATES_TO 边，返回完整列表。每页请求自带重试。

    Args:
        graphiti: 已初始化的 Graphiti 客户端实例。
        graph_id: 图谱分组 ID（Neo4j 中的 group_id 属性）。
        page_size: 每页边数，默认 100。
        max_retries: 每页最大重试次数，默认 3。
        retry_delay: 首次重试延迟秒数，默认 2.0（指数退避）。

    Returns:
        Neo4j 关系对象列表（neo4j.graph.Relationship）。
    """
    all_edges: list[Any] = []
    skip = 0
    page_num = 0

    while True:
        page_num += 1

        batch = _fetch_page_with_retry(
            run_async,
            _fetch_edges_page(graphiti, graph_id, skip, page_size),
            max_retries=max_retries,
            retry_delay=retry_delay,
            page_description=f"fetch edges page {page_num} (graph={graph_id})",
        )

        if not batch:
            break

        all_edges.extend(batch)

        if len(batch) < page_size:
            break

        skip += page_size

    logger.debug(f"fetch_all_edges: retrieved {len(all_edges)} edges for graph {graph_id}")
    return all_edges


def fetch_node_by_uuid(
    graphiti: Graphiti,
    node_uuid: str,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    retry_delay: float = _DEFAULT_RETRY_DELAY,
) -> Any | None:
    """UUID で単一の Entity ノードを取得する。

    Args:
        graphiti: 已初始化的 Graphiti 客户端实例。
        node_uuid: 要查询的节点 uuid。
        max_retries: 最大重试次数，默认 3。
        retry_delay: 首次重试延迟秒数，默认 2.0。

    Returns:
        Neo4j 节点对象，或 None（节点不存在时）。
    """
    return _fetch_page_with_retry(
        run_async,
        _fetch_node_by_uuid(graphiti, node_uuid),
        max_retries=max_retries,
        retry_delay=retry_delay,
        page_description=f"fetch node by uuid={node_uuid[:8]}...",
    )
