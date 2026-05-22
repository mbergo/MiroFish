"""
图谱构建服务
接口2：使用Graphiti (Neo4j) 构建Standalone Graph
"""

import asyncio
import time
import threading
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from ..utils.async_runner import run_async

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from ..config import Config
from ..models.task import TaskManager, TaskStatus
from .text_processor import TextProcessor
from ..utils.locale import get_locale, set_locale, t


# ---------------------------------------------------------------------------
# Async-in-sync helper
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GraphInfo:
    """图谱信息"""
    graph_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class GraphBuilderService:
    """
    图谱构建服务
    负责调用Graphiti / Neo4j API构建知识图谱
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ):
        self._uri = neo4j_uri or Config.NEO4J_URI
        self._user = neo4j_user or Config.NEO4J_USER
        self._password = neo4j_password or Config.NEO4J_PASSWORD

        if not self._uri:
            raise ValueError("NEO4J_URI 未配置")

        self.client: Graphiti = Graphiti(
            uri=self._uri,
            user=self._user,
            password=self._password,
        )
        # Ensure indices and constraints exist (idempotent)
        run_async(self.client.build_indices_and_constraints())

        self.task_manager = TaskManager()
        self._ontologies: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API — signatures kept identical to the previous Zep version
    # ------------------------------------------------------------------

    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 3,
    ) -> str:
        """
        异步构建图谱

        Args:
            text: 输入文本
            ontology: 本体定义（来自接口1的输出；Graphiti中仅作本地存储，不上传）
            graph_name: 图谱名称
            chunk_size: 文本块大小
            chunk_overlap: 块重叠大小
            batch_size: 每批发送的块数量

        Returns:
            任务ID
        """
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
            },
        )

        current_locale = get_locale()

        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(task_id, text, ontology, graph_name, chunk_size, chunk_overlap, batch_size, current_locale),
        )
        thread.daemon = True
        thread.start()

        return task_id

    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
        locale: str = "zh",
    ):
        """图谱构建工作线程"""
        set_locale(locale)
        try:
            self.task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=5,
                message=t("progress.startBuildingGraph"),
            )

            # 1. 创建图谱（生成 group_id）
            graph_id = self.create_graph(graph_name)
            self.task_manager.update_task(
                task_id,
                progress=10,
                message=t("progress.graphCreated", graphId=graph_id),
            )

            # 2. 本体在Graphiti中无需上传；本地保存供其他用途
            self.set_ontology(graph_id, ontology)
            self.task_manager.update_task(
                task_id,
                progress=15,
                message=t("progress.ontologySet"),
            )

            # 3. 文本分块
            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            total_chunks = len(chunks)
            self.task_manager.update_task(
                task_id,
                progress=20,
                message=t("progress.textSplit", count=total_chunks),
            )

            # 4. 分批发送数据
            episode_uuids = self.add_text_batches(
                graph_id,
                chunks,
                batch_size,
                lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=20 + int(prog * 0.4),  # 20-60 %
                    message=msg,
                ),
            )

            # 5. 等待处理完成
            self.task_manager.update_task(
                task_id,
                progress=60,
                message=t("progress.waitingZepProcess"),
            )

            self._wait_for_episodes(
                episode_uuids,
                lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=60 + int(prog * 0.3),  # 60-90 %
                    message=msg,
                ),
            )

            # 6. 获取图谱信息
            self.task_manager.update_task(
                task_id,
                progress=90,
                message=t("progress.fetchingGraphInfo"),
            )

            graph_info = self._get_graph_info(graph_id)

            self.task_manager.complete_task(
                task_id,
                {
                    "graph_id": graph_id,
                    "graph_info": graph_info.to_dict(),
                    "chunks_processed": total_chunks,
                },
            )

        except Exception as e:
            import traceback

            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.task_manager.fail_task(task_id, error_msg)

    def create_graph(self, name: str) -> str:
        """创建图谱，返回 group_id（公开方法）

        Graphiti uses group_id for namespacing instead of a server-side graph
        create call.  We generate a stable UUID-based group_id and ensure the
        driver indices are up to date.
        """
        group_id = f"mirofish_{uuid.uuid4().hex[:16]}"
        # build_indices_and_constraints is idempotent — safe to call again
        run_async(self.client.build_indices_and_constraints())
        return group_id

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        """本地存储本体定义（公开方法）

        Graphiti does not have a server-side ontology concept.  The ontology
        dict is stored on the instance so callers that depend on this method
        signature continue to work without changes.
        """
        # Store locally — available to other methods if needed in the future
        if not hasattr(self, "_ontologies"):
            self._ontologies: Dict[str, Dict[str, Any]] = {}
        self._ontologies[graph_id] = ontology

    def add_text_batches(
        self,
        graph_id: str,
        chunks: List[str],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None,
    ) -> List[str]:
        """分批添加文本到图谱，返回所有 episode 的 uuid 列表"""
        episode_uuids: List[str] = []
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size

            if progress_callback:
                progress = (i + len(batch_chunks)) / total_chunks
                progress_callback(
                    t(
                        "progress.sendingBatch",
                        current=batch_num,
                        total=total_batches,
                        chunks=len(batch_chunks),
                    ),
                    progress,
                )

            # Add each chunk as an individual episode
            try:
                for chunk in batch_chunks:
                    ep_name = str(uuid.uuid4())
                    run_async(
                        self.client.add_episode(
                            name=ep_name,
                            episode_body=chunk,
                            source=EpisodeType.text,
                            source_description="MiroFish document chunk",
                            group_id=graph_id,
                            reference_time=datetime.now(),
                        )
                    )
                    episode_uuids.append(ep_name)

                # Avoid overwhelming the database
                time.sleep(1)

            except Exception as e:
                if progress_callback:
                    progress_callback(
                        t("progress.batchFailed", batch=batch_num, error=str(e)), 0
                    )
                raise

        return episode_uuids

    def _wait_for_episodes(
        self,
        episode_uuids: List[str],
        progress_callback: Optional[Callable] = None,
        timeout: int = 600,
    ):
        """等待所有 episode 处理完成

        With Graphiti / Neo4j the add_episode call is synchronous from the
        caller's perspective (we already awaited it in add_text_batches).
        We perform a lightweight existence check in Neo4j rather than polling
        a 'processed' flag, and complete immediately when all episodes are
        found — falling through with a log if any are missing after timeout.
        """
        if not episode_uuids:
            if progress_callback:
                progress_callback(t("progress.noEpisodesWait"), 1.0)
            return

        start_time = time.time()
        pending_episodes = set(episode_uuids)
        completed_count = 0
        total_episodes = len(episode_uuids)

        if progress_callback:
            progress_callback(t("progress.waitingEpisodes", count=total_episodes), 0)

        while pending_episodes:
            if time.time() - start_time > timeout:
                if progress_callback:
                    progress_callback(
                        t(
                            "progress.episodesTimeout",
                            completed=completed_count,
                            total=total_episodes,
                        ),
                        completed_count / total_episodes,
                    )
                break

            for ep_name in list(pending_episodes):
                try:
                    found = run_async(self._episode_exists(ep_name))
                    if found:
                        pending_episodes.remove(ep_name)
                        completed_count += 1
                except Exception:
                    pass

            elapsed = int(time.time() - start_time)
            if progress_callback:
                progress_callback(
                    t(
                        "progress.zepProcessing",
                        completed=completed_count,
                        total=total_episodes,
                        pending=len(pending_episodes),
                        elapsed=elapsed,
                    ),
                    completed_count / total_episodes if total_episodes > 0 else 0,
                )

            if pending_episodes:
                time.sleep(3)

        if progress_callback:
            progress_callback(
                t("progress.processingComplete", completed=completed_count, total=total_episodes),
                1.0,
            )

    async def _episode_exists(self, episode_name: str) -> bool:
        """Check whether an Episodic node with the given name exists in Neo4j."""
        async with self.client.driver.session() as session:
            result = await session.run(
                "MATCH (e:Episodic {name: $name}) RETURN count(e) AS c",
                name=episode_name,
            )
            record = await result.single()
            return bool(record and record["c"] > 0)

    def _get_graph_info(self, graph_id: str) -> GraphInfo:
        """获取图谱信息（直接查询 Neo4j）"""
        node_count, edge_count, entity_types = run_async(
            self._fetch_graph_stats(graph_id)
        )
        return GraphInfo(
            graph_id=graph_id,
            node_count=node_count,
            edge_count=edge_count,
            entity_types=entity_types,
        )

    async def _fetch_graph_stats(
        self, group_id: str
    ) -> tuple:
        """Query Neo4j for node count, edge count, and entity type labels."""
        async with self.client.driver.session() as session:
            # Node count
            r = await session.run(
                "MATCH (n:Entity {group_id: $gid}) RETURN count(n) AS c",
                gid=group_id,
            )
            rec = await r.single()
            node_count: int = rec["c"] if rec else 0

            # Edge count
            r = await session.run(
                "MATCH ()-[e {group_id: $gid}]->() RETURN count(e) AS c",
                gid=group_id,
            )
            rec = await r.single()
            edge_count: int = rec["c"] if rec else 0

            # Entity type labels (exclude generic labels)
            r = await session.run(
                "MATCH (n:Entity {group_id: $gid}) "
                "UNWIND labels(n) AS lbl "
                "WHERE lbl NOT IN ['Entity', 'Node'] "
                "RETURN DISTINCT lbl",
                gid=group_id,
            )
            records = await r.data()
            entity_types: List[str] = [row["lbl"] for row in records]

        return node_count, edge_count, entity_types

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """
        获取完整图谱数据（包含详细信息）

        Args:
            graph_id: 图谱ID (group_id)

        Returns:
            包含nodes和edges的字典，包括时间信息、属性等详细数据
        """
        return run_async(self._fetch_graph_data(graph_id))

    async def _fetch_graph_data(self, group_id: str) -> Dict[str, Any]:
        """Async implementation of get_graph_data querying Neo4j directly."""
        async with self.client.driver.session() as session:
            # Fetch entity nodes
            r = await session.run(
                "MATCH (n:Entity {group_id: $gid}) "
                "RETURN n.uuid AS uuid, n.name AS name, "
                "       labels(n) AS labels, n.summary AS summary, "
                "       n.created_at AS created_at",
                gid=group_id,
            )
            node_records = await r.data()

            nodes_data: List[Dict[str, Any]] = []
            node_map: Dict[str, str] = {}
            for row in node_records:
                node_uuid = row.get("uuid") or ""
                node_name = row.get("name") or ""
                node_map[node_uuid] = node_name
                created_at = row.get("created_at")
                nodes_data.append(
                    {
                        "uuid": node_uuid,
                        "name": node_name,
                        "labels": row.get("labels") or [],
                        "summary": row.get("summary") or "",
                        "attributes": {},
                        "created_at": str(created_at) if created_at else None,
                    }
                )

            # Fetch edges
            r = await session.run(
                "MATCH (s:Entity {group_id: $gid})-[e]->(t:Entity {group_id: $gid}) "
                "RETURN e.uuid AS uuid, type(e) AS name, e.fact AS fact, "
                "       s.uuid AS source_node_uuid, t.uuid AS target_node_uuid, "
                "       e.created_at AS created_at, e.valid_at AS valid_at, "
                "       e.invalid_at AS invalid_at, e.expired_at AS expired_at, "
                "       e.episodes AS episodes",
                gid=group_id,
            )
            edge_records = await r.data()

            edges_data: List[Dict[str, Any]] = []
            for row in edge_records:
                ep_raw = row.get("episodes")
                if ep_raw and not isinstance(ep_raw, list):
                    episodes = [str(ep_raw)]
                elif ep_raw:
                    episodes = [str(e) for e in ep_raw]
                else:
                    episodes = []

                src_uuid = row.get("source_node_uuid") or ""
                tgt_uuid = row.get("target_node_uuid") or ""
                edge_name = row.get("name") or ""

                edges_data.append(
                    {
                        "uuid": row.get("uuid") or "",
                        "name": edge_name,
                        "fact": row.get("fact") or "",
                        "fact_type": edge_name,
                        "source_node_uuid": src_uuid,
                        "target_node_uuid": tgt_uuid,
                        "source_node_name": node_map.get(src_uuid, ""),
                        "target_node_name": node_map.get(tgt_uuid, ""),
                        "attributes": {},
                        "created_at": str(row["created_at"]) if row.get("created_at") else None,
                        "valid_at": str(row["valid_at"]) if row.get("valid_at") else None,
                        "invalid_at": str(row["invalid_at"]) if row.get("invalid_at") else None,
                        "expired_at": str(row["expired_at"]) if row.get("expired_at") else None,
                        "episodes": episodes,
                    }
                )

        return {
            "graph_id": group_id,
            "nodes": nodes_data,
            "edges": edges_data,
            "node_count": len(nodes_data),
            "edge_count": len(edges_data),
        }

    def delete_graph(self, graph_id: str):
        """删除图谱（删除 Neo4j 中所有属于该 group_id 的节点和边）"""
        run_async(self._delete_group(graph_id))

    async def _delete_group(self, group_id: str):
        async with self.client.driver.session() as session:
            await session.run(
                "MATCH (n {group_id: $gid}) DETACH DELETE n",
                gid=group_id,
            )
