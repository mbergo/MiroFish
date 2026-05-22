"""
Graphiti 实体读取与过滤服务
从 Neo4j / Graphiti 图谱中读取节点，筛选出符合预定义实体类型的节点
"""

import time
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar
from ..utils.async_runner import run_async

from dataclasses import dataclass, field

from graphiti_core import Graphiti

from ..config import Config
from ..utils.logger import get_logger
from ..utils.zep_paging import (
    fetch_all_edges,
    fetch_all_nodes,
    fetch_node_by_uuid,
)

logger = get_logger('mirofish.zep_entity_reader')

# 用于泛型返回类型
T = TypeVar('T')


@dataclass
class EntityNode:
    """实体节点数据结构"""

    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    # 相关的边信息
    related_edges: List[Dict[str, Any]] = field(default_factory=list)
    # 相关的其他节点信息
    related_nodes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "related_edges": self.related_edges,
            "related_nodes": self.related_nodes,
        }

    def get_entity_type(self) -> Optional[str]:
        """获取实体类型（排除默认的Entity标签）"""
        for label in self.labels:
            if label not in ["Entity", "Node"]:
                return label
        return None


@dataclass
class FilteredEntities:
    """过滤后的实体集合"""

    entities: List[EntityNode]
    entity_types: Set[str]
    total_count: int
    filtered_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "entity_types": list(self.entity_types),
            "total_count": self.total_count,
            "filtered_count": self.filtered_count,
        }


def _neo4j_node_to_dict(node: Any) -> Dict[str, Any]:
    """将 Neo4j 节点对象转换为统一的字典格式。

    Args:
        node: neo4j.graph.Node 对象。

    Returns:
        包含 uuid、name、labels、summary、attributes 字段的字典。
    """
    props = dict(node)
    return {
        "uuid": props.get("uuid", ""),
        "name": props.get("name", ""),
        "labels": list(node.labels),
        "summary": props.get("summary", ""),
        "attributes": props,
    }


def _neo4j_edge_to_dict(edge: Any) -> Dict[str, Any]:
    """将 Neo4j 关系对象转换为统一的字典格式。

    Args:
        edge: neo4j.graph.Relationship 对象。

    Returns:
        包含 uuid、name、fact、source_node_uuid、target_node_uuid、attributes 字段的字典。
    """
    props = dict(edge)
    # Neo4j relationship 的端点节点通过 edge.start_node / edge.end_node 暴露，
    # 但 Graphiti 将 source/target uuid 存储为关系属性。
    source_uuid = props.get("source_node_uuid") or props.get("source_uuid", "")
    target_uuid = props.get("target_node_uuid") or props.get("target_uuid", "")

    return {
        "uuid": props.get("uuid", ""),
        "name": props.get("name", ""),
        "fact": props.get("fact", ""),
        "source_node_uuid": source_uuid,
        "target_node_uuid": target_uuid,
        "attributes": props,
    }


class ZepEntityReader:
    """
    Graphiti / Neo4j 实体读取与过滤服务

    主要功能：
    1. 从 Neo4j 图谱读取所有节点
    2. 筛选出符合预定义实体类型的节点（Labels 不只是 Entity 的节点）
    3. 获取每个实体的相关边和关联节点信息

    公共 API 与原 Zep 版本完全兼容。
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        # api_key 参数保留以兼容现有调用方，但 Graphiti 使用 Neo4j 凭据。
        _ = api_key  # intentionally unused

        neo4j_uri = Config.NEO4J_URI
        neo4j_user = Config.NEO4J_USER
        neo4j_password = Config.NEO4J_PASSWORD

        if not neo4j_uri:
            raise ValueError("NEO4J_URI 未配置")
        if not neo4j_user:
            raise ValueError("NEO4J_USER 未配置")
        if not neo4j_password:
            raise ValueError("NEO4J_PASSWORD 未配置")

        self.client = Graphiti(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
        )

    def _call_with_retry(
        self,
        func: Callable[[], T],
        operation_name: str,
        max_retries: int = 3,
        initial_delay: float = 2.0,
    ) -> T:
        """带重试机制的调用封装。

        Args:
            func: 要执行的无参可调用对象。
            operation_name: 操作名称，用于日志。
            max_retries: 最大重试次数（默认 3）。
            initial_delay: 初始延迟秒数（指数退避）。

        Returns:
            func() 的返回值。

        Raises:
            最后一次异常。
        """
        last_exception: Optional[Exception] = None
        delay = initial_delay

        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Graphiti {operation_name} 第 {attempt + 1} 次尝试失败: "
                        f"{str(e)[:100]}, {delay:.1f}秒后重试..."
                    )
                    time.sleep(delay)
                    delay *= 2  # 指数退避
                else:
                    logger.error(
                        f"Graphiti {operation_name} 在 {max_retries} 次尝试后仍失败: {str(e)}"
                    )

        raise last_exception  # type: ignore[misc]

    def get_all_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        """获取图谱的所有节点（分页获取）。

        Args:
            graph_id: 图谱分组 ID。

        Returns:
            节点字典列表，每项包含 uuid、name、labels、summary、attributes。
        """
        logger.info(f"获取图谱 {graph_id} 的所有节点...")

        raw_nodes = fetch_all_nodes(self.client, graph_id)
        nodes_data = [_neo4j_node_to_dict(n) for n in raw_nodes]

        logger.info(f"共获取 {len(nodes_data)} 个节点")
        return nodes_data

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        """获取图谱的所有边（分页获取）。

        Args:
            graph_id: 图谱分组 ID。

        Returns:
            边字典列表，每项包含 uuid、name、fact、source_node_uuid、
            target_node_uuid、attributes。
        """
        logger.info(f"获取图谱 {graph_id} 的所有边...")

        raw_edges = fetch_all_edges(self.client, graph_id)
        edges_data = [_neo4j_edge_to_dict(e) for e in raw_edges]

        logger.info(f"共获取 {len(edges_data)} 条边")
        return edges_data

    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        """获取指定节点的所有相关边（直接从 Neo4j 查询，含重试机制）。

        Args:
            node_uuid: 节点 UUID。

        Returns:
            边字典列表；查询失败时返回空列表。
        """
        try:
            from ..utils.zep_paging import _run, _fetch_edges_page  # local import to avoid circularity

            # Fetch both incoming and outgoing edges for this specific node.
            async def _query_node_edges(graphiti: Graphiti, uuid: str) -> List[Any]:
                async with graphiti.driver.session() as session:
                    result = await session.run(
                        "MATCH (a)-[e:RELATES_TO]-(b) "
                        "WHERE a.uuid = $uuid OR b.uuid = $uuid "
                        "RETURN e",
                        uuid=uuid,
                    )
                    return [rec["e"] async for rec in result]

            raw_edges = self._call_with_retry(
                func=lambda: run_async(_query_node_edges(self.client, node_uuid)),
                operation_name=f"获取节点边(node={node_uuid[:8]}...)",
            )

            return [_neo4j_edge_to_dict(e) for e in raw_edges]

        except Exception as e:
            logger.warning(f"获取节点 {node_uuid} 的边失败: {str(e)}")
            return []

    def filter_defined_entities(
        self,
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True,
    ) -> FilteredEntities:
        """筛选出符合预定义实体类型的节点。

        筛选逻辑：
        - 节点的 Labels 只包含 "Entity"/"Node" → 跳过。
        - Labels 中存在其他自定义标签 → 保留。

        Args:
            graph_id: 图谱分组 ID。
            defined_entity_types: 预定义实体类型列表（可选）。
                若提供，只保留匹配类型的节点。
            enrich_with_edges: 是否为每个实体附加相关边和关联节点信息。

        Returns:
            FilteredEntities 对象。
        """
        logger.info(f"开始筛选图谱 {graph_id} 的实体...")

        all_nodes = self.get_all_nodes(graph_id)
        total_count = len(all_nodes)

        all_edges = self.get_all_edges(graph_id) if enrich_with_edges else []

        node_map: Dict[str, Dict[str, Any]] = {n["uuid"]: n for n in all_nodes}

        filtered_entities: List[EntityNode] = []
        entity_types_found: Set[str] = set()

        for node in all_nodes:
            labels = node.get("labels", [])

            custom_labels = [lb for lb in labels if lb not in ("Entity", "Node")]

            if not custom_labels:
                continue

            if defined_entity_types:
                matching_labels = [lb for lb in custom_labels if lb in defined_entity_types]
                if not matching_labels:
                    continue
                entity_type = matching_labels[0]
            else:
                entity_type = custom_labels[0]

            entity_types_found.add(entity_type)

            entity = EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=labels,
                summary=node["summary"],
                attributes=node["attributes"],
            )

            if enrich_with_edges:
                related_edges: List[Dict[str, Any]] = []
                related_node_uuids: Set[str] = set()

                for edge in all_edges:
                    if edge["source_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "outgoing",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "target_node_uuid": edge["target_node_uuid"],
                        })
                        related_node_uuids.add(edge["target_node_uuid"])
                    elif edge["target_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "incoming",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "source_node_uuid": edge["source_node_uuid"],
                        })
                        related_node_uuids.add(edge["source_node_uuid"])

                entity.related_edges = related_edges

                related_nodes: List[Dict[str, Any]] = []
                for related_uuid in related_node_uuids:
                    if related_uuid in node_map:
                        rn = node_map[related_uuid]
                        related_nodes.append({
                            "uuid": rn["uuid"],
                            "name": rn["name"],
                            "labels": rn["labels"],
                            "summary": rn.get("summary", ""),
                        })

                entity.related_nodes = related_nodes

            filtered_entities.append(entity)

        logger.info(
            f"筛选完成: 总节点 {total_count}, 符合条件 {len(filtered_entities)}, "
            f"实体类型: {entity_types_found}"
        )

        return FilteredEntities(
            entities=filtered_entities,
            entity_types=entity_types_found,
            total_count=total_count,
            filtered_count=len(filtered_entities),
        )

    def get_entity_with_context(
        self,
        graph_id: str,
        entity_uuid: str,
    ) -> Optional[EntityNode]:
        """获取单个实体及其完整上下文（边和关联节点，含重试机制）。

        Args:
            graph_id: 图谱分组 ID。
            entity_uuid: 实体 UUID。

        Returns:
            EntityNode，或 None（节点不存在或查询失败时）。
        """
        try:
            raw_node = self._call_with_retry(
                func=lambda: fetch_node_by_uuid(self.client, entity_uuid),
                operation_name=f"获取节点详情(uuid={entity_uuid[:8]}...)",
            )

            if raw_node is None:
                return None

            node = _neo4j_node_to_dict(raw_node)

            edges = self.get_node_edges(entity_uuid)

            all_nodes = self.get_all_nodes(graph_id)
            node_map: Dict[str, Dict[str, Any]] = {n["uuid"]: n for n in all_nodes}

            related_edges: List[Dict[str, Any]] = []
            related_node_uuids: Set[str] = set()

            for edge in edges:
                if edge["source_node_uuid"] == entity_uuid:
                    related_edges.append({
                        "direction": "outgoing",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "target_node_uuid": edge["target_node_uuid"],
                    })
                    related_node_uuids.add(edge["target_node_uuid"])
                else:
                    related_edges.append({
                        "direction": "incoming",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "source_node_uuid": edge["source_node_uuid"],
                    })
                    related_node_uuids.add(edge["source_node_uuid"])

            related_nodes: List[Dict[str, Any]] = []
            for related_uuid in related_node_uuids:
                if related_uuid in node_map:
                    rn = node_map[related_uuid]
                    related_nodes.append({
                        "uuid": rn["uuid"],
                        "name": rn["name"],
                        "labels": rn["labels"],
                        "summary": rn.get("summary", ""),
                    })

            return EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=node["labels"],
                summary=node["summary"],
                attributes=node["attributes"],
                related_edges=related_edges,
                related_nodes=related_nodes,
            )

        except Exception as e:
            logger.error(f"获取实体 {entity_uuid} 失败: {str(e)}")
            return None

    def get_entities_by_type(
        self,
        graph_id: str,
        entity_type: str,
        enrich_with_edges: bool = True,
    ) -> List[EntityNode]:
        """获取指定类型的所有实体。

        Args:
            graph_id: 图谱分组 ID。
            entity_type: 实体类型标签（如 "Student"、"PublicFigure" 等）。
            enrich_with_edges: 是否获取相关边信息。

        Returns:
            EntityNode 列表。
        """
        result = self.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=[entity_type],
            enrich_with_edges=enrich_with_edges,
        )
        return result.entities
