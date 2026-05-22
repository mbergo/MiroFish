"""
图谱记忆更新服务
将模拟中的Agent活动动态更新到Graphiti图谱中
"""

import asyncio
import time
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from queue import Queue, Empty
from uuid import uuid4
from ..utils.async_runner import run_async

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from ..config import Config
from ..utils.logger import get_logger
from ..utils.locale import get_locale, set_locale

logger = get_logger('mirofish.zep_graph_memory_updater')




@dataclass
class AgentActivity:
    """Agent活动记录"""
    platform: str           # twitter / reddit
    agent_id: int
    agent_name: str
    action_type: str        # CREATE_POST, LIKE_POST, etc.
    action_args: Dict[str, Any]
    round_num: int
    timestamp: str

    def to_episode_text(self) -> str:
        """将活动转换为可以发送给图谱的文本描述

        采用自然语言描述格式，让图谱能够从中提取实体和关系。
        不添加模拟相关的前缀，避免误导图谱更新。
        """
        action_descriptions = {
            "CREATE_POST": self._describe_create_post,
            "LIKE_POST": self._describe_like_post,
            "DISLIKE_POST": self._describe_dislike_post,
            "REPOST": self._describe_repost,
            "QUOTE_POST": self._describe_quote_post,
            "FOLLOW": self._describe_follow,
            "CREATE_COMMENT": self._describe_create_comment,
            "LIKE_COMMENT": self._describe_like_comment,
            "DISLIKE_COMMENT": self._describe_dislike_comment,
            "SEARCH_POSTS": self._describe_search,
            "SEARCH_USER": self._describe_search_user,
            "MUTE": self._describe_mute,
        }

        describe_func = action_descriptions.get(self.action_type, self._describe_generic)
        description = describe_func()

        # 直接返回 "agent名称: 活动描述" 格式，不添加模拟前缀
        return f"{self.agent_name}: {description}"

    def _describe_create_post(self) -> str:
        content = self.action_args.get("content", "")
        if content:
            return f"发布了一条帖子：「{content}」"
        return "发布了一条帖子"

    def _describe_like_post(self) -> str:
        """点赞帖子 - 包含帖子原文和作者信息"""
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")

        if post_content and post_author:
            return f"点赞了{post_author}的帖子：「{post_content}」"
        elif post_content:
            return f"点赞了一条帖子：「{post_content}」"
        elif post_author:
            return f"点赞了{post_author}的一条帖子"
        return "点赞了一条帖子"

    def _describe_dislike_post(self) -> str:
        """踩帖子 - 包含帖子原文和作者信息"""
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")

        if post_content and post_author:
            return f"踩了{post_author}的帖子：「{post_content}」"
        elif post_content:
            return f"踩了一条帖子：「{post_content}」"
        elif post_author:
            return f"踩了{post_author}的一条帖子"
        return "踩了一条帖子"

    def _describe_repost(self) -> str:
        """转发帖子 - 包含原帖内容和作者信息"""
        original_content = self.action_args.get("original_content", "")
        original_author = self.action_args.get("original_author_name", "")

        if original_content and original_author:
            return f"转发了{original_author}的帖子：「{original_content}」"
        elif original_content:
            return f"转发了一条帖子：「{original_content}」"
        elif original_author:
            return f"转发了{original_author}的一条帖子"
        return "转发了一条帖子"

    def _describe_quote_post(self) -> str:
        """引用帖子 - 包含原帖内容、作者信息和引用评论"""
        original_content = self.action_args.get("original_content", "")
        original_author = self.action_args.get("original_author_name", "")
        quote_content = self.action_args.get("quote_content", "") or self.action_args.get("content", "")

        base = ""
        if original_content and original_author:
            base = f"引用了{original_author}的帖子「{original_content}」"
        elif original_content:
            base = f"引用了一条帖子「{original_content}」"
        elif original_author:
            base = f"引用了{original_author}的一条帖子"
        else:
            base = "引用了一条帖子"

        if quote_content:
            base += f"，并评论道：「{quote_content}」"
        return base

    def _describe_follow(self) -> str:
        """关注用户 - 包含被关注用户的名称"""
        target_user_name = self.action_args.get("target_user_name", "")

        if target_user_name:
            return f"关注了用户「{target_user_name}」"
        return "关注了一个用户"

    def _describe_create_comment(self) -> str:
        """发表评论 - 包含评论内容和所评论的帖子信息"""
        content = self.action_args.get("content", "")
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")

        if content:
            if post_content and post_author:
                return f"在{post_author}的帖子「{post_content}」下评论道：「{content}」"
            elif post_content:
                return f"在帖子「{post_content}」下评论道：「{content}」"
            elif post_author:
                return f"在{post_author}的帖子下评论道：「{content}」"
            return f"评论道：「{content}」"
        return "发表了评论"

    def _describe_like_comment(self) -> str:
        """点赞评论 - 包含评论内容和作者信息"""
        comment_content = self.action_args.get("comment_content", "")
        comment_author = self.action_args.get("comment_author_name", "")

        if comment_content and comment_author:
            return f"点赞了{comment_author}的评论：「{comment_content}」"
        elif comment_content:
            return f"点赞了一条评论：「{comment_content}」"
        elif comment_author:
            return f"点赞了{comment_author}的一条评论"
        return "点赞了一条评论"

    def _describe_dislike_comment(self) -> str:
        """踩评论 - 包含评论内容和作者信息"""
        comment_content = self.action_args.get("comment_content", "")
        comment_author = self.action_args.get("comment_author_name", "")

        if comment_content and comment_author:
            return f"踩了{comment_author}的评论：「{comment_content}」"
        elif comment_content:
            return f"踩了一条评论：「{comment_content}」"
        elif comment_author:
            return f"踩了{comment_author}的一条评论"
        return "踩了一条评论"

    def _describe_search(self) -> str:
        """搜索帖子 - 包含搜索关键词"""
        query = self.action_args.get("query", "") or self.action_args.get("keyword", "")
        return f"搜索了「{query}」" if query else "进行了搜索"

    def _describe_search_user(self) -> str:
        """搜索用户 - 包含搜索关键词"""
        query = self.action_args.get("query", "") or self.action_args.get("username", "")
        return f"搜索了用户「{query}」" if query else "搜索了用户"

    def _describe_mute(self) -> str:
        """屏蔽用户 - 包含被屏蔽用户的名称"""
        target_user_name = self.action_args.get("target_user_name", "")

        if target_user_name:
            return f"屏蔽了用户「{target_user_name}」"
        return "屏蔽了一个用户"

    def _describe_generic(self) -> str:
        return f"执行了{self.action_type}操作"


class ZepGraphMemoryUpdater:
    """图谱记忆更新器 (Graphiti / Neo4j backend)

    监控模拟的 actions 日志，将新的 agent 活动实时写入 Graphiti 知识图谱。
    按平台分组，每累积 BATCH_SIZE 条活动后合并为单条 episode 批量写入。

    所有有意义的行为都会被写入图谱，action_args 中包含完整的上下文信息：
    - 点赞/踩的帖子原文
    - 转发/引用的帖子原文
    - 关注/屏蔽的用户名
    - 点赞/踩的评论原文
    """

    # 批量发送大小（每个平台累积多少条后发送）
    BATCH_SIZE = 5

    # 平台名称映射（用于控制台显示）
    PLATFORM_DISPLAY_NAMES: Dict[str, str] = {
        'twitter': '世界1',
        'reddit': '世界2',
    }

    # 发送间隔（秒），避免请求过快
    SEND_INTERVAL = 0.5

    # 重试配置
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # 秒

    def __init__(self, graph_id: str) -> None:
        """初始化更新器

        Args:
            graph_id: Graphiti group_id，用于隔离不同模拟的图谱数据。
        """
        self.graph_id = graph_id

        if not Config.NEO4J_URI:
            raise ValueError("NEO4J_URI 未配置")
        if not Config.NEO4J_USER:
            raise ValueError("NEO4J_USER 未配置")
        if not Config.NEO4J_PASSWORD:
            raise ValueError("NEO4J_PASSWORD 未配置")

        self.client = Graphiti(
            uri=Config.NEO4J_URI,
            user=Config.NEO4J_USER,
            password=Config.NEO4J_PASSWORD,
        )

        # 活动队列
        self._activity_queue: Queue = Queue()

        # 按平台分组的活动缓冲区（每个平台各自累积到 BATCH_SIZE 后批量发送）
        self._platform_buffers: Dict[str, List[AgentActivity]] = {
            'twitter': [],
            'reddit': [],
        }
        self._buffer_lock = threading.Lock()

        # 控制标志
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        # 统计
        self._total_activities = 0   # 实际添加到队列的活动数
        self._total_sent = 0         # 成功发送到图谱的批次数
        self._total_items_sent = 0   # 成功发送到图谱的活动条数
        self._failed_count = 0       # 发送失败的批次数
        self._skipped_count = 0      # 被过滤跳过的活动数（DO_NOTHING）

        logger.info(
            "ZepGraphMemoryUpdater 初始化完成: graph_id=%s, batch_size=%d",
            graph_id,
            self.BATCH_SIZE,
        )

    def _get_platform_display_name(self, platform: str) -> str:
        """获取平台的显示名称"""
        return self.PLATFORM_DISPLAY_NAMES.get(platform.lower(), platform)

    def start(self) -> None:
        """启动后台工作线程"""
        if self._running:
            return

        # Capture locale before spawning background thread
        current_locale = get_locale()

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(current_locale,),
            daemon=True,
            name=f"GraphitiMemoryUpdater-{self.graph_id[:8]}",
        )
        self._worker_thread.start()
        logger.info("ZepGraphMemoryUpdater 已启动: graph_id=%s", self.graph_id)

    def stop(self) -> None:
        """停止后台工作线程"""
        self._running = False

        # 发送剩余的活动
        self._flush_remaining()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10)

        logger.info(
            "ZepGraphMemoryUpdater 已停止: graph_id=%s, "
            "total_activities=%d, batches_sent=%d, items_sent=%d, "
            "failed=%d, skipped=%d",
            self.graph_id,
            self._total_activities,
            self._total_sent,
            self._total_items_sent,
            self._failed_count,
            self._skipped_count,
        )

    def add_activity(self, activity: AgentActivity) -> None:
        """添加一个 agent 活动到队列

        所有有意义的行为都会被添加到队列，包括：
        CREATE_POST, CREATE_COMMENT, QUOTE_POST, SEARCH_POSTS, SEARCH_USER,
        LIKE_POST, DISLIKE_POST, REPOST, FOLLOW, MUTE, LIKE_COMMENT,
        DISLIKE_COMMENT.

        DO_NOTHING 类型的活动会被静默丢弃。

        Args:
            activity: Agent 活动记录。
        """
        if activity.action_type == "DO_NOTHING":
            self._skipped_count += 1
            return

        self._activity_queue.put(activity)
        self._total_activities += 1
        logger.debug(
            "添加活动到图谱队列: %s - %s", activity.agent_name, activity.action_type
        )

    def add_activity_from_dict(self, data: Dict[str, Any], platform: str) -> None:
        """从字典数据添加活动

        Args:
            data: 从 actions.jsonl 解析的字典数据。
            platform: 平台名称 (twitter / reddit)。
        """
        # 跳过事件类型的条目
        if "event_type" in data:
            return

        activity = AgentActivity(
            platform=platform,
            agent_id=data.get("agent_id", 0),
            agent_name=data.get("agent_name", ""),
            action_type=data.get("action_type", ""),
            action_args=data.get("action_args", {}),
            round_num=data.get("round", 0),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )

        self.add_activity(activity)

    def _worker_loop(self, locale: str = 'zh') -> None:
        """后台工作循环 - 按平台批量发送活动到图谱"""
        set_locale(locale)
        while self._running or not self._activity_queue.empty():
            try:
                try:
                    activity = self._activity_queue.get(timeout=1)

                    platform = activity.platform.lower()
                    with self._buffer_lock:
                        if platform not in self._platform_buffers:
                            self._platform_buffers[platform] = []
                        self._platform_buffers[platform].append(activity)

                        # 检查该平台是否达到批量大小
                        if len(self._platform_buffers[platform]) >= self.BATCH_SIZE:
                            batch = self._platform_buffers[platform][: self.BATCH_SIZE]
                            self._platform_buffers[platform] = self._platform_buffers[platform][self.BATCH_SIZE :]
                            # 释放锁后再发送
                            self._send_batch_activities(batch, platform)
                            time.sleep(self.SEND_INTERVAL)

                except Empty:
                    pass

            except Exception as e:
                logger.error("工作循环异常: %s", e)
                time.sleep(1)

    def _send_batch_activities(
        self, activities: List[AgentActivity], platform: str
    ) -> None:
        """批量发送活动到 Graphiti 图谱（合并为一条 episode 文本）

        Args:
            activities: Agent 活动列表。
            platform: 平台名称。
        """
        if not activities:
            return

        # 将多条活动合并为一条文本，用换行分隔
        episode_texts = [activity.to_episode_text() for activity in activities]
        combined_text = "\n".join(episode_texts)

        # 使用批次中第一条活动的 agent 信息构建 episode 元数据
        first = activities[0]
        episode_name = (
            f"activity_{first.agent_id}_{first.round_num}_{uuid4()}"
        )
        source_description = f"Agent {first.agent_name} on {platform}"

        for attempt in range(self.MAX_RETRIES):
            try:
                run_async(
                    self.client.add_episode(
                        name=episode_name,
                        episode_body=combined_text,
                        source=EpisodeType.text,
                        source_description=source_description,
                        group_id=self.graph_id,
                        reference_time=datetime.now(),
                    )
                )

                self._total_sent += 1
                self._total_items_sent += len(activities)
                display_name = self._get_platform_display_name(platform)
                logger.info(
                    "成功批量发送 %d 条%s活动到图谱 %s",
                    len(activities),
                    display_name,
                    self.graph_id,
                )
                logger.debug("批量内容预览: %s...", combined_text[:200])
                return

            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(
                        "批量发送到图谱失败 (尝试 %d/%d): %s",
                        attempt + 1,
                        self.MAX_RETRIES,
                        e,
                    )
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(
                        "批量发送到图谱失败，已重试 %d 次: %s", self.MAX_RETRIES, e
                    )
                    self._failed_count += 1

    def _flush_remaining(self) -> None:
        """发送队列和缓冲区中剩余的活动"""
        # 首先处理队列中剩余的活动，添加到缓冲区
        while not self._activity_queue.empty():
            try:
                activity = self._activity_queue.get_nowait()
                platform = activity.platform.lower()
                with self._buffer_lock:
                    if platform not in self._platform_buffers:
                        self._platform_buffers[platform] = []
                    self._platform_buffers[platform].append(activity)
            except Empty:
                break

        # 然后发送各平台缓冲区中剩余的活动（即使不足 BATCH_SIZE 条）
        # Copy buffer contents under lock, then release before doing network I/O.
        with self._buffer_lock:
            buffers_to_flush = dict(self._platform_buffers)
            self._platform_buffers.clear()
        # Lock released — now send without holding it
        for platform, buffer in buffers_to_flush.items():
            if buffer:
                display_name = self._get_platform_display_name(platform)
                logger.info(
                    "发送%s平台剩余的 %d 条活动", display_name, len(buffer)
                )
                self._send_batch_activities(buffer, platform)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._buffer_lock:
            buffer_sizes = {p: len(b) for p, b in self._platform_buffers.items()}

        return {
            "graph_id": self.graph_id,
            "batch_size": self.BATCH_SIZE,
            "total_activities": self._total_activities,
            "batches_sent": self._total_sent,
            "items_sent": self._total_items_sent,
            "failed_count": self._failed_count,
            "skipped_count": self._skipped_count,
            "queue_size": self._activity_queue.qsize(),
            "buffer_sizes": buffer_sizes,
            "running": self._running,
        }


class ZepGraphMemoryManager:
    """管理多个模拟的图谱记忆更新器

    每个模拟可以有自己的 :class:`ZepGraphMemoryUpdater` 实例，由
    ``simulation_id`` 索引。
    """

    _updaters: Dict[str, ZepGraphMemoryUpdater] = {}
    _lock = threading.Lock()

    @classmethod
    def create_updater(
        cls, simulation_id: str, graph_id: str
    ) -> ZepGraphMemoryUpdater:
        """为模拟创建图谱记忆更新器

        如果该 simulation_id 已存在一个运行中的更新器，它会被停止后替换。

        Args:
            simulation_id: 模拟 ID。
            graph_id: Graphiti group_id。

        Returns:
            已启动的 :class:`ZepGraphMemoryUpdater` 实例。
        """
        with cls._lock:
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()

            updater = ZepGraphMemoryUpdater(graph_id)
            updater.start()
            cls._updaters[simulation_id] = updater

            logger.info(
                "创建图谱记忆更新器: simulation_id=%s, graph_id=%s",
                simulation_id,
                graph_id,
            )
            return updater

    @classmethod
    def get_updater(cls, simulation_id: str) -> Optional[ZepGraphMemoryUpdater]:
        """获取模拟的更新器"""
        return cls._updaters.get(simulation_id)

    @classmethod
    def stop_updater(cls, simulation_id: str) -> None:
        """停止并移除模拟的更新器"""
        with cls._lock:
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()
                del cls._updaters[simulation_id]
                logger.info("已停止图谱记忆更新器: simulation_id=%s", simulation_id)

    # 防止 stop_all 重复调用的标志
    _stop_all_done = False

    @classmethod
    def stop_all(cls) -> None:
        """停止所有更新器"""
        if cls._stop_all_done:
            return
        cls._stop_all_done = True

        with cls._lock:
            if cls._updaters:
                for simulation_id, updater in list(cls._updaters.items()):
                    try:
                        updater.stop()
                    except Exception as e:
                        logger.error(
                            "停止更新器失败: simulation_id=%s, error=%s",
                            simulation_id,
                            e,
                        )
                cls._updaters.clear()
            logger.info("已停止所有图谱记忆更新器")

    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        """获取所有更新器的统计信息"""
        return {
            sim_id: updater.get_stats()
            for sim_id, updater in cls._updaters.items()
        }
