"""
查询知识Command组件

提供知识库查询功能，支持段落和关系查询。
"""

import time
import re
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path

from src.common.logger import get_logger
from src.plugin_system.apis import person_api
from src.plugin_system.base.base_command import BaseCommand
from src.chat.message_receive.message import MessageRecv

# 导入核心模块
from ...core import (
    DualPathRetriever,
    RetrievalStrategy,
    DualPathRetrieverConfig,
    TemporalQueryOptions,
    DynamicThresholdFilter,
    ThresholdMethod,
    ThresholdConfig,
    SparseBM25Config,
    FusionConfig,
    RelationIntentConfig,
)
from ...core.utils.time_parser import parse_query_time_range
from ...core.utils.person_profile_service import PersonProfileService
from ...core.utils.relation_query import parse_relation_query_spec

logger = get_logger("A_Memorix.QueryCommand")


class QueryCommand(BaseCommand):
    """查询知识Command

    功能：
    - 双路检索查询
    - 实体查询
    - 关系查询
    - 统计信息查询
    """

    # Command基本信息
    command_name = "query"
    command_description = "查询知识库，支持检索、实体、关系和统计信息"
    command_pattern = r"^\/query(?:\s+(?P<mode>\w+))?(?:\s+(?P<content>.+))?$"

    def __init__(self, message: MessageRecv, plugin_config: Optional[dict] = None):
        """初始化查询Command"""
        super().__init__(message, plugin_config)

        logger.info(f"QueryCommand 初始化开始")
        logger.info(f"  plugin_config keys: {list(self.plugin_config.keys()) if self.plugin_config else 'None'}")

        # 获取存储实例 (优先从配置获取，兜底从插件实例获取)
        self.vector_store = self.plugin_config.get("vector_store")
        self.graph_store = self.plugin_config.get("graph_store")
        self.metadata_store = self.plugin_config.get("metadata_store")
        self.embedding_manager = self.plugin_config.get("embedding_manager")
        self.sparse_index = self.plugin_config.get("sparse_index")

        logger.info(f"  从 plugin_config 获取: vector_store={self.vector_store is not None}, "
                   f"graph_store={self.graph_store is not None}, "
                   f"metadata_store={self.metadata_store is not None}, "
                   f"embedding_manager={self.embedding_manager is not None}")

        # 兜底逻辑：如果配置中没有存储实例，尝试直接从插件系统获取
        # 使用 is not None 检查，因为空对象可能布尔值为 False
        if not all([
            self.vector_store is not None,
            self.graph_store is not None,
            self.metadata_store is not None,
            self.embedding_manager is not None
        ]):
            logger.warning(f"  配置不完整，尝试从插件实例获取...")
            try:
                from ...plugin import A_MemorixPlugin
                instances = A_MemorixPlugin.get_storage_instances()
                logger.info(f"  get_storage_instances() 返回: {list(instances.keys()) if instances else 'empty dict'}")
                
                if instances:
                    self.vector_store = self.vector_store or instances.get("vector_store")
                    self.graph_store = self.graph_store or instances.get("graph_store")
                    self.metadata_store = self.metadata_store or instances.get("metadata_store")
                    self.embedding_manager = self.embedding_manager or instances.get("embedding_manager")
                    self.sparse_index = self.sparse_index or instances.get("sparse_index")
                    
                    logger.info(f"  兜底后: vector_store={self.vector_store is not None}, "
                               f"graph_store={self.graph_store is not None}, "
                               f"metadata_store={self.metadata_store is not None}, "
                               f"embedding_manager={self.embedding_manager is not None}")
                else:
                    logger.error(f"  get_storage_instances() 返回空字典！")
            except Exception as e:
                logger.error(f"  兜底逻辑异常: {e}")
                import traceback
                traceback.print_exc()

        # 初始化检索器
        self.retriever: Optional[DualPathRetriever] = None
        self.threshold_filter: Optional[DynamicThresholdFilter] = None

        # 设置日志前缀
        if self.message and self.message.chat_stream:
            self.log_prefix = f"[QueryCommand-{self.message.chat_stream.stream_id}]"
        else:
            self.log_prefix = "[QueryCommand]"

        # 初始化组件
        self._initialize_components()

    @property
    def debug_enabled(self) -> bool:
        """检查是否启用了调试模式"""
        advanced = self.plugin_config.get("advanced", {})
        if isinstance(advanced, dict):
            return advanced.get("debug", False)
        return self.plugin_config.get("debug", False)

    def _initialize_components(self) -> None:
        """初始化检索和过滤组件"""
        try:
            # 检查存储是否可用 (使用 is not None 而非布尔值，因为空对象可能为 False)
            if not all([
                self.vector_store is not None,
                self.graph_store is not None,
                self.metadata_store is not None,
                self.embedding_manager is not None
            ]):
                logger.warning(f"{self.log_prefix} 存储组件未完全初始化")
                return

            # 创建检索器配置
            sparse_cfg_raw = self.get_config("retrieval.sparse", {}) or {}
            if not isinstance(sparse_cfg_raw, dict):
                sparse_cfg_raw = {}
            fusion_cfg_raw = self.get_config("retrieval.fusion", {}) or {}
            if not isinstance(fusion_cfg_raw, dict):
                fusion_cfg_raw = {}
            relation_intent_cfg_raw = self.get_config("retrieval.search.relation_intent", {}) or {}
            if not isinstance(relation_intent_cfg_raw, dict):
                relation_intent_cfg_raw = {}
            try:
                sparse_cfg = SparseBM25Config(**sparse_cfg_raw)
            except Exception as e:
                logger.warning(f"{self.log_prefix} sparse 配置非法，回退默认: {e}")
                sparse_cfg = SparseBM25Config()
            try:
                fusion_cfg = FusionConfig(**fusion_cfg_raw)
            except Exception as e:
                logger.warning(f"{self.log_prefix} fusion 配置非法，回退默认: {e}")
                fusion_cfg = FusionConfig()
            try:
                relation_intent_cfg = RelationIntentConfig(**relation_intent_cfg_raw)
            except Exception as e:
                logger.warning(f"{self.log_prefix} relation_intent 配置非法，回退默认: {e}")
                relation_intent_cfg = RelationIntentConfig()
            config = DualPathRetrieverConfig(
                top_k_paragraphs=self.get_config("retrieval.top_k_paragraphs", 20),
                top_k_relations=self.get_config("retrieval.top_k_relations", 10),
                top_k_final=self.get_config("retrieval.top_k_final", 10),
                alpha=self.get_config("retrieval.alpha", 0.5),
                enable_ppr=self.get_config("retrieval.enable_ppr", True),
                ppr_alpha=self.get_config("retrieval.ppr_alpha", 0.85),
                ppr_concurrency_limit=self.get_config("retrieval.ppr_concurrency_limit", 4),
                enable_parallel=self.get_config("retrieval.enable_parallel", True),
                retrieval_strategy=RetrievalStrategy.DUAL_PATH,
                debug=self.debug_enabled,
                sparse=sparse_cfg,
                fusion=fusion_cfg,
                relation_intent=relation_intent_cfg,
            )

            # 创建检索器
            self.retriever = DualPathRetriever(
                vector_store=self.vector_store,
                graph_store=self.graph_store,
                metadata_store=self.metadata_store,
                embedding_manager=self.embedding_manager,
                sparse_index=self.sparse_index,
                config=config,
            )

            # 创建阈值过滤器
            threshold_config = ThresholdConfig(
                method=ThresholdMethod.ADAPTIVE,
                min_threshold=self.get_config("threshold.min_threshold", 0.3),
                max_threshold=self.get_config("threshold.max_threshold", 0.95),
                percentile=self.get_config("threshold.percentile", 75.0),
                std_multiplier=self.get_config("threshold.std_multiplier", 1.5),
                min_results=self.get_config("threshold.min_results", 3),
                enable_auto_adjust=self.get_config("threshold.enable_auto_adjust", True),
            )

            self.threshold_filter = DynamicThresholdFilter(threshold_config)

            logger.info(f"{self.log_prefix} 查询组件初始化完成")

        except Exception as e:
            logger.error(f"{self.log_prefix} 组件初始化失败: {e}")

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        """执行查询命令

        Returns:
            Tuple[bool, Optional[str], int]: (是否成功, 回复消息, 拦截级别)
        """
        # 检查组件是否初始化
        if not self.retriever:
            error_msg = "❌ 查询组件未初始化"
            return False, error_msg, 1

        # 获取匹配的参数
        mode = self.matched_groups.get("mode", "search")
        content = self.matched_groups.get("content", "")

        # 如果没有内容，显示帮助
        if not content and mode not in ["stats", "help"]:
            help_msg = self._get_help_message()
            return True, help_msg, 1

        logger.info(f"{self.log_prefix} 执行查询: mode={mode}, content='{content}'")

        try:
            # 根据模式执行查询
            if mode == "search" or mode == "s":
                success, result = await self._query_search(content)
            elif mode == "time" or mode == "t":
                success, result = await self._query_time(content)
            elif mode == "entity" or mode == "e":
                success, result = await self._query_entity(content)
            elif mode == "relation" or mode == "r":
                success, result = await self._query_relation(content)
            elif mode == "person" or mode == "p":
                success, result = await self._query_person(content)
            elif mode == "stats":
                success, result = self._query_stats()
            elif mode == "help":
                success, result = True, self._get_help_message()
            else:
                success, result = False, f"❌ 未知的查询模式: {mode}"

            # 显式回消息到当前对话流，避免仅返回结果导致输出丢失或落到错误链路。
            if result:
                try:
                    await self.send_text(result)
                except Exception as send_err:
                    logger.warning(f"{self.log_prefix} 发送查询结果失败: {send_err}")

            return success, result, 1

        except Exception as e:
            error_msg = f"❌ 查询失败: {str(e)}"
            logger.error(f"{self.log_prefix} {error_msg}")
            return False, error_msg, 1

    async def _query_search(self, query: str) -> Tuple[bool, str]:
        """执行检索查询

        Args:
            query: 查询文本

        Returns:
            Tuple[bool, str]: (是否成功, 结果消息)
        """
        start_time = time.time()

        # 执行检索（异步调用）
        results = await self.retriever.retrieve(query, top_k=10)

        if self.debug_enabled:
            logger.info(f"{self.log_prefix} [DEBUG] 原始检索结果数量: {len(results)}")
            for i, r in enumerate(results):
                logger.info(f"{self.log_prefix} [DEBUG] Result {i}: type={r.result_type}, score={r.score:.4f}, hash={r.hash_value}")

        # 应用阈值过滤
        if self.threshold_filter:
            results = self.threshold_filter.filter(results)
            if self.debug_enabled:
                logger.info(f"{self.log_prefix} [DEBUG] 过滤后结果数量: {len(results)}")

        elapsed = time.time() - start_time

        # 格式化结果
        if not results:
            return True, f"🔍 未找到相关内容（耗时: {elapsed*1000:.1f}ms）"

        # 按类型分组
        paragraphs = [r for r in results if r.result_type == "paragraph"]
        relations = [r for r in results if r.result_type == "relation"]

        # 构建响应
        lines = [
            f"🔍 检索结果（查询: '{query}'，耗时: {elapsed*1000:.1f}ms）",
            "",
        ]

        if paragraphs:
            lines.append("📄 匹配的段落：")
            for i, result in enumerate(paragraphs[:5], 1):
                score_pct = result.score * 100
                content = result.content[:80] + "..." if len(result.content) > 80 else result.content
                lines.append(f"  {i}. [{score_pct:.1f}%] {content}")
            lines.append("")

        if relations:
            lines.append("🔗 匹配的关系：")
            for i, result in enumerate(relations[:5], 1):
                score_pct = result.score * 100
                subject = result.metadata.get("subject", "")
                predicate = result.metadata.get("predicate", "")
                obj = result.metadata.get("object", "")
                lines.append(f"  {i}. [{score_pct:.1f}%] {subject} {predicate} {obj}")
            lines.append("")

        lines.append(f"📊 共 {len(results)} 条结果（段落: {len(paragraphs)}, 关系: {len(relations)}）")

        return True, "\n".join(lines)

    def _parse_kv_args(self, raw: str) -> Dict[str, str]:
        """
        解析 k=v 参数，支持引号。
        示例: q="项目进展" from=2025/01/01 to="2025/01/31 12:00"
        """
        pattern = re.compile(r"(\w+)=((?:\"[^\"]*\")|(?:'[^']*')|(?:\S+))")
        parsed: Dict[str, str] = {}
        for match in pattern.finditer(raw):
            key = match.group(1).strip().lower()
            value = match.group(2).strip()
            if len(value) >= 2 and (
                (value[0] == '"' and value[-1] == '"')
                or (value[0] == "'" and value[-1] == "'")
            ):
                value = value[1:-1]
            parsed[key] = value.strip()
        return parsed

    async def _query_time(self, content: str) -> Tuple[bool, str]:
        """
        时序检索: /query time q=... from=... to=... person=... source=... top_k=...
        """
        if not bool(self.get_config("retrieval.temporal.enabled", True)):
            return False, "❌ 时序检索已禁用（retrieval.temporal.enabled=false）"

        args = self._parse_kv_args(content)
        query = args.get("q") or args.get("query") or ""
        time_from = args.get("from") or args.get("start")
        time_to = args.get("to") or args.get("end")
        person = args.get("person")
        source = args.get("source")

        if not time_from and not time_to:
            return False, "❌ time 模式至少需要 from/start 或 to/end 参数"

        top_k = int(self.get_config("retrieval.temporal.default_top_k", 10))
        if "top_k" in args:
            try:
                top_k = max(1, int(args["top_k"]))
            except ValueError:
                return False, "❌ top_k 必须是整数"

        try:
            ts_from, ts_to = parse_query_time_range(time_from, time_to)
        except ValueError as e:
            return False, f"❌ 时间参数错误: {e}"

        temporal = TemporalQueryOptions(
            time_from=ts_from,
            time_to=ts_to,
            person=person,
            source=source,
            allow_created_fallback=self.get_config(
                "retrieval.temporal.allow_created_fallback",
                True,
            ),
            candidate_multiplier=int(
                self.get_config("retrieval.temporal.candidate_multiplier", 8)
            ),
            max_scan=int(self.get_config("retrieval.temporal.max_scan", 1000)),
        )

        start_time = time.time()
        results = await self.retriever.retrieve(
            query=query,
            top_k=top_k,
            temporal=temporal,
        )

        # query 非空时可以应用阈值；纯 time 窗口扫描时不做阈值过滤
        if query and self.threshold_filter:
            results = self.threshold_filter.filter(results)

        elapsed = time.time() - start_time
        if not results:
            return True, f"🕒 未找到符合时间条件的内容（耗时: {elapsed*1000:.1f}ms）"

        paragraphs = [r for r in results if r.result_type == "paragraph"]
        relations = [r for r in results if r.result_type == "relation"]

        lines = [
            f"🕒 时间检索结果（query='{query or 'N/A'}'，耗时: {elapsed*1000:.1f}ms）",
            "",
        ]

        if paragraphs:
            lines.append("📄 匹配段落：")
            for i, result in enumerate(paragraphs[:top_k], 1):
                score_pct = result.score * 100
                content_text = result.content[:80] + "..." if len(result.content) > 80 else result.content
                time_meta = result.metadata.get("time_meta", {})
                s_text = time_meta.get("effective_start_text", "N/A")
                e_text = time_meta.get("effective_end_text", "N/A")
                basis = time_meta.get("match_basis", "none")
                lines.append(f"  {i}. [{score_pct:.1f}%] {content_text}")
                lines.append(f"     ⏱️ {s_text} ~ {e_text} ({basis})")
            lines.append("")

        if relations:
            lines.append("🔗 匹配关系：")
            for i, result in enumerate(relations[:top_k], 1):
                score_pct = result.score * 100
                subject = result.metadata.get("subject", "")
                predicate = result.metadata.get("predicate", "")
                obj = result.metadata.get("object", "")
                time_meta = result.metadata.get("time_meta", {})
                s_text = time_meta.get("effective_start_text", "N/A")
                e_text = time_meta.get("effective_end_text", "N/A")
                basis = time_meta.get("match_basis", "none")
                lines.append(f"  {i}. [{score_pct:.1f}%] {subject} {predicate} {obj}")
                lines.append(f"     ⏱️ {s_text} ~ {e_text} ({basis})")
            lines.append("")

        lines.append(f"📊 共 {len(results)} 条结果（段落: {len(paragraphs)}, 关系: {len(relations)}）")
        return True, "\n".join(lines)

    async def _query_entity(self, entity_name: str) -> Tuple[bool, str]:
        """查询实体信息

        Args:
            entity_name: 实体名称

        Returns:
            Tuple[bool, str]: (是否成功, 结果消息)
        """
        # 检查实体是否存在
        if not self.graph_store.has_node(entity_name):
            return False, f"❌ 实体不存在: {entity_name}"

        # 获取邻居节点
        neighbors = self.graph_store.get_neighbors(entity_name)

        if self.debug_enabled:
            logger.info(f"{self.log_prefix} [DEBUG] 实体 '{entity_name}' 邻居节点: {neighbors}")

        # 获取相关段落
        paragraphs = self.metadata_store.get_paragraphs_by_entity(entity_name)

        # 构建响应
        lines = [
            f"🏷️ 实体信息: {entity_name}",
            "",
            f"🔗 关联实体 ({len(neighbors)}):",
        ]

        if neighbors:
            for neighbor in neighbors[:10]:
                lines.append(f"  - {neighbor}")
        else:
            lines.append("  (无)")

        lines.append("")
        lines.append(f"📄 相关段落 ({len(paragraphs)}):")

        if paragraphs:
            for i, para in enumerate(paragraphs[:5], 1):
                content = para["content"][:80] + "..." if len(para["content"]) > 80 else para["content"]
                lines.append(f"  {i}. {content}")
        else:
            lines.append("  (无)")

        return True, "\n".join(lines)

    async def _query_relation(self, relation_spec: str) -> Tuple[bool, str]:
        """查询关系信息

        Args:
            relation_spec: 关系规格 (格式: subject|predicate|object 或 subject predicate)

        Returns:
            Tuple[bool, str]: (是否成功, 结果消息)
        """
        parsed = parse_relation_query_spec(relation_spec)
        if parsed.error in {"empty", "invalid_pipe_format", "invalid_arrow_format"}:
            return False, "❌ 关系格式错误，应使用: subject|predicate 或 subject|predicate|object"

        subject = parsed.subject
        predicate = parsed.predicate
        obj = parsed.object
        if not subject or not predicate:
            return False, "❌ 关系格式错误，应使用: subject|predicate 或 subject|predicate|object"

        # 查询关系
        relations = self.metadata_store.get_relations(
            subject=subject if subject else None,
            predicate=predicate if predicate else None,
            object=obj if obj else None,
        )

        # 构建响应
        lines = [
            f"🔗 关系查询结果",
            f"📌 规格: {subject} {predicate} {obj or '*' }",
            f"📊 找到 {len(relations)} 条关系",
            "",
        ]

        if relations:
            for i, rel in enumerate(relations[:10], 1):
                s = rel.get("subject", "")
                p = rel.get("predicate", "")
                o = rel.get("object", "")
                conf = rel.get("confidence", 1.0)
                lines.append(f"  {i}. {s} {p} {o} (置信度: {conf:.2f})")
        else:
            lines.append("  (无匹配结果)")

        return True, "\n".join(lines)

    async def _query_person(self, content: str) -> Tuple[bool, str]:
        """查询人物画像。

        支持：
        - /query person id=<person_id>
        - /query person person_id=<person_id>
        - /query person <人名或别名>
        """
        if not bool(self.get_config("person_profile.enabled", True)):
            return False, "❌ 人物画像功能未启用（person_profile.enabled=false）"

        args = self._parse_kv_args(content)
        query = content.strip()
        person_id = (args.get("id") or args.get("person_id") or "").strip()
        if person_id:
            query = args.get("q", "").strip() or args.get("query", "").strip() or query
        else:
            # 若未显式指定 id，优先用去掉 k=v 参数后的 query
            query = args.get("q", "").strip() or args.get("query", "").strip() or query

        service = PersonProfileService(
            metadata_store=self.metadata_store,
            graph_store=self.graph_store,
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            sparse_index=self.sparse_index,
            plugin_config=self.plugin_config,
            retriever=self.retriever,
        )

        if not person_id:
            if query:
                person_id = service.resolve_person_id(query)
            if not person_id and self.message and self.message.chat_stream:
                try:
                    platform = str(getattr(self.message.chat_stream, "platform", "") or "").strip()
                    uid = str(getattr(self.message.message_info.user_info, "user_id", "") or "").strip()
                    if platform and uid:
                        person_id = person_api.get_person_id(platform, uid)
                except Exception:
                    person_id = ""

        if not person_id:
            return False, "❌ 未能解析人物ID，请使用 /query person id=<person_id> 或提供可识别的人名/别名"

        ttl_minutes = float(self.get_config("person_profile.profile_ttl_minutes", 360))
        profile = await service.query_person_profile(
            person_id=person_id,
            person_keyword=query,
            top_k=12,
            ttl_seconds=max(60.0, ttl_minutes * 60.0),
            force_refresh=False,
            source_note="query_command:person",
        )

        if not profile.get("success"):
            return False, f"❌ 人物画像查询失败: {profile.get('error', 'unknown')}"

        block = PersonProfileService.format_persona_profile_block(profile)
        if not block:
            block = "暂无足够证据形成该人物画像。"
        return True, block

    def _query_stats(self) -> Tuple[bool, str]:
        """查询统计信息

        Returns:
            Tuple[bool, str]: (是否成功, 统计信息)
        """
        # 收集统计信息
        stats = {
            "vector_store": {
                "向量数量": self.vector_store.num_vectors if self.vector_store else 0,
                "维度": self.vector_store.dimension if self.vector_store else 0,
            },
            "graph_store": {
                "节点数": self.graph_store.num_nodes if self.graph_store else 0,
                "边数": self.graph_store.num_edges if self.graph_store else 0,
            },
            "metadata_store": {
                "段落数": self.metadata_store.count_paragraphs() if self.metadata_store else 0,
                "关系数": self.metadata_store.count_relations() if self.metadata_store else 0,
                "实体数": self.metadata_store.count_entities() if self.metadata_store else 0,
            },
            "sparse": self.sparse_index.stats() if self.sparse_index else None,
            "relation_vectorization": {},
        }
        plugin_instance = self.plugin_config.get("plugin_instance")
        if plugin_instance is not None and hasattr(plugin_instance, "get_relation_vector_stats"):
            try:
                stats["relation_vectorization"] = plugin_instance.get_relation_vector_stats()
            except Exception as e:
                logger.warning(f"{self.log_prefix} 读取关系向量统计失败: {e}")
        
        # 获取知识类型分布
        type_distribution = {}
        if self.metadata_store:
            cursor = self.metadata_store._conn.cursor()
            cursor.execute("""
                SELECT knowledge_type, COUNT(*) as count
                FROM paragraphs
                GROUP BY knowledge_type
            """)
            for row in cursor.fetchall():
                type_name = row[0] if row[0] else "未分类"
                count = row[1]
                type_distribution[type_name] = count

        # 构建响应
        lines = [
            "📊 知识库统计信息",
            "",
            "📦 向量存储:",
            f"  - 向量数量: {stats['vector_store']['向量数量']}",
            f"  - 维度: {stats['vector_store']['维度']}",
            "",
            "🕸️ 图存储:",
            f"  - 节点数: {stats['graph_store']['节点数']}",
            f"  - 边数: {stats['graph_store']['边数']}",
            "",
            "📝 元数据存储:",
            f"  - 段落数: {stats['metadata_store']['段落数']}",
            f"  - 关系数: {stats['metadata_store']['关系数']}",
            f"  - 实体数: {stats['metadata_store']['实体数']}",
        ]

        sparse_stats = stats.get("sparse")
        if sparse_stats:
            lines.extend([
                "",
                "🧩 稀疏检索:",
                f"  - 启用: {'是' if sparse_stats.get('enabled') else '否'}",
                f"  - 已加载: {'是' if sparse_stats.get('loaded') else '否'}",
                f"  - Tokenizer: {sparse_stats.get('tokenizer_mode', 'N/A')}",
                f"  - FTS文档数: {sparse_stats.get('doc_count', 0)}",
            ])

        rel_vec_stats = stats.get("relation_vectorization") or {}
        rel_state_stats = rel_vec_stats.get("states") if isinstance(rel_vec_stats, dict) else None
        if rel_state_stats:
            ready_cov = float(rel_vec_stats.get("relation_ready_coverage", 0.0) or 0.0) * 100
            vector_cov = float(rel_vec_stats.get("relation_vector_coverage", 0.0) or 0.0) * 100
            lines.extend([
                "",
                "🧠 关系向量化:",
                f"  - total: {rel_state_stats.get('total', 0)}",
                f"  - ready: {rel_state_stats.get('ready', 0)}",
                f"  - pending: {rel_state_stats.get('pending', 0)}",
                f"  - failed: {rel_state_stats.get('failed', 0)}",
                f"  - none: {rel_state_stats.get('none', 0)}",
                f"  - orphan_vectors: {rel_vec_stats.get('orphan_vectors', 0)}",
                f"  - ready_coverage: {ready_cov:.1f}%",
                f"  - vector_coverage: {vector_cov:.1f}%",
                f"  - ready_but_missing_vector: {rel_vec_stats.get('ready_but_missing_vector', 0)}",
            ])
        
        # 添加类型分布
        if type_distribution:
            lines.append("")
            lines.append("🏷️ 知识类型分布:")
            for type_name, count in sorted(type_distribution.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / stats['metadata_store']['段落数'] * 100) if stats['metadata_store']['段落数'] > 0 else 0
                lines.append(f"  - {type_name}: {count} ({percentage:.1f}%)")

        return True, "\n".join(lines)

    def _get_help_message(self) -> str:
        """获取帮助消息

        Returns:
            帮助消息文本
        """
        return """📖 查询命令帮助

用法:
  /query search <查询文本>      - 检索相关内容（默认模式）
  /query time <k=v参数>         - 时间检索（支持语义+时间）
  /query entity <实体名称>      - 查询实体信息
  /query relation <关系规格>    - 查询关系信息
  /query person <id|别名>      - 查询人物画像
  /query stats                  - 显示统计信息
  /query help                   - 显示此帮助

快捷模式:
  /query s <查询文本>           - 检索（search的简写）
  /query t <k=v参数>            - 时间检索（time的简写）
  /query e <实体名称>           - 实体查询（entity的简写）
  /query r <关系规格>           - 关系查询（relation的简写）
  /query p <id|别名>            - 人物画像（person的简写）

示例:
  /query search 人工智能的应用
  /query time q="项目进展" from=2025/01/01 to="2025/01/31 18:30"
  /query entity Apple
  /query relation Apple|founded|Steve Jobs
  /query person id=7fa7f...
  /query person 晨曦
  /query relation founded by
  /query stats

说明:
  - 检索模式会同时搜索段落和关系
  - time 模式参数: q/query, from/start, to/end, person, source, top_k
  - person 模式支持 id/person_id 参数或直接输入别名
  - time 格式仅支持 YYYY/MM/DD 或 YYYY/MM/DD HH:mm
  - 实体查询显示关联实体和相关段落
  - 关系格式支持 "|" 或空格分隔
  - 统计模式显示知识库概览
"""
