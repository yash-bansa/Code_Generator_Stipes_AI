import asyncio
import ast
from datetime import datetime
import hashlib
import logging
import os
import re
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, cast
from pathlib import Path
import time
import json
from enum import Enum
import asyncpg
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from config.agents_io import DocumentGeneratorInput, DocumentGeneratorOutput, SearchResult
from utils.llm_client import llm_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()
SCHEMA_NAME = os.getenv("PG_SCHEMA_NAME")
file_embeddings = os.getenv("PG_COLLECTION_FILE_NAME")
class_embeddings = os.getenv("PG_COLLECTION_CLASS_NAME")
function_embeddings = os.getenv("PG_COLLECTION_FUNCTION_NAME")

class SimpleDatabase:
    """Simplified PostgreSQL-only database class"""

    def __init__(self, database_config: Dict[str, Any]):
        self.config = database_config
        self.pool = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize PostgreSQL connection pool."""
        try:
            self.pool = await asyncpg.create_pool(**self.config)
            self.logger.info("PostgreSQL connection pool initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            raise

    async def execute_query(self, query: str, *params) -> List[Dict[str, Any]]:
        """
        Execute a parameterized PostgreSQL query and return results.

        Args:
            query: SQL query with PostgreSQL parameter placeholders ($1, $2, etc.)
            *params: Query parameters in order

        Returns:
            List of dictionaries representing query results
        """
        if not self.pool:
            await self.initialize()

        try:
            async with self.pool.acquire() as connection:
                rows = await connection.fetch(query, *params)

                # Convert asyncpg Record objects to dictionaries
                results = [dict(row) for row in rows]

                return results

        except Exception as e:
            self.logger.error(f"Query exection failed: {e}")
            self.logger.error(f"Query: {query}")
            self.logger.error(f"Params: {params}")
            return []



    async def get_file_info_for_vector(self, search_level, conn, query_embedding_str, repo_id, limit):
        if search_level == "file" or search_level == "all":
            file_results = await conn.fetch(f"""
                SELECT 'file' as type, file_path, file_path as name,
                        embedding <=> $1 as distance,
                        programming_language, enhanced_content, rich_metadata
                FROM {SCHEMA_NAME}.{file_embeddings}
                WHERE repo_id = $2
                ORDER BY embedding <=> $1
                LIMIT $3
            """, query_embedding_str, repo_id, limit)
        else:
            file_results = []
        return file_results

    async def get_class_info_for_vector(self, search_level, conn, query_embedding_str, repo_id, limit):
        if search_level == "class" or search_level == "all":
            class_results = await conn.fetch(f"""
                SELECT 'class' as type, file_path, class_name as name,
                        embedding <=> $1 as distance,
                        enhanced_content, rich_metadata
                FROM {SCHEMA_NAME}.{class_embeddings}
                WHERE repo_id = $2
                ORDER BY embedding <=> $1
                LIMIT $3
            """, query_embedding_str, repo_id, limit)
        else:
            class_results = []
        return class_results

    async def get_function_info_for_vector(self, search_level, conn, query_embedding_str, repo_id, limit):
        if search_level == "function" or search_level == "all":
            function_results = await conn.fetch(f"""
                SELECT 'function' as type, file_path, function_name as name,
                        embedding <=> $1 as distance,
                        enhanced_content, parent_class, rich_metadata
                FROM {SCHEMA_NAME}.{function_embeddings}
                WHERE repo_id = $2
                ORDER BY embedding <=> $1
                LIMIT $3
            """, query_embedding_str, repo_id, limit)
        else:
            function_results = []
        return function_results


    async def search_by_vector(self, query_embedding: List[float], repo_id: str,
                               search_level:str = "all", limit: int=20):
        print("THE SEARCH LEVEL SPECIFIED HERE IS:", search_level)
        if not self.pool:
            await self.initialize()
        if not self.pool:
            raise RuntimeError("Database connection failed")

        async with self.pool.acquire() as conn:
            # Convert query embedding to vector string format
            query_embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

            file_results = await self.get_file_info_for_vector(search_level, conn, query_embedding_str, repo_id, limit)
            class_results = await self.get_class_info_for_vector(search_level, conn, query_embedding_str, repo_id, limit)
            function_results = await self.get_function_info_for_vector(search_level, conn, query_embedding_str, repo_id, limit)

            # Combine and sort all the results
            all_results = list(file_results) + list(class_results) + list(function_results)
            all_results.sort(key=lambda x: x['distance'])

            return all_results[:limit]



    async def get_function_match_for_metadata(self, conn, repo_id, term, limit):
        function_matches = await conn.fetch(f"""
            SELECT 'function' as type, file_path, function_name as name,
                    0.95 as score, enhanced_content, rich_metadata
            FROM {SCHEMA_NAME}.{function_embeddings}
            WHERE repo_id = $1 AND function_name ILIKE $2
            LIMIT $3
        """, repo_id, f"%{term}%", limit)
        return function_matches

    async def get_class_match_for_metadata(self, conn, repo_id, term, limit):
        class_matches = await conn.fetch(f"""
            SELECT 'class' as type, file_path, class_name as name,
                    0.9 as score, enhanced_content, rich_metadata
            FROM {SCHEMA_NAME}.{class_embeddings}
            WHERE repo_id = $1 AND class_name ILIKE $2
            LIMIT $3
        """, repo_id, f"%{term}%", limit)
        return class_matches

    async def get_file_match_for_metadata(self, conn, repo_id, term, limit):
        file_matches = await conn.fetch(f"""
            SELECT 'file' as type, file_path, file_path as name,
                    0.85 as score, enhanced_content, rich_metadata
            FROM {SCHEMA_NAME}.{file_embeddings}
            WHERE repo_id = $1 AND file_path ILIKE $2
            LIMIT $3
        """, repo_id, f"%{term}%", limit)
        return file_matches

    def remove_duplicates_metadata(self, results):
        seen = set()
        unique_results = []
        for result in results:
            key = (result['type'], result['file_path'], result['name'])
            if key not in seen:
                seen.add(key)
                unique_results.append(result)

        unique_results.sort(key=lambda x: x['score'], reverse=True)
        return unique_results

    async def search_by_metadata(self, repo_id: str, query_terms: List[str], limit: int = 20):
        """Fast metadata search for exact matches"""
        if not self.pool:
            await self.initialize()
        if not self.pool:
            raise RuntimeError("Database connection failed")

        async with self.pool.acquire() as conn:
            results = []

            for term in query_terms:
                # Search function names
                function_matches = await self.get_function_match_for_metadata(conn, repo_id, term, limit)

                # Search class names
                class_matches = await self.get_class_match_for_metadata(conn, repo_id, term, limit)

                # Search file paths
                file_matches = await self.get_file_match_for_metadata(conn, repo_id, term, limit)

                results.extend(list(function_matches) + list(class_matches) + list(file_matches))

            # Remove duplicates and sort by score
            unique_results = self.remove_duplicates_metadata(results)
            return unique_results[:limit]

    async def close(self):
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            self.logger.info("PostgreSQL connection pool closed")

class SimpleDocumentGenerator:
    """
    Simple Document Generator with Integrated Search Tools
    """

    def __init__(self):
        self.database_config = self.load_database_config()
        self.simple_database_obj = SimpleDatabase(self.database_config)
        self.embedding_client, self.llm_client = self.initialize_azure_clients()
        self.logger = logging.getLogger(__name__)

    async def get_simpledatabase_obj(self):
        database_config = self.load_database_config()
        simple_database_obj = SimpleDatabase(database_config)
        await simple_database_obj.initialize()
        return simple_database_obj

    def initialize_azure_clients(self):
        """Initialize Azure OpenAI clients with proper error handling."""
        try:
            embedding_client = AzureOpenAIEmbeddings(
                model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
                azure_endpoint = os.getenv("AZURE_OPENAI_BASE_URL"),
                api_key = os.getenv("AZURE_OPENAI_API_KEY"),
                api_version = os.getenv("AZURE_API_VERSION")
            )


            print("Azure OpenAI clients initialized successfully")
            return embedding_client, llm_client

        except Exception as e:
            print(f"Failed to initialize Azure clients: {e}")
            return None, None


    def load_database_config(self):
        """Load database configuration from environment."""

        database_config = {
            'host' : os.getenv('PG_DB_HOST'),
            'port' : os.getenv('PG_DB_PORT'),
            'user' : os.getenv('PG_DB_USER'),
            'password' : os.getenv('PG_DB_PASSWORD'),
            'database' : os.getenv('PG_DB_NAME')
        }

        return database_config

    def get_enhanced_candidates(self, candidates):
        enhanced_candidates = []
        for i, result in enumerate(candidates):
            candidate_data = {
                "index": i,
                "file": result.file_path,
                "name": result.name,
                "type": result.type,
                "base_score": round(result.relevance_score, 3),
                "parent_class": result.parent_class if hasattr(result, 'parent_class') else None,
                "description": result.enhanced_content[:150] + "..." if len(result.enhanced_content) > 150 else result.enhanced_content
            }
            enhanced_candidates.append(candidate_data)
        return enhanced_candidates

    def get_enhanced_prompt(self, query, enhanced_candidates):
        prompt = f"""You are an expert code search assistant. Analyze these search results for: "{query}"

        Candidates:
        {json.dumps(enhanced_candidates, indent=2)}

        Consider:
        1. Functional relevance - does this code do what the user is looking for?
        2. File/function naming - does the name suggest it's relevant?
        3. Code context - based on the description, is this useful?
        4. Architectural fit - is this the right type of component?

        Return JSON with ranked results (most relevant first):
        {{"data":
        [
            {{"index" : 0, "relevance": 9.2, "reason":"Perfect match - main function that directly addressess the query"}},
            {{"index" : 3, "relevance" : 8.1, "reason":"Supporting component that would be useful for the task" }},
            ...
        ]
        }}
        Rules: Only include relevance >=7.0, order by relevance.
        return only the results which are absolutely necessary to solve the problem.
        U can return one or multiple results.
        Important Instruction: Make sure that every relevance score should be between the range of 1 - 10.
        """

        return prompt

    async def _enhanced_llm_rerank(self, query: str, candidates:List[SearchResult]) -> List[SearchResult]:
        """Enhanced LLM reranking using available metadata."""
        if not self.llm_client or not candidates:
            return candidates

        try:
            # Prepare candidate data with available information
            enhanced_candidates = self.get_enhanced_candidates(candidates)
            prompt = self.get_enhanced_prompt(query, enhanced_candidates)

            response = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )
            rankings = self._extract_json(response)
            rankings = json.loads(rankings)['data']
            print("printing rankings:",rankings)

            # Apply rankings
            reranked_results = []
            for item in rankings:
                idx = item["index"]
                if 0 <= idx < len(candidates):
                    result = candidates[idx]
                    result.relevance_score = item["relevance"] / 10.0
                    result.match_reason = item["reason"]
                    reranked_results.append(result)
            return reranked_results[:8]

        except Exception as e:
            self.logger.warning(f"Enhanced LLM reranking failed enhanced_llm: {e}")
            return candidates[:8]


    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract meaningful terms from query for metadata search."""
        # Simple term extraction - split on whitespace and remove common words
        terms = query.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

        meaningful_terms = []
        for term in terms:
            # Remove punctuation
            clean_term = re.sub(r'[^\w]', '', term)
            if clean_term and len(clean_term) > 2 and clean_term not in stop_words:
                meaningful_terms.append(clean_term)

        print("COMPLETED EXTRACTING QUERY TERMS")
        return meaningful_terms


    async def close(self):
        """Close database connections."""
        await self.simple_database_obj.close()


    # ========================================================================
    # TOOL SELECTION SYSTEM
    # ========================================================================

    def add_metadata_results_for_combining_hybrid(self, metadata_results):
        combined = {}
        # Add metadata results (higher priority)
        for row in metadata_results:
            key = (row['file_path'], row['name'], row['type'])
            # Convert score to float to handle Decimal types from PostgreSQL
            base_score = float(row.get('score', 0.5))
            # Create SearchResult with rich_metadata if available
            search_result = SearchResult(
                file_path = row['file_path'],
                name = row['name'],
                type = row['type'],
                relevance_score = base_score + 0.1, # Boost metadata matches
                enhanced_content = row.get('enhanced_content', ''),
                parent_class = "",
                match_reason = "",
                rich_metadata = ""
            )
            # Add rich_metadata if available from database
            if 'rich_metadata' in row and row['rich_metadata']:
                search_result.rich_metadata = row['rich_metadata']
            combined[key] = search_result
        return combined


    def _combine_results(self, metadata_results: List, vector_results: List[SearchResult]) -> List[SearchResult]:
        """Combine metadata and vector search results."""
        combined = self.add_metadata_results_for_combining_hybrid(metadata_results)

        # Add vector results
        for result in vector_results:
            key = (result.file_path, result.name, result.type)
            if key not in combined:
                combined[key] = result
            else:
                # Take higher score but preserve rich_metadata
                if result.relevance_score > combined[key].relevance_score:
                    # Preserve rich_metadata if the existing result has it
                    existing_rich_metadata = getattr(combined[key], 'rich_metadata', None)
                    combined[key] = result
                    if existing_rich_metadata:
                        combined[key].rich_metadata = existing_rich_metadata

        # Sort by relevance score
        results = list(combined.values())
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results

    def _extract_json(self, response: str) -> str:
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            return response[start:end].strip()
        return response.strip()


    def get_query_for_tool_selection(self, query):
        prompt = f"""You are an intelligent tool selector. Analyze this developer user query and select the most appropriate search tool:

            Query: "{query}"

            Available Tools:
            1. **identify_file** - Use when user specifies exact file/class/function names
            - Parameters: type ("file"|"class"|"function"), name (string)
            - Examples: "find UserService class", "show login.py file", "get authenticate function"

            2. **hybrid_search** - Use for complex queries requiring semantic understanding.
            - Parameters: query(string), focus_area ("files"|"classes"|"functions"|"all")
            - Examples: "authentication logic", "payment processing code", "error handling patterns"

            3. **keyword_match** - Use for finding specific patterns or code snippets.
            - Parameters: keywords (list of strings), search_scope ("all"|"files"|"classes"|"functions")
            - Examples: "find all TODO comments", "search for database connections", "find error handling"

            Decision Rules:
            - Analyze, think before you choose the right tool for the job. I need higher accuracy.
            - Do not use Hybrid search for all the user queries.
            - If query contains specific names such as file names, classes , function names -> use identify_file
            - If query is conceptual/semantic (logic, patterns, behavior) -> use hybrid_search
            - If query asks for specific code patterns/keywords -> use keyword_match
            -

            Return JSON with your decision:
            {{
                "tool_name" : "identify_file|hybrid_search|keyword_match",
                "parameters" : {{
                    // tool-specific parameters based on the tool selected
                }},
                "reasoning" : "Explain why this tool was chosen over the other tools for this query."
            }}

            Analysis:"""
        return prompt

    async def _analyze_query_for_tool_selection(self, query: str) -> Dict[str, Any]:
        """LLM analyzes query and selects the best tool."""

        prompt = self.get_query_for_tool_selection(query)

        try:
            response = await self.llm_client.chat_completion(
                messages=[{"role":"user", "content": prompt}]
            )
            clean_json_string = self._extract_json(response)
            decision = json.loads(clean_json_string)

            self.logger.info(f"LLM selected tool: {decision['tool_name']}")
            self.logger.info("*******************************************************************************")
            self.logger.info(f"Reasoning: {decision.get('reasoning','No reason provided')}")

            return decision

        except Exception as e:
            self.logger.warning(f"LLM tool selection failed: {e}")
            # Fallback to hybrid search
            return {
                "tool_name" : "hybrid_search",
                "parameters": {"query": query, "focus_area": "all"},
                "reasoning": "LLM tool selection failed using hybrid search feedback"
            }

    async def _execute_selected_tool(self, tool_call: Dict[str, Any], repo_id:str, max_results:int) -> List[SearchResult]:
        """Execute the selected tool with parameters"""

        tool_name = tool_call["tool_name"]
        parameters = tool_call["parameters"]

        if tool_name == "identify_file":
            print("************************ Inside Identify Files ********************")
            return await self._tool_identify_file(
                type=parameters["type"],
                name=parameters["name"],
                repo_id=repo_id
            )

        elif tool_name == "hybrid_search":
            print("******************* Inside Hybrid Search tool *********************")
            return await self._tool_hybrid_search(
                query = parameters["query"],
                repo_id=repo_id,
                focus_area=parameters.get("focus_area","all"),
                max_results=max_results
            )

        elif tool_name == "keyword_match":
            print("********************* Inside tool keyword match *********************")
            print("KEYWORDS:", parameters["keywords"])
            return await self._tool_keyword_match(
                keywords=parameters["keywords"],
                repo_id=repo_id,
                search_scope=parameters.get("search_scope", "all"),
                max_results=max_results
            )

        else:
            self.logger.warning(f"Unknown tool: {tool_name}, falling back to hybrid search")
            return await self._tool_hybrid_search(parameters["query"], repo_id, "all", max_results)

    # =====================================================================================
    # TOOL 1: IDENTIFY FILE (Exact Name Search)
    #  =====================================================================================

    def convert_to_search_result_object_for_file_identification_tool(self, results, type, name):
        search_results = []
        for row in results:
            print("------------------ type of ", row['type'])
            if type == "file":
                variable = row["file_path"]
            elif type == "class":
                variable = row["class_name"]
            elif type == "function":
                variable = row["function_name"]

            result = SearchResult(
                file_path=row["file_path"],
                type=row["type"],
                name=variable,
                relevance_score=1.0,
                enhanced_content=row.get('enhanced_content', ''),
                parent_class=row.get('parent_class',''),
                rich_metadata=row.get('rich_metadata'),
                match_reason=f"Exact {type} match: {name}"
            )
            search_results.append(result)
        return search_results

    async def _tool_identify_file(self, type: str, name: str, repo_id: str) -> List[SearchResult]:
        """
        Tool 1: Exact metadata search by type and name.
        Use when user specifies exact file/class/function names.
        """
        try:
            self.logger.info(f"Tool 1: Searching for {type} named '{name}'")

            if type == "file":
                results = await self.search_file_by_name(repo_id, name)
            elif type == "class":
                results = await self.search_class_by_name(repo_id, name)
            elif type == "function":
                results = await self.search_function_by_name(repo_id, name)
            else:
                raise ValueError(f"Invalid type: {type}")

            # Convert to SearchResult objects
            search_results = self.convert_to_search_result_object_for_file_identification_tool(results, type, name)

            self.logger.info(f"Tool 1 found  {len(search_results)} exact matches.")
            return search_results

        except Exception as e:
            self.logger.error(f"Tool 1 failed: {e}")
            return []

    # ========================================================================================================
    # Tool 2 : Hybrid Search (Vector + Metadata)
    # ========================================================================================================

    async def get_vector_response_for_hybrid_search(self, query, focus_area, max_results, repo_id):
        if self.embedding_client:
            print("************************ Inside vector search for semantic similarity **************************")
            query_embedding = self.embedding_client.embed_query(query)
            search_level = self._map_focus_to_search_level(focus_area)
            print("** Search LEVEL **", search_level)

            vector_results = await self.simple_database_obj.search_by_vector(
                query_embedding, repo_id, search_level=search_level, limit=max_results
            )

            # Convert vector results to SearchResult objects
            vector_search_results = []
            for row in vector_results:
                distance = float(row['distance'])
                score = 1.0 - distance

                result = SearchResult(
                    file_path=row['file_path'],
                    name=row['name'],
                    type=row['type'],
                    relevance_score=score,
                    enhanced_content=row.get('enhanced_content', ''),
                    parent_class=row.get('parent_class', ''),
                    rich_metadata=row.get('rich_metadata', ''),
                    match_reason = ""
                )
                vector_search_results.append(result)
        else:
            vector_search_results = []
        return vector_search_results

    def combine_results_and_remove_duplicates_for_hybrid_search(self, metadata_results, enhanced_results):

        combined_results = self._combine_results(metadata_results, enhanced_results)

        for result in combined_results:
            result.match_reason = "Hybrid match: semantic similarity + metadata keywords"

        dict_remove_duplicate = {}
        for result in combined_results:
            if result.file_path not in dict_remove_duplicate:
                dict_remove_duplicate[result.file_path] = result

        for file_path in dict_remove_duplicate:
            print("combined results", file_path)

        combined_results = [dict_remove_duplicate[key] for key in dict_remove_duplicate]
        return combined_results

    def print_enhanced_and_metadata_results(self, enhanced_results, metadata_results):
        for vector_data in enhanced_results:
            print(vector_data.file_path, vector_data.relevance_score)

        for row in metadata_results:
            dict_meta = dict(row)
            print("meta_datasearch:", dict_meta['file_path'])


    async def _tool_hybrid_search(self, query: str, repo_id: str, focus_area: str = "all", max_results: int = 20) -> List[SearchResult]:
        """
        Tool 2: Hybrid search combining vector similarity + metadata filtering.
        Use for complex queries requiring semantic understanding + context.
        """
        try:
            self.logger.info(f"Tool 2 : Hybrid search for '{query}' (focus: {focus_area})")

            # Step 1: Vector search for semantic similarity
            vector_search_results = await self.get_vector_response_for_hybrid_search(query, focus_area, max_results, repo_id)

            # Call the LLM reranker here. Only for vector data.
            print("before going to enhanced results", len(vector_search_results))
            enhanced_results = await self._enhanced_llm_rerank(query, vector_search_results)
            print("after enhancing", len(enhanced_results))

            # Step 2: Metadata keyword search
            search_terms = self._extract_query_terms(query)

            metadata_results = await self.simple_database_obj.search_by_metadata(
                repo_id, search_terms, limit=max_results//2
            )

            self.print_enhanced_and_metadata_results(enhanced_results, metadata_results)
            combined_results = self.combine_results_and_remove_duplicates_for_hybrid_search(metadata_results, enhanced_results)

            self.logger.info(f"Tool 2 found {len(combined_results)} hybrid matches")
            return combined_results[:max_results]

        except Exception as e:
            self.logger.error(f"Tool 2 failed {e}")
            return []

    # ===========================================================================================================================
    # Tool 3: Keyword match (Full-text Search)
    # ===========================================================================================================================


    def get_search_result_for_keyword_match(self, results, keywords):
        search_results = []
        for row in results:
            # Calculate relevance based on keyword matches
            relevance = self._calculate_keyword_relevance(row.get('enhanced_content', ''), keywords)

            result = SearchResult(
                file_path=row['file_path'],
                name=row['name'],
                type=row['type'],
                relevance_score=relevance,
                enhanced_content=row.get('enhanced_content', ''),
                parent_class = row.get('parent_class', ''),
                rich_metadata=row.get('rich_metadata', ''),
                match_reason=f"Keyword matches: {', '.join(keywords)}"
            )
            search_results.append(result)

        # Sort by relevance
        search_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return search_results


    async def _tool_keyword_match(self, keywords: List[str], repo_id: str, search_scope: str = "all", max_results: int = 20) -> List[SearchResult]:
        """
        Tool 3: Keyword-based full-text search across code content.
        Use for finding specific patterns, function names or code snippets.
        """
        try:
            self.logger.info(f"Tool 3: Keyword search for {keywords} (scope: {search_scope})")

            # Search using database keyword search
            results = await self.search_by_keywords(
                repo_id, keywords, search_scope=search_scope
            )
            self.logger.info("Length of RESULTS KEYWORD: "+str(len(results)))

            # Convert to SearchResult objects
            search_results = self.get_search_result_for_keyword_match(results, keywords)

            self.logger.info(f"Tool 3 found {len(search_results)} keyword matches.")
            return search_results[:max_results]

        except Exception as e:
            self.logger.error(f"Tool 3 failed: {e}")
            return []

    # =======================================================================================================
    # HELPER METHODS FOR TOOLS
    # =======================================================================================================

    def _map_focus_to_search_level(self, focus_area: str) -> str:
        """Map focus area to database search level."""
        print("INSIDE MAP FOCUS TO SEARCH LEVEL")
        mapping = {
            "files": "file",
            "classes": "class",
            "functions": "function",
            "all": "all"
        }
        return mapping.get(focus_area, "all")

    def _calculate_keyword_relevance(self, content: str, keywords: List[str]) -> float:
        """Calculate relevance score based on keyword frequency"""
        if not content or not keywords:
            return 0.0

        content_lower = content.lower()
        total_matches = 0

        for keyword in keywords:
            matches = content_lower.count(keyword.lower())
            total_matches += matches

        # Normalize by content length and number of keywords
        content_length = len(content.split())
        if content_length == 0:
            return 0.0

        relevance = min(total_matches / (content_length * len(keywords) * 0.01), 1.0)
        return relevance


    async def search_file_by_name(self, repo_id: str, name: str):
        files = [file.strip() for file in name.split(',')]
        print("FILESSS:", files)
        conditions = []
        params = [repo_id] # Start with repo_id as first parameter
        param_index = 2 # Start from $2 since $1 is repo_id

        for name in files:
            # Add two conditions for each name (with % on both sides and exact match)
            conditions.append(f"file_path ILIKE ${param_index} OR file_path ILIKE ${param_index + 1}")
            params.extend([f"%{name}%", f"%{name}"])
            param_index += 2

        # Join all conditions with OR
        where_clause = " OR ".join([f"({condition})" for condition in conditions])

        query = f"""
            SELECT file_path, 'file' as type, enhanced_content, rich_metadata
            FROM {SCHEMA_NAME}.{file_embeddings}
            WHERE repo_id = $1 AND ({where_clause})
            ORDER BY file_path
            """

        results = await self.simple_database_obj.execute_query(query, *params)
        self.logger.info("LENGTH OF RESULTS file:")
        self.logger.info(len(results))
        return results

    async def search_class_by_name(self, repo_id: str, name: str):

        """Search for classes by exact name."""
        classes = [file.strip() for file in name.split(',')]
        print("Classes:", classes)
        conditions = []
        params = [repo_id] # Start with repo_id as first parameter
        param_index = 2 # Start from $2 since $1 is repo_id

        for name in classes:
            # Add two conditions for each name (with % on both sides and exact match)
            conditions.append(f"class_name ILIKE ${param_index} OR class_name ILIKE ${param_index + 1}")
            params.extend([f"%{name}%", f"%{name}"])
            param_index += 2

        # Join all conditions with OR
        where_clause = " OR ".join([f"({condition})" for condition in conditions])
        query = f"""
            SELECT file_path, 'class_name' as type, enhanced_content, rich_metadata, class_name
            FROM {SCHEMA_NAME}.{class_embeddings}
            WHERE repo_id = $1 AND ({where_clause})
            ORDER BY file_path
        """
        results = await self.simple_database_obj.execute_query(query, *params)
        self.logger.info("LENGTH OF RESULTS CLSS:")
        self.logger.info(len(results))
        return results

    async def search_function_by_name(self, repo_id: str, name: str):
        """Search for functions by exact name."""

        functions = [file.strip() for file in name.split(',')]
        print("Functions:", functions)
        conditions = []
        params = [repo_id] # Start with repo_id as first parameter
        param_index = 2 # Start from $2 since $1 is repo_id

        for name in functions:
            # Add two conditions for each name (with % on both sides and exact match)
            conditions.append(f"function_name ILIKE ${param_index} OR function_name ILIKE ${param_index + 1}")
            params.extend([f"%{name}%", f"%{name}"])
            param_index += 2

        # Join all conditions with OR
        where_clause = " OR ".join([f"({condition})" for condition in conditions])
        query = f"""
            SELECT file_path, 'function_name' as type, enhanced_content, rich_metadata, parent_class, function_name
            FROM {SCHEMA_NAME}.{function_embeddings}
            WHERE repo_id = $1 AND ({where_clause})
            ORDER BY file_path
        """
        results = await self.simple_database_obj.execute_query(query, *params)
        self.logger.info("LENGTH OF RESULTS function:")
        self.logger.info(len(results))
        return results


    def get_query_for_keyword_search(self, table, keyword_clause, search_scope):
        if table == "file_embeddings":
            query = f"""
            SELECT file_path, '{search_scope[:-1]}' as type, enhanced_content, rich_metadata, file_path as name
            FROM {SCHEMA_NAME}.{table}
            WHERE repo_id = $1 AND {keyword_clause}
            ORDER BY file_path
            """
        elif table == "class_embeddings":
            query = f"""
            SELECT file_path, '{search_scope[:-1]}' as type, enhanced_content, rich_metadata, class_name as name
            FROM {SCHEMA_NAME}.{table}
            WHERE repo_id = $1 AND {keyword_clause}
            ORDER BY file_path
            """
        elif table == "function_embeddings":
            query = f"""
            SELECT file_path, '{search_scope[:-1]}' as type, enhanced_content, rich_metadata, function_name as name
            FROM {SCHEMA_NAME}.{table}
            WHERE repo_id = $1 AND {keyword_clause}
            ORDER BY file_path
            """
        return query


    def get_query_for_search_by_keywords(self, keyword_clause):
        query = f"""
            (SELECT file_path, 'file' as type, file_path as name, enhanced_content, rich_metadata FROM {SCHEMA_NAME}.{file_embeddings}
            WHERE repo_id = $1 AND {keyword_clause})
            UNION ALL
            (SELECT file_path, 'class_name' as type, class_name as name, enhanced_content, rich_metadata FROM {SCHEMA_NAME}.{class_embeddings}
            WHERE repo_id = $1 AND {keyword_clause})
            UNION ALL
            (SELECT file_path, 'function_name' as type, function_name as name, enhanced_content, rich_metadata FROM {SCHEMA_NAME}.{function_embeddings}
            WHERE repo_id = $1 AND {keyword_clause})
            ORDER BY file_path
            """
        return query


    async def search_by_keywords(self, repo_id: str, keywords: List[str], search_scope: str = "all"):
        """Search by keywords across content."""
        keyword_conditions = []
        params = [repo_id]

        for i, keyword in enumerate(keywords):
            keyword_conditions.append(f"(enhanced_content ILIKE ${i+2} OR original_content ILIKE ${i+2})")
            params.append(f"%{keyword}%")

        keyword_clause = " OR ".join(keyword_conditions)

        if search_scope == "all":
            query = self.get_query_for_search_by_keywords(keyword_clause)
        else:
            table_map = {"files": "file_embeddings", "classes": "class_embeddings", "functions":"function_embeddings"}
            table = table_map.get(search_scope, "file_embeddings")
            query = self.get_query_for_keyword_search(table, keyword_clause, search_scope)
        results = await self.simple_database_obj.execute_query(query, *params)
        return results


class DocumentGeneratorAgent:
    async def generate_document(self, input_data: DocumentGeneratorInput) -> DocumentGeneratorOutput:
        try:
            query = input_data.developer_task_query
            print("The query is ", query)
            self.generator = SimpleDocumentGenerator()
            self.repo_id = "simple_repo_sample_repo"
            tool_decision = await self.generator._analyze_query_for_tool_selection(query)

            selected_tool = tool_decision.get("tool_name", "unknown")
            parameters = tool_decision.get("parameters", {})
            reasoning = tool_decision.get("reasoning", "No reasoning provided")

            print(f"Selected Tool : {selected_tool}")
            print(f"Parameters: {json.dumps(parameters, indent=2)}")
            print(f"LLM Reasoning: {reasoning}")

            # Step 2: Tool Execution
            print(f"\n Step 2: Executing {selected_tool} tool...")
            search_results = await self.generator._execute_selected_tool(
                tool_decision, self.repo_id, max_results=10
            )

            output_list = []
            for search_result in search_results:
                output_list.append(search_result.dict())


            output = str(output_list)
            return DocumentGeneratorOutput(
                generated_doc = output,
                success = True,
                message = "Successfully generated doc"
            )

        except Exception as e:
            logging.error(f"[DocumentGeneratorAgent] Error: {e}", exc_info=True)
            return DocumentGeneratorOutput(
                generated_doc = "",
                success = False,
                message = f"Failed to generate: {str(e)}"
            )