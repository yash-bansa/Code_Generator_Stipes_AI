import json
import logging
from typing import Dict, Any, List
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
from pathlib import Path
from config.agents_io import DependencyAnalyzerInput, DependencyAnalyzerOutput 
from utils.llm_client import llm_client

load_dotenv()

logger = logging.getLogger(__name__)

class DependencyExtractorAgent:

    async def read_config_file(self,config_path: str) -> Dict[str, Any]:

        config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"config file not found {config_path}")
            config_path.parent.mkdir(parents= True, exist_ok = True)
            default_config = {
                "project_type" : "python",
                "framework" : "general",
                "main_files" : ["main.py"],
                "config_files" : ["config.py"] 
            }
            with open(config_path, "w") as f:
                json.dump(default_config,f,indent=2)
            logger.info(f"default config is created at {config_path}")

        with open(config_path, "r") as f:
            parsed_config = json.load(f)
        return parsed_config
    
    async def analyze_dependencies(self, input_data: DependencyAnalyzerInput) -> DependencyAnalyzerOutput:

        logger.info("Strating the config generator agent")

        user_query = input_data.user_query
        config = input_data.config
        config = config["sequence"]
        chunk_size = 10
        config_chunks = [config[i: i + chunk_size] for i in range(0, len(config), chunk_size)]

        final_updated_config = []
        feedback_config = {}

        for chunk_idx, config_chunk in enumerate(config_chunks):
            logger.info(f"Processing chunk {chunk_idx +1}/ {len(config_chunks)} ...")

            print("here lie the feedback -----", feedback_config)

            prompt = f"""
You are a config generator agent. Analyze the provided configuration chunk based on the user query.
You should also take into account the previously updated config from earlier analysis as feedback if available.
Extract **only** the relevant information necessary to fulfill the user's query and adhere to the following schema:
{{{{ "<key>" : "<value>", ....}}}}
## Inputs:
### User Query:
{user_query}
### Previously Updated Config (Feedback):
{json.dumps(feedback_config, indent=2)}
### Configuration Chunk:
{json.dumps(config_chunk, indent=2)}
## Task:
 - Use the inputs, including feedback and summary if available from previous updated config, to analyze necessary dependencies and configurations.
 - Generated an output JSON with the schema:
   {{{{ "<key>" : "<value>", ....}}}}
 - Only include the relavant configuration required to fulfil the query.
 - The generated new config must have the same schema and casing which is followed in Configuration Chunk version so it can be easily integrated.
 - Do not new keys strictly in the updated config, so if you are creating a operation use same keys as Configuration Chunk.
 - please provide a summary of each object change ypu have done for a chunk so it will be used as feedback for next loop.


 output will be have two components :
 1. Generated and output JSON with the schema 
 2. The summary of the config change 

 output :- {{{{ "<key>" : "<value>", ....}}}} | summary
 
 - the output must be simple json based on Configuration Chunk only take information the feedback.
 - dont add keys in output as edits1 or edit2 just simple json.

 --- Generated only the JSON output below ----
     
"""
            logger.debug(f"Generated prompt for LLM (chunk {chunk_idx +1}): {prompt}")

            try:
                raw_output = await llm_client.chat_completion(
                    messages = [{"role" : "user" , "content" : prompt}],
                )

                logger.info(f"Received LLM response successfully for chunk {chunk_idx+1}.")
                val = raw_output.split("|")

                result = self._extract_json(val[0])
                final_updated_config.append(result)

                count_val = len(feedback_config)

                try:
                    jason_req = {}
                    jason_req[f"edit_{count_val}"] ={
                        "edit_summary" : val[1],
                        "actual_edits" : val[0]
                    }
                except Exception as e :
                    print(e)

                feedback_config.update(jason_req)

                logger.info(f"Chunk {chunk_idx + 1} processed, feedback loop updated.")
            except Exception as e:
                logger.error(f"Failed to interact with the LLM for chunk processing : {str(e)}")
                raise ValueError(f"Error communication with the LLM : {str(e)}")

        return final_updated_config

    async def extract_table_names_from_query(self, user_query: str) -> List[str]:
        """
        Extract table names from a user query using an LLM call.
        
        Args:
            user_query (str): The user's query describing the task or SQL-like operation.
            
        Returns:
            List[str]: A list of table names extracted from the query.
        """
        try:
            # Define the prompt to instruct the LLM effectively
            prompt = f"""
            You are a database expert. 
            - Please analyze the following user query and extract all table names mentioned in the query.
            - Make sure that you return all the table names in the query separated by ",".
            - Only use the table names which are used in the query, do not make something up.
            - Always choose the table which has to be updated.
            - if there are multiple tables return only the tables where modification has to be made.
            - Do not include the source/reference tables.
            - Do not include any extra information in the output.
            
            Query: "{user_query}"
            """
            
            # Make the API call to the LLM
            response = await llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
            )
                        
            # Post-process the output to ensure it contains a list of table names
            table_names = [table.strip() for table in response.split(",") if table.strip()]
            
            return table_names

        except Exception as e:
            # Handle errors (API failure, LLM issues, etc.)
            print(f"Error extracting table names: {e}")
            return []
        
    async def filter_and_format_table_paths(self, input_tables: List[str]) -> List[str]:

        predefined_tables = ["example"]

        filtered_tables = [table for table in input_tables if table in predefined_tables]

        formatted_paths = [f"./examples/{table}.json" for table in filtered_tables]

        return formatted_paths
    
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

                


               
