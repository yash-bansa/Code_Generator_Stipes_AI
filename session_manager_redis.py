from config.settings import settings
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json,sys
from config.agents_io import BotStateSchema
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("langgraph_query_flow.log")
    ]
)
logger = logging.getLogger(__name__)

# --------- Redis Connection ------------
redis_client = settings.get_redis_connection()


class AuxiliaryFunctions:
    def aux_redis_to_py(self, value:Any) -> Any:
        if isinstance(value,str):
            try :
                parsed_value = json.loads(value)
                return parsed_value
            except (json.JSONDecodeError, TypeError):
                pass

            if value.isdigit():
                return int(value)

            if value.lower() == 'true':
                return True
            if value.lower() == 'false':
                return False

            return value
        
        return value

    def convert_redis_output_to_python(self, data : Dict[str, Any]) -> Dict[str, Any]:
        return {key : self.aux_redis_to_py(value) for key, value in data.items()}
    
    def convert_python_to_redis_format(self, data : Dict[str, Any]) -> Dict[str, str]:
        return {key : self.aux_py_to_redis(value) for key, value in data.items()}
    
    def aux_py_to_redis(self, value: Any) -> str:
        if isinstance(value, (list,dict)):
            return json.dumps(value)
        elif isinstance(value,bool):
            return 'True' if value else "False"
        else:
            return str(value)            

class RedisSessionManager:
    def __init__(self):
        self.aux_functions = AuxiliaryFunctions()
    
    def get_session_dictionary_for_create_session(self, session_id, initial_state):
        now = datetime.now()
        current_time = now.isoformat()

        session_dict = {
            "session_id": session_id,
            "created_at": current_time,
            "last_activity": current_time,
            "status": "active",
            "current_stage": "initial",
            "state": initial_state,
            "waiting_for_input": False,
            "validation_attempts": 0,
            "retry_count": 0,
            "workflow_history": [],
        }
        session_dict = self.aux_functions.convert_python_to_redis_format(session_dict)
        return session_dict

    def create_session(self, session_id):
        session_history = self.get_session_history(session_id)
        initial_state = BotStateSchema(
            latest_query="",
            user_history=session_history,
            current_user=session_id
        ).dict()
        session_dict = self.get_session_dictionary_for_create_session(session_id, initial_state)
        state_key = f"session: {session_id} : state"
        redis_client.hset(state_key, mapping=session_dict)
        redis_client.expire(state_key, 604800)
        print("Created a new session", session_id)
        session = self.get_session(session_id)
        return session
    
    def get_session(self, session_id):
        state_key = f"session:{session_id} : state"
        session = redis_client.hgetall(state_key)
        if session:
            session = self.aux_functions.convert_redis_output_to_python(session)
            return session
        return None
    
    def update_session(self, session_id, updates):
        key = f"session:{session_id} : state"
        updates = self.aux_functions.convert_python_to_redis_format(updates)
        redis_client.hset(key, mapping=updates)
        redis_client.expire(key, 604800)
    
    def update_session_state(self, session_id: str, state: Dict[str, Any]):
        key = f"session:{session_id} : state"

        now = datetime.now()
        current_time = now.isoformat()
        updates = {
            "state" : state,
            "last_activity" : current_time
        }
        updates = self.aux_functions.convert_python_to_redis_format(updates)
        redis_client.hset(key, mapping=updates)
        redis_client.expire(key,604800)

        user_history = state.get("user_history", [])
        self.update_history(session_id, user_history)
    
    def get_session_history(self, session_id) -> List[str]:

        history_key = f"session:{session_id}:history"
        try:
            if redis_client:
                history = redis_client.lrange(history_key, 0, -1)
                return history if history else []
            else:
                logger.warning("Redis not avalable, using empty history")
                return []
        except Exception as e:
            logger.error(f"Error getting history from redis : {e}")
            return []

    def add_to_session_history(self, session_id: str, query: str):
        history_key = f"session:{session_id}:history"
        try:
            if redis_client:
                redis_client.rpush(history_key, query)
                redis_client.ltrim(history_key, -50, -1)
                redis_client.expire(history_key, 604800)
                logger.debug(f"Saved query to Redis for session  {session_id}")
            else:
                logger.warning("Redis not available")
        except Exception as e :
            logger.error("Error saving to Redis: {e}") 

    def update_history(self, session_id, messages):
        history_key = f"session:{session_id}:history"
        if redis_client.exists(history_key):
            redis_client.delete(history_key)
        if messages != []:
            redis_client.rpush(history_key, *[m for m in messages])
            redis_client.expire(history_key, 604800)    

    def get_all_sessions(self):
        pattern = "session:*:state"
        cursor = 0
        session_keys = []

        while True:
            cursor, keys = redis_client.scan(cursor=cursor, match=pattern, count=100 )
            session_keys.extend([Key for Key in keys])
            if cursor == 0:
                break
        
        return {"sessions" : session_keys}
    
    def delete_session(self, session_id):
        history_key = f"session:{session_id}:history"
        key = f"session:{session_id}:state"
        if redis_client.exists(key):
            redis_client.delete(key)
            redis_client.delete(history_key)
            return {"status" : f"session {session_id} deleted"}
        else:
            return {"status" : f"session {session_id} doesni't exist" }

session_manager_redis = RedisSessionManager()