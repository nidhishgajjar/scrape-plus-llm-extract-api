import os
from typing import Any, Dict, Literal, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json
from urllib.parse import urlparse
from datetime import datetime
import httpx
import asyncio
from app.config import get_settings

ModelType = Literal["gpt-4o", "gpt-4o-mini"]

class LLMProcessor:
    def __init__(self, model: ModelType = "gpt-4o-mini"):
        # Create custom httpx client with timeout
        timeout = httpx.Timeout(60.0, connect=10.0)
        http_client = httpx.AsyncClient(timeout=timeout)
        
        self.llm = ChatOpenAI(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            model=model,
            temperature=0,
            request_timeout=60,  # 60 second timeout
            http_client=http_client,
        )
        # Create JSON-mode LLM
        self.json_llm = self.llm.bind(
            response_format={"type": "json_object"}
        )
        self.settings = get_settings()
    
    async def extract_information(
        self, 
        content: str, 
        extraction_prompt: str, 
        output_format: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], str]:
        system_prompt = f"""
        You are a precise data extraction assistant. Your task is to:
        1. Analyze the provided content
        2. Extract specific information based on the given prompt
        3. Return the information in the exact JSON format specified
        
        Output Format Example:
        {json.dumps(output_format, indent=2)}
        
        Extraction Requirements:
        {extraction_prompt}
        
        Rules:
        - Strictly follow the output format
        - If a field cannot be found, use null
        - Do not include additional fields
        - Ensure valid JSON output
        - For URLs: 
          * Include complete URLs only
          * Ensure URLs are properly formatted
          * Remove any duplicate URLs
          * Validate URLs match the specified criteria
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content)
        ]
        
        try:
            # Use timeout to prevent worker hanging
            response = await asyncio.wait_for(
                self.json_llm.ainvoke(messages),
                timeout=120  # 2 minute timeout
            )
            
            print(response)
            
            extracted_data = json.loads(response.content)
            file_path = "File saving disabled in production mode"
            
            if self.settings.DEBUG_MODE:
                file_path = self._save_extracted_data(extracted_data)
                
            return extracted_data, file_path
            
        except asyncio.TimeoutError:
            # Handle timeout error
            error_data = {"error": "LLM processing timed out after 120 seconds"}
            return error_data, "timeout_error"
            
        except json.JSONDecodeError:
            # Handle JSON decoding error
            error_data = {"error": "LLM response was not valid JSON"}
            return error_data, "json_error"
            
        except Exception as e:
            # Handle any other exceptions
            error_data = {"error": f"Extraction failed: {str(e)}"}
            return error_data, "extraction_error"
    
    def _save_extracted_data(self, data: Dict[str, Any]) -> str:
        # Create extractions directory if it doesn't exist
        os.makedirs("extractions", exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"extracted_{timestamp}.json"
        file_path = os.path.join("extractions", filename)
        
        # Save the data as JSON
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        return file_path 