import os
from typing import Any, Dict, Literal, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json
from urllib.parse import urlparse
from datetime import datetime

ModelType = Literal["gpt-4o", "gpt-4o-mini"]

class LLMProcessor:
    def __init__(self, model: ModelType = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            model=model,
            temperature=0,
        )
        # Create JSON-mode LLM
        self.json_llm = self.llm.bind(
            response_format={"type": "json_object"}
        )
    
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
        
        # Use JSON-mode LLM
        response = await self.json_llm.ainvoke(messages)
        
        print(response)
        
        try:
            extracted_data = json.loads(response.content)
            file_path = self._save_extracted_data(extracted_data)
            return extracted_data, file_path
        except json.JSONDecodeError:
            raise ValueError("LLM response was not valid JSON")
    
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