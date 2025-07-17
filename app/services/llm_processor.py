import os
from typing import Any, Dict, Literal, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json
from urllib.parse import urlparse
from datetime import datetime
import httpx
import asyncio
import gc
from app.config import get_settings

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

ModelType = Literal["gpt-4o", "gpt-4o-mini", "gemini-2.0-flash-exp", "gemini-2.5-pro-preview-06-05"]

class LLMProcessor:
    def __init__(self, model: ModelType = "gpt-4o-mini"):
        self.model = model
        self.settings = get_settings()
        
        # Determine if this is a Gemini or OpenAI model
        if model.startswith("gemini"):
            if not GEMINI_AVAILABLE:
                raise ImportError("google-genai package not installed. Run: pip install google-genai")
            
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            
            if not api_key:
                raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable required for Gemini models")
            
            self.client_type = "gemini"
            self.gemini_client = genai.Client(api_key=api_key)
            self.llm = None
            self.json_llm = None
        else:
            # OpenAI models
            timeout = httpx.Timeout(60.0, connect=10.0)
            http_client = httpx.Client(timeout=timeout)
            
            self.client_type = "openai"
            self.gemini_client = None
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
        
        try:
            if self.client_type == "gemini":
                # Use Gemini API
                prompt = f"{system_prompt}\n\nContent to analyze:\n{content}"
                
                response = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.gemini_client.models.generate_content(
                            model=self.model,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                max_output_tokens=65000,
                                temperature=0.0,
                            )
                        )
                    ),
                    timeout=300  # 5 minute timeout
                )
                
                response_text = response.text
            else:
                # Use OpenAI API via LangChain
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=content)
                ]
                
                response = await asyncio.wait_for(
                    self.json_llm.ainvoke(messages),
                    timeout=120  # 2 minute timeout
                )
                
                response_text = response.content
                messages = None  # Free memory
            
            # Free up memory after large operations
            gc.collect()
            
            print(f"LLM Response: {response_text}")
            
            # Try to parse JSON response
            try:
                extracted_data = json.loads(response_text)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    extracted_data = json.loads(json_match.group())
                else:
                    raise json.JSONDecodeError("No valid JSON found in response", response_text, 0)
            
            file_path = "File saving disabled in production mode"
            
            if self.settings.DEBUG_MODE:
                file_path = self._save_extracted_data(extracted_data)
                
            return extracted_data, file_path
            
        except asyncio.TimeoutError:
            # Handle timeout error
            error_data = {"error": "LLM processing timed out after 120 seconds"}
            return error_data, "timeout_error"
            
        except json.JSONDecodeError as e:
            # Handle JSON decoding error
            error_data = {"error": f"LLM response was not valid JSON: {str(e)}"}
            return error_data, "json_error"
            
        except Exception as e:
            # Handle any other exceptions
            error_data = {"error": f"Extraction failed: {str(e)}"}
            return error_data, "extraction_error"
        finally:
            # Always try to free memory
            gc.collect()
    
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