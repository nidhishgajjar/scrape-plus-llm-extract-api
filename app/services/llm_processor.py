import os
from typing import Any, Dict, Literal, Tuple, Optional
import json
from datetime import datetime
import asyncio
import gc
from app.config import get_settings
import litellm
import traceback
from app.utils.logger import setup_logger

ModelType = Literal["gpt-4o", "gpt-4o-mini", "gemini-2.5-flash", "gemini-2.5-pro"]

logger = setup_logger(__name__)

class LLMProcessor:
    def __init__(self, model: ModelType = "gpt-4o-mini", request_id: Optional[str] = None):
        self.model = model
        self.request_id = request_id or "unknown"
        self.settings = get_settings()
        
        logger.debug(f"[{self.request_id}] Initializing LLMProcessor with model: {model}")
        
        # Set up environment variables for litellm
        if model.startswith("gemini"):
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                logger.error(f"[{self.request_id}] GEMINI_API_KEY or GOOGLE_API_KEY not found")
                raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable required for Gemini models")
            os.environ["GEMINI_API_KEY"] = api_key
            # Convert to litellm format
            self.litellm_model = f"gemini/{model}"
            logger.debug(f"[{self.request_id}] Using Gemini model: {self.litellm_model}")
        else:
            # OpenAI models
            if not os.environ.get("OPENAI_API_KEY"):
                logger.error(f"[{self.request_id}] OPENAI_API_KEY not found")
                raise ValueError("OPENAI_API_KEY environment variable required for OpenAI models")
            self.litellm_model = model
            logger.debug(f"[{self.request_id}] Using OpenAI model: {self.litellm_model}")
    
    def _get_max_tokens(self) -> int:
        """Get appropriate max tokens based on model"""
        if self.model.startswith("gemini"):
            return 65535  # Gemini models (both pro and flash)
        elif self.model == "gpt-4o-mini":
            return 16384  # GPT-4o mini
        else:
            return 4096  # GPT-4o and other OpenAI models
    
    async def extract_information(
        self, 
        content: str, 
        extraction_prompt: str, 
        output_format: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], str]:
        # Starting LLM extraction - removed verbose logging
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
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ]
            
            # Sending request to LLM
            
            # Set drop_params for Together AI models
            if "together_ai" in self.litellm_model:
                litellm.drop_params = True
            
            # Use litellm acompletion
            response = await litellm.acompletion(
                model=self.litellm_model,
                messages=messages,
                temperature=0,
                max_tokens=self._get_max_tokens(),
                timeout=300,  # 5 minute timeout
                response_format={"type": "json_object"} if not self.model.startswith("gemini") else None
            )
            
            # Received response from LLM
            
            response_text = response.choices[0].message.content
            
            # Log truncated response for debugging
            truncated_response = response_text[:300] + "..." if len(response_text) > 300 else response_text
            logger.info(f"[{self.request_id}] LLM RESPONSE: {truncated_response}")
            
            # Free up memory after large operations
            gc.collect()
            
            # Try to parse JSON response
            try:
                extracted_data = json.loads(response_text)
            except json.JSONDecodeError as json_err:
                logger.warning(f"[{self.request_id}] Direct JSON parsing failed: {str(json_err)}")
                # If direct parsing fails, try to extract JSON from response
                import re
                logger.debug(f"[{self.request_id}] Attempting to extract JSON from response text...")
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    extracted_data = json.loads(json_match.group())
                    # Successfully extracted JSON from response
                else:
                    logger.error(f"[{self.request_id}] No valid JSON found in response")
                    raise json.JSONDecodeError("No valid JSON found in response", response_text, 0)
            
            file_path = "File saving disabled in production mode"
            
            if self.settings.DEBUG_MODE:
                file_path = self._save_extracted_data(extracted_data)
                # Extracted data saved
            
            # Extraction completed
                
            return extracted_data, file_path
            
        except asyncio.TimeoutError as e:
            # Handle timeout error
            logger.error(f"[{self.request_id}] LLM processing timed out after 300 seconds")
            logger.error(f"[{self.request_id}] Timeout details: {str(e)}")
            error_data = {"error": "LLM processing timed out after 300 seconds", "raw_error": str(e)}
            return error_data, "timeout_error"
            
        except json.JSONDecodeError as e:
            # Handle JSON decoding error
            logger.error(f"[{self.request_id}] LLM response was not valid JSON")
            logger.error(f"[{self.request_id}] JSON Error: {str(e)}")
            logger.error(f"[{self.request_id}] Raw response that failed parsing: {response_text[:1000] if 'response_text' in locals() else 'No response text'}")
            error_data = {"error": f"LLM response was not valid JSON: {str(e)}", "raw_error": str(e)}
            return error_data, "json_error"
            
        except Exception as e:
            # Handle any other exceptions
            logger.error(f"[{self.request_id}] LLM extraction failed with unexpected error")
            logger.error(f"[{self.request_id}] Error type: {type(e).__name__}")
            logger.error(f"[{self.request_id}] Error message: {str(e)}")
            logger.error(f"[{self.request_id}] Full traceback: {traceback.format_exc()}")
            error_data = {"error": f"Extraction failed: {str(e)}", "raw_error": traceback.format_exc()}
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