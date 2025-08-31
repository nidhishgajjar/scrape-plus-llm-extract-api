import os
from typing import Any, Dict, Literal, Tuple, Optional, List
import json
from datetime import datetime
import asyncio
import gc
from app.config import get_settings
import litellm
import traceback
from app.utils.logger import setup_logger

ModelType = Literal["gpt-4o", "gpt-4o-mini", "gemini-2.5-flash", "gemini-2.5-pro", "claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219", "claude-3-7-sonnet-latest", "claude-3-5-haiku-20241022", "claude-3-5-haiku-latest", "grok-4", "grok-4-latest", "gpt-oss-20b", "gpt-oss-120b"]

logger = setup_logger(__name__)

class LLMProcessor:
    def __init__(self, model: ModelType = "gpt-4o-mini", request_id: Optional[str] = None):
        self.model = model
        self.original_model = model  # Store original for fallback tracking
        self.request_id = request_id or "unknown"
        self.settings = get_settings()
        self.using_fallback = False
        
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
        elif model.startswith("claude"):
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                logger.error(f"[{self.request_id}] ANTHROPIC_API_KEY not found")
                raise ValueError("ANTHROPIC_API_KEY environment variable required for Anthropic models")
            os.environ["ANTHROPIC_API_KEY"] = api_key
            # Convert to litellm format
            self.litellm_model = f"anthropic/{model}"
            logger.debug(f"[{self.request_id}] Using Anthropic model: {self.litellm_model}")
        elif model.startswith("grok"):
            api_key = os.environ.get("XAI_API_KEY")
            if not api_key:
                logger.error(f"[{self.request_id}] XAI_API_KEY not found")
                raise ValueError("XAI_API_KEY environment variable required for Grok models")
            os.environ["XAI_API_KEY"] = api_key
            # Convert to litellm format
            self.litellm_model = f"xai/{model}"
            logger.debug(f"[{self.request_id}] Using Grok model: {self.litellm_model}")
        elif model.startswith("gpt-oss"):
            api_key = os.environ.get("TOGETHER_API_KEY") or os.environ.get("TOGETHERAI_API_KEY")
            if not api_key:
                logger.error(f"[{self.request_id}] TOGETHER_API_KEY or TOGETHERAI_API_KEY not found")
                raise ValueError("TOGETHER_API_KEY or TOGETHERAI_API_KEY environment variable required for TogetherAI models")
            # Ensure env var is set for LiteLLM's together_ai provider
            os.environ["TOGETHER_API_KEY"] = api_key
            # If the model starts with 'gpt', Together requires the 'openai/' prefix under their provider
            together_model = f"openai/{model}" if model.startswith("gpt") else model
            # Use LiteLLM provider route: together_ai/{resolved_model}
            self.litellm_model = f"together_ai/{together_model}"
            self._using_together = True
            logger.debug(f"[{self.request_id}] Using TogetherAI model: {self.litellm_model}")
        else:
            # OpenAI models
            if not os.environ.get("OPENAI_API_KEY"):
                logger.error(f"[{self.request_id}] OPENAI_API_KEY not found")
                raise ValueError("OPENAI_API_KEY environment variable required for OpenAI models")
            self.litellm_model = model
            logger.debug(f"[{self.request_id}] Using OpenAI model: {self.litellm_model}")

    def _get_system_prompt(self, output_format: Dict[str, Any]) -> str:
        """Get the system prompt for extraction"""
        return f"""
        You are a precise data extraction assistant. Your task is to:
        1. Analyze the provided content
        2. Extract specific information based on the given prompt
        3. Return the information in the exact JSON format specified
        
        Output Format Example:
        {json.dumps(output_format, indent=2)}
        
        CRITICAL JSON FORMATTING RULES:
        - Return ONLY valid JSON - no markdown code blocks, no ```json wrapper, no extra text
        - Your response MUST start with "{{" and end with "}}"
        - DO NOT wrap the JSON in markdown code blocks (no ```json...```)
        - All string values must be properly escaped for JSON
        - Replace newlines with \\n, tabs with \\t
        - Escape quotes with \\ (e.g., "description": "He said \\"Hello\\"")
        - No markdown formatting (##, **, etc.) in JSON values
        - All values must be valid JSON data types
        - IMPORTANT: Ensure proper comma placement between all array elements and object properties
        - Double-check JSON syntax before responding
        - REMEMBER: Return RAW JSON only, starting with "{{" and ending with "}}"
        
        Rules:
        - Strictly follow the output format
        - If a field cannot be found, use null
        - Do not include additional fields
        - Ensure valid JSON output
        - For URLs: 
          * Include complete URLs only
          * Ensure URLs are properly formatted
          * Remove any duplicates
          * Validate URLs match the specified criteria
        """

    async def extract_information(
        self, 
        content: str, 
        extraction_prompt: str, 
        output_format: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], str]:
        # Starting LLM extraction - removed verbose logging
        logger.info(f"[{self.request_id}] Starting extraction")
        return await self._extract_information_single(content, extraction_prompt, output_format)

    async def _extract_information_single(self, content: str, extraction_prompt: str, output_format: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """Original single-pass extraction method"""
        system_prompt = self._get_system_prompt(output_format)
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": extraction_prompt},
                {"role": "user", "content": content}
            ]
            
            # Sending request to LLM
            
            # Set drop_params for Together AI models
            if getattr(self, "_using_together", False):
                litellm.drop_params = True
            
            # Use litellm acompletion without max_tokens - let LLM use its default
            request_kwargs = {
                "model": self.litellm_model,
                "messages": messages,
                "temperature": 0,
                "timeout": 300,
            }
            # Only OpenAI-compatible providers support response_format
            if not (self.model.startswith("gemini") or self.model.startswith("claude") or self.model.startswith("grok")):
                request_kwargs["response_format"] = {"type": "json_object"}

            # For together_ai provider via env, no need to override api_base/api_key here

            response = await litellm.acompletion(**request_kwargs)
            
            # Received response from LLM
            
            response_text = response.choices[0].message.content
            
            # Validate response quality
            if not response_text or response_text.strip() == "":
                logger.error(f"[{self.request_id}] LLM returned empty response")
                return {"error": "LLM returned empty response - possible context overflow or API issue"}, "empty_response_error"
            
            if len(response_text) < 10:  # Very short response
                logger.warning(f"[{self.request_id}] LLM returned very short response: {response_text}")
                return {"error": "LLM response too short - possible context issues or incomplete generation"}, "short_response_error"
            
            # Log truncated response for debugging - show beginning and end
            if len(response_text) > 300:
                truncated_response = f"{response_text[:150]}...MIDDLE_TRUNCATED...{response_text[-150:]}"
            else:
                truncated_response = response_text
            logger.info(f"[{self.request_id}] LLM RESPONSE: {truncated_response}")
            
            # Free up memory after large operations
            gc.collect()
            
            # Try to parse JSON response - handle both raw JSON and markdown-wrapped JSON
            extracted_data = None
            
            # First, try direct JSON parsing (for raw JSON responses)
            try:
                extracted_data = json.loads(response_text)
                logger.debug(f"[{self.request_id}] Successfully parsed raw JSON response")
            except json.JSONDecodeError as json_err:
                # If direct parsing fails, likely wrapped in markdown code blocks or incomplete
                logger.debug(f"[{self.request_id}] Direct JSON parsing failed: {str(json_err)}")
                logger.debug(f"[{self.request_id}] Response length: {len(response_text)} chars")
                
                # Check if response appears to be truncated (starts with JSON but incomplete)
                if response_text.strip().startswith('{') and not response_text.strip().endswith('}'):
                    logger.error(f"[{self.request_id}] Response appears to be truncated JSON (starts with {{ but doesn't end with }})")
                    logger.error(f"[{self.request_id}] Last 100 chars: ...{response_text[-100:]}")
                    raise json.JSONDecodeError("Response appears to be incomplete/truncated JSON", response_text, len(response_text))
                
                # Try to extract JSON from markdown code blocks or find JSON object
                import re
                
                # First try to remove markdown code blocks if present
                cleaned_text = response_text.strip()
                if cleaned_text.startswith('```json'):
                    # Remove opening ```json
                    cleaned_text = cleaned_text[7:]
                    # Remove closing ```
                    if cleaned_text.endswith('```'):
                        cleaned_text = cleaned_text[:-3]
                    cleaned_text = cleaned_text.strip()
                    
                    try:
                        extracted_data = json.loads(cleaned_text)
                        logger.info(f"[{self.request_id}] Successfully extracted JSON from markdown code block")
                    except json.JSONDecodeError:
                        logger.debug(f"[{self.request_id}] Failed to parse cleaned markdown text, trying regex extraction")
                
                # If still no success, try regex extraction as fallback
                if extracted_data is None:
                    logger.debug(f"[{self.request_id}] Attempting regex extraction of JSON object...")
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        try:
                            extracted_data = json.loads(json_match.group())
                            logger.info(f"[{self.request_id}] Successfully extracted JSON using regex")
                        except json.JSONDecodeError as e:
                            logger.error(f"[{self.request_id}] Regex-extracted text is not valid JSON: {str(e)}")
                            raise json.JSONDecodeError("Extracted text is not valid JSON", response_text, 0)
                    else:
                        logger.error(f"[{self.request_id}] No JSON object pattern found in response")
                        raise json.JSONDecodeError("No valid JSON found in response", response_text, 0)
            
            file_path = "File saving disabled in production mode"
            
            if self.settings.DEBUG_MODE:
                file_path = self._save_extracted_data(extracted_data)
                # Extracted data saved
            
            # Extraction completed
                
            return extracted_data, file_path
            
        except (litellm.exceptions.InternalServerError, litellm.exceptions.ServiceUnavailableError) as e:
            # Handle Together.ai failures with Gemini fallback
            if self.model.startswith("gpt-oss") and not self.using_fallback:
                logger.warning(f"[{self.request_id}] Together.ai failed, falling back to Gemini Flash")
                logger.warning(f"[{self.request_id}] Original error: {str(e)}")
                
                # Switch to Gemini Flash
                self.model = "gemini-2.5-flash"
                self.using_fallback = True
                
                # Reinitialize with Gemini
                api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                if api_key:
                    os.environ["GEMINI_API_KEY"] = api_key
                    self.litellm_model = f"gemini/{self.model}"
                    
                    # Retry with Gemini
                    logger.info(f"[{self.request_id}] Retrying with Gemini Flash...")
                    result = await self._extract_information_single(content, extraction_prompt, output_format)
                    
                    # Add fallback info to result if successful
                    if isinstance(result[0], dict) and "error" not in result[0]:
                        result[0]["_fallback_used"] = {
                            "original_model": self.original_model,
                            "fallback_model": "gemini-2.5-flash",
                            "reason": "Together.ai internal server error"
                        }
                    
                    return result
                else:
                    logger.error(f"[{self.request_id}] Gemini API key not found for fallback")
                    error_data = {"error": "Primary model failed and fallback unavailable", "raw_error": str(e)}
                    return error_data, "fallback_unavailable_error"
            else:
                # Not a Together.ai model or already using fallback
                logger.error(f"[{self.request_id}] Service unavailable: {str(e)}")
                error_data = {"error": "LLM service temporarily unavailable. Please try again later.", "raw_error": str(e)}
                return error_data, "service_unavailable_error"
            
        except litellm.exceptions.BadRequestError as e:
            # Handle specific token limit errors
            if "token count exceeds" in str(e).lower() or "context length" in str(e).lower():
                logger.error(f"[{self.request_id}] Token limit exceeded: {str(e)}")
                error_data = {"error": "Content too long for this model. Please reduce input size or use a model with larger context.", "raw_error": str(e)}
                return error_data, "token_limit_error"
            else:
                logger.error(f"[{self.request_id}] Bad request error: {str(e)}")
                error_data = {"error": f"Bad request to LLM service: {str(e)}", "raw_error": str(e)}
                return error_data, "bad_request_error"
                
        except litellm.exceptions.RateLimitError as e:
            # Handle rate limiting
            logger.error(f"[{self.request_id}] Rate limit exceeded: {str(e)}")
            error_data = {"error": "Rate limit exceeded. Please try again later.", "raw_error": str(e)}
            return error_data, "rate_limit_error"
            
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
            
            if 'response_text' in locals():
                # Log first and last parts of response for better debugging
                if len(response_text) > 2000:
                    logger.error(f"[{self.request_id}] Response preview (first 500 chars): {response_text[:500]}")
                    logger.error(f"[{self.request_id}] Response preview (last 500 chars): ...{response_text[-500:]}")
                else:
                    logger.error(f"[{self.request_id}] Full response that failed parsing: {response_text}")
                
                # Check for common issues
                if "incomplete" in str(e).lower() or "truncated" in str(e).lower():
                    error_msg = "LLM response appears to be truncated/incomplete JSON"
                else:
                    error_msg = f"LLM response was not valid JSON: {str(e)}"
            else:
                logger.error(f"[{self.request_id}] No response text available")
                error_msg = "No response text available from LLM"
            
            error_data = {"error": error_msg, "raw_error": str(e)}
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
            
            # Force cleanup of any remaining sessions if possible
            try:
                if hasattr(litellm, '_cleanup'):
                    await litellm._cleanup()
            except:
                pass
            
            # Allow pending tasks to complete
            await asyncio.sleep(0.1)
    
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