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

    def _should_use_batch_processing(self, extraction_prompt: str, output_format: Dict[str, Any]) -> bool:
        """Determine if batch processing is needed based on prompt and format"""
        # Check if this is a job URL extraction that might return many results
        job_extraction_keywords = ["job_posting_urls", "jobs", "urls", "links"]
        prompt_lower = extraction_prompt.lower()
        
        # Use batch processing if:
        # 1. Output format expects arrays of URLs/jobs
        # 2. Prompt asks for "ALL" or "multiple" items
        # 3. Likely to return many results
        has_array_output = any(key in str(output_format) for key in job_extraction_keywords)
        asks_for_all = any(word in prompt_lower for word in ["all", "multiple", "every", "each"])
        
        return has_array_output and asks_for_all

    def _create_batch_prompts(self, extraction_prompt: str, output_format: Dict[str, Any], total_expected: int = 50) -> List[Dict[str, Any]]:
        """Create batch prompts for processing large extractions"""
        batch_size = 15  # Process 15 items per batch to avoid truncation
        
        batches = []
        for i in range(0, total_expected, batch_size):
            start_idx = i + 1
            end_idx = min(i + batch_size, total_expected)
            
            batch_prompt = f"""
            {extraction_prompt}
            
            IMPORTANT: Extract ONLY items {start_idx} to {end_idx} from the content.
            If fewer items exist, extract what's available.
            Ensure complete JSON response without truncation.
            """
            
            batches.append({
                "prompt": batch_prompt,
                "batch_number": (i // batch_size) + 1,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "expected_count": end_idx - start_idx + 1
            })
        
        return batches

    async def _process_batch(self, content: str, batch_info: Dict[str, Any], output_format: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """Process a single batch of extraction"""
        try:
            messages = [
                {"role": "system", "content": self._get_system_prompt(output_format)},
                {"role": "user", "content": batch_info["prompt"]},
                {"role": "user", "content": content}
            ]
            
            # Set drop_params for Together AI models
            if getattr(self, "_using_together", False):
                litellm.drop_params = True
            
            # Let LLM use default max_tokens to prevent truncation
            request_kwargs = {
                "model": self.litellm_model,
                "messages": messages,
                "temperature": 0,
                "timeout": 120,  # Shorter timeout for batches
            }
            
            # Only OpenAI-compatible providers support response_format
            if not (self.model.startswith("gemini") or self.model.startswith("claude") or self.model.startswith("grok")):
                request_kwargs["response_format"] = {"type": "json_object"}

            response = await litellm.acompletion(**request_kwargs)
            response_text = response.choices[0].message.content
            
            # Validate response quality
            if not response_text or response_text.strip() == "":
                return {"error": f"Batch {batch_info['batch_number']} returned empty response"}, "empty_response_error"
            
            if len(response_text) < 10:
                return {"error": f"Batch {batch_info['batch_number']} response too short"}, "short_response_error"
            
            # Parse JSON response
            try:
                extracted_data = json.loads(response_text)
                return extracted_data, "success"
            except json.JSONDecodeError as json_err:
                logger.warning(f"[{self.request_id}] Batch {batch_info['batch_number']} initial JSON parsing failed, attempting recovery...")
                # Try to clean and extract JSON
                import re
                
                # First, try to extract JSON object
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        # Clean up common JSON formatting issues
                        cleaned_json = json_match.group()
                        # Fix missing commas between array elements or object properties
                        cleaned_json = re.sub(r'}(\s*)\{', r'},\1{', cleaned_json)
                        cleaned_json = re.sub(r'"(\s*)"', r'",\1"', cleaned_json)
                        cleaned_json = re.sub(r'](\s*)\[', r'],\1[', cleaned_json)
                        
                        extracted_data = json.loads(cleaned_json)
                        logger.info(f"[{self.request_id}] Batch {batch_info['batch_number']} JSON recovered successfully")
                        return extracted_data, "success"
                    except json.JSONDecodeError:
                        # If cleaning didn't help, try to extract valid JSON fragments
                        logger.warning(f"[{self.request_id}] Batch {batch_info['batch_number']} JSON recovery failed")
                        pass
                
                return {"error": f"Batch {batch_info['batch_number']} failed: {str(json_err)}"}, "json_error"
                    
        except Exception as e:
            logger.error(f"[{self.request_id}] Batch {batch_info['batch_number']} processing failed: {str(e)}")
            return {"error": f"Batch {batch_info['batch_number']} failed: {str(e)}"}, "batch_error"

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
        - Return ONLY valid JSON - no markdown, no extra text
        - All string values must be properly escaped for JSON
        - Replace newlines with \\n, tabs with \\t
        - Escape quotes with \\ (e.g., "description": "He said \\"Hello\\"")
        - No markdown formatting (##, **, etc.) in JSON values
        - All values must be valid JSON data types
        - Ensure the response starts with {{ and ends with }}
        - IMPORTANT: Ensure proper comma placement between all array elements and object properties
        - Double-check JSON syntax before responding
        
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
        
        # Check if batch processing is needed
        if self._should_use_batch_processing(extraction_prompt, output_format):
            logger.info(f"[{self.request_id}] Using batch processing for large extraction")
            return await self._extract_information_batched(content, extraction_prompt, output_format)
        else:
            logger.info(f"[{self.request_id}] Using single-pass extraction")
            return await self._extract_information_single(content, extraction_prompt, output_format)

    async def _extract_information_batched(self, content: str, extraction_prompt: str, output_format: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """Extract information using batch processing to avoid truncation"""
        try:
            # Create batch prompts
            batches = self._create_batch_prompts(extraction_prompt, output_format)
            logger.info(f"[{self.request_id}] Created {len(batches)} batches for processing")
            
            # Process each batch
            batch_results = []
            successful_batches = 0
            
            for batch_info in batches:
                logger.info(f"[{self.request_id}] Processing batch {batch_info['batch_number']}/{len(batches)}")
                
                batch_data, batch_status = await self._process_batch(content, batch_info, output_format)
                
                if batch_status == "success":
                    batch_results.append(batch_data)
                    successful_batches += 1
                    logger.info(f"[{self.request_id}] Batch {batch_info['batch_number']} completed successfully")
                else:
                    logger.warning(f"[{self.request_id}] Batch {batch_info['batch_number']} failed: {batch_data.get('error', 'Unknown error')}")
                
                # Small delay between batches
                await asyncio.sleep(0.5)
            
            # Combine batch results
            if successful_batches == 0:
                return {"error": "All batches failed to process"}, "all_batches_failed"
            
            # Merge results based on output format
            final_result = self._merge_batch_results(batch_results, output_format)
            
            logger.info(f"[{self.request_id}] Batch processing completed: {successful_batches}/{len(batches)} batches successful")
            
            # Save combined result
            file_path = "File saving disabled in production mode"
            if self.settings.DEBUG_MODE:
                file_path = self._save_extracted_data(final_result)
            
            return final_result, file_path
            
        except Exception as e:
            logger.error(f"[{self.request_id}] Batch processing failed: {str(e)}")
            return {"error": f"Batch processing failed: {str(e)}"}, "batch_processing_error"

    def _merge_batch_results(self, batch_results: List[Dict[str, Any]], output_format: Dict[str, Any]) -> Dict[str, Any]:
        """Merge results from multiple batches into final output"""
        merged_result = {}
        
        for batch_data in batch_results:
            for key, value in batch_data.items():
                if key not in merged_result:
                    merged_result[key] = value
                elif isinstance(value, list) and isinstance(merged_result[key], list):
                    # Merge arrays (e.g., job_posting_urls)
                    merged_result[key].extend(value)
                elif isinstance(value, dict) and isinstance(merged_result[key], dict):
                    # Merge objects
                    merged_result[key].update(value)
                # For other types, keep the last value
        
        # Remove duplicates from arrays
        for key, value in merged_result.items():
            if isinstance(value, list):
                merged_result[key] = list(dict.fromkeys(value))  # Preserve order while removing duplicates
        
        # Add batch processing metadata
        merged_result["_batch_processing"] = {
            "total_batches": len(batch_results),
            "successful_batches": len(batch_results),
            "processing_method": "chunked_batch"
        }
        
        return merged_result

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
            
            # Use litellm acompletion without max_tokens - let LLM decide
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
            
        except litellm.exceptions.ServiceUnavailableError as e:
            # Handle service unavailable
            logger.error(f"[{self.request_id}] LLM service unavailable: {str(e)}")
            error_data = {"error": "LLM service temporarily unavailable. Please try again later.", "raw_error": str(e)}
            return error_data, "service_unavailable_error"
            
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