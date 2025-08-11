#!/usr/bin/env python3
"""
Test script to verify litellm integration for all supported models
"""
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from app.services.llm_processor import LLMProcessor

async def test_model(model_name: str):
    """Test a specific model"""
    print(f"\n{'='*50}")
    print(f"Testing {model_name}")
    print(f"{'='*50}")
    
    try:
        processor = LLMProcessor(model=model_name)
        
        # Test content about AI
        content = """
        Artificial Intelligence (AI) is revolutionizing various industries. Machine learning algorithms 
        can now process vast amounts of data to identify patterns and make predictions. Deep learning, 
        a subset of machine learning, uses neural networks to solve complex problems. Natural language 
        processing enables computers to understand and generate human language. Computer vision allows 
        machines to interpret visual information. These technologies are being applied in healthcare 
        for diagnostic imaging, in finance for fraud detection, and in transportation for autonomous 
        vehicles. The future of AI promises even more advanced capabilities with the development of 
        artificial general intelligence (AGI). Companies like OpenAI, Google, Anthropic, and xAI are 
        leading the development of cutting-edge AI models including GPT-4, Gemini, Claude, and Grok.
        """
        
        extraction_prompt = """
        Extract key information about AI technologies, applications, industries, and companies mentioned in the text.
        Focus on technologies, applications, industries, and leading AI companies.
        """
        
        output_format = {
            "technologies": ["string"],
            "applications": ["string"],
            "industries": ["string"],
            "companies": ["string"],
            "future_developments": ["string"]
        }
        
        print(f"Max tokens configured: {processor._get_max_tokens()}")
        print("Processing request...")
        
        result, file_path = await processor.extract_information(
            content=content,
            extraction_prompt=extraction_prompt,
            output_format=output_format
        )
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return False
        
        print("✅ Success!")
        print(f"Technologies found: {len(result.get('technologies', []))}")
        print(f"Applications found: {len(result.get('applications', []))}")
        print(f"Industries found: {len(result.get('industries', []))}")
        print(f"Companies found: {len(result.get('companies', []))}")
        print(f"Future developments found: {len(result.get('future_developments', []))}")
        
        # Print first few items from each category
        for key, items in result.items():
            if isinstance(items, list) and items:
                print(f"{key.title()}: {items[:3]}{'...' if len(items) > 3 else ''}")
        
        if file_path != "File saving disabled in production mode":
            print(f"Result saved to: {file_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        if hasattr(e, '__traceback__'):
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
        return False

async def main():
    print("Testing LiteLLM Integration")
    print("=" * 50)
    
    # Test models
    models_to_test = []
    
    # Check which models we can test based on available API keys
    if os.getenv("OPENAI_API_KEY"):
        models_to_test.extend(["gpt-4o-mini", "gpt-4o"])
    
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        models_to_test.extend(["gemini-2.5-flash", "gemini-2.5-pro"])
    
    if os.getenv("ANTHROPIC_API_KEY"):
        models_to_test.extend(["claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219", "claude-3-7-sonnet-latest", "claude-3-5-haiku-20241022", "claude-3-5-haiku-latest"])
    
    if os.getenv("XAI_API_KEY"):
        models_to_test.extend(["grok-4", "grok-4-latest"])
    
    if not models_to_test:
        print("❌ No API keys found. Please set one of the following:")
        print("   - OPENAI_API_KEY (for GPT models)")
        print("   - GEMINI_API_KEY or GOOGLE_API_KEY (for Gemini models)")
        print("   - ANTHROPIC_API_KEY (for Claude models)")
        print("   - XAI_API_KEY (for Grok models)")
        return
    
    print(f"Testing {len(models_to_test)} models: {', '.join(models_to_test)}")
    
    results = {}
    for model in models_to_test:
        results[model] = await test_model(model)
    
    # Summary
    print(f"\n{'='*50}")
    print("Test Summary")
    print(f"{'='*50}")
    
    for model, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{model:30} {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")

if __name__ == "__main__":
    asyncio.run(main())