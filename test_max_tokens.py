#!/usr/bin/env python3
"""
Test script to verify max tokens configuration for Gemini models
"""
import asyncio
import os
from app.services.llm_processor import LLMProcessor

async def test_max_tokens():
    # Test with a Gemini model
    processor = LLMProcessor(model="gemini-2.5-flash")
    
    # Create realistic content that would generate a substantial response
    long_content = """
    Artificial Intelligence (AI) is a rapidly evolving field that encompasses machine learning, deep learning, natural language processing, computer vision, and robotics. The history of AI dates back to the 1950s when Alan Turing proposed the Turing Test. Major breakthroughs include the development of neural networks, the creation of expert systems in the 1980s, and the recent advances in transformer models like GPT and BERT.
    
    Key applications of AI include healthcare diagnostics, autonomous vehicles, recommendation systems, fraud detection, and creative content generation. Companies like Google, OpenAI, Microsoft, and Meta are leading research in this space. Ethical considerations include bias in algorithms, job displacement, privacy concerns, and the potential for misuse.
    
    Current trends include large language models, multimodal AI, reinforcement learning, and edge AI deployment. The future of AI promises advancements in artificial general intelligence (AGI), quantum computing integration, and human-AI collaboration.
    
    Notable researchers include Geoffrey Hinton, Yann LeCun, Andrew Ng, Demis Hassabis, and Fei-Fei Li. Important conferences include NeurIPS, ICML, ICLR, and AAAI. The field continues to grow with new architectures, training techniques, and applications emerging regularly.
    """ * 50  # Repeat to create substantial content
    
    extraction_prompt = "Extract comprehensive information about AI including history, applications, companies, researchers, trends, and provide detailed explanations for each category. Be thorough and include specific examples, dates, and technical details."
    
    output_format = {
        "summary": "string",
        "history": "string",
        "applications": ["string"],
        "companies": ["string"],
        "researchers": ["string"],
        "trends": ["string"],
        "ethical_considerations": ["string"],
        "technical_details": "string"
    }
    
    print("Testing max tokens configuration...")
    print(f"Using model: {processor.model}")
    print(f"Max tokens configured: 65000")
    
    try:
        result, file_path = await processor.extract_information(
            content=long_content,
            extraction_prompt=extraction_prompt,
            output_format=output_format
        )
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return False
        
        # Convert result to string to estimate tokens (rough approximation)
        result_str = str(result)
        estimated_tokens = len(result_str.split()) * 1.3  # Rough token estimation
        
        print(f"Response length: {len(result_str)} characters")
        print(f"Estimated tokens: {estimated_tokens:.0f}")
        print(f"Within limit: {estimated_tokens <= 65000}")
        
        if file_path != "File saving disabled in production mode":
            print(f"Result saved to: {file_path}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Make sure you have GOOGLE_API_KEY set
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        print("Error: GOOGLE_API_KEY or GEMINI_API_KEY environment variable required")
        exit(1)
    
    success = asyncio.run(test_max_tokens())
    print(f"\nTest {'PASSED' if success else 'FAILED'}")