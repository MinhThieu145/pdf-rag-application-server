"""
Evidence Extraction Helper

This module contains helper functions for extracting and processing evidence from research papers.
It handles the core logic for evidence extraction, including text processing and GPT-4 analysis.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import os
import boto3
from botocore.exceptions import ClientError
from fastapi import HTTPException
from openai import OpenAI
from pydantic import ValidationError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

# Initialize clients
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

openai_client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

async def process_raw_evidence(request_file_name: str, request_essay_topic: str, extraction_response_model: Any) -> Dict[str, Any]:
    """
    Process and extract evidence from a research paper using GPT-4.
    
    Args:
        request_file_name: Name of the file to process
        request_essay_topic: Topic to analyze evidence against
        extraction_response_model: Pydantic model for validating the response
        
    Returns:
        Dict containing the processed evidence and metadata
    """
    try:
        logger.info(f"Starting evidence extraction for file: {request_file_name}")
        logger.info(f"Essay topic: {request_essay_topic}")
        
        # Get parsed JSON data
        document_name = Path(request_file_name).stem
        parsed_json_key = f"documents/{document_name}/parsed_json/{request_file_name.replace('.pdf', '.json')}"
        research_paper = await _get_parsed_json_content(parsed_json_key)
        
        # First step: Analyze the topic
        topic_response = await _analyze_topic(research_paper, request_essay_topic)
        topic_analysis = json.loads(topic_response.choices[0].message.content)
        
        # Second step: Analyze evidence based on topic analysis
        evidence_response = await _analyze_evidence(topic_analysis, research_paper)
        evidence_analysis = json.loads(evidence_response.choices[0].message.content)        
        # Third step: Verify and improve evidence
        verified_response = await _verify_evidence(evidence_analysis, research_paper)
        
        # Parse and validate response
        result = await _parse_and_validate_response(
            verified_response, 
            request_file_name, 
            request_essay_topic, 
            extraction_response_model
        )
        
        # Save to S3
        await _save_evidence_to_s3(result, document_name, request_file_name)
        
        return result
        
    except Exception as e:
        logger.info(f"Error processing evidence: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process evidence: {str(e)}"
        )

async def _get_parsed_json_content(parsed_json_key: str) -> str:
    """Get and process parsed JSON content from S3."""
    try:
        logger.info(f"Attempting to read from S3 bucket: {os.getenv('AWS_BUCKET_NAME')}")
        json_response = s3_client.get_object(
            Bucket=os.getenv('AWS_BUCKET_NAME'),
            Key=parsed_json_key
        )
        parsed_data = json.loads(json_response['Body'].read().decode('utf-8'))
        logger.info("Successfully read and parsed JSON from S3")
        
        research_paper = ""
        if 'documents' in parsed_data:
            research_paper = "\n\n".join(parsed_data['documents'])
            logger.info(f"Extracted text length: {len(research_paper)}")
            logger.info(f"First 500 chars of text: {research_paper[:500]}...")
        else:
            logger.info("No 'documents' field found in parsed data")
            logger.info(f"Available fields: {list(parsed_data.keys())}")
            
        return research_paper
            
    except ClientError as e:
        logger.info(f"Failed to read parsed JSON from S3: {str(e)}")
        logger.info(f"S3 Error Code: {e.response['Error']['Code']}")
        logger.info(f"S3 Error Message: {e.response['Error']['Message']}")
        raise HTTPException(
            status_code=404,
            detail=f"Could not find parsed content for file"
        )
    except Exception as e:
        logger.info("Failed to extract text from parsed JSON", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract text from parsed JSON: {str(e)}"
        )

async def _analyze_topic(research_paper: str, essay_topic: str) -> Any:
    """Analyze the topic query using GPT-4."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "adapted_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "original_topic": {
                            "description": "The original topic string",
                            "type": "string"
                        },
                        "key_terms": {
                            "description": "A set of terms with their definitions or explanations",
                            "type": "object",
                            "additionalProperties": {
                                "type": "string"
                            }
                        },
                        "sub_questions": {
                            "description": "A list of sub-questions related to the topic",
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                        "perspectives": {
                            "description": "A list of different perspectives or viewpoints",
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                        "refined_topic": {
                            "description": "A refined version of the original topic",
                            "type": "string"
                        }
                    },
                }
            }
        }
        ,
            messages=[
                {
                    "role": "developer",
                    "content": """You are an advanced AI designed to analyze and refine topic queries for use in argumentative essay writing. Your goal is to deconstruct the topic, clarify its components, and propose sub-questions and perspectives for deeper exploration. Follow these steps precisely:

1. Understand the Core Topic:
- Identify the main idea or claim in the topic query.
- Highlight any ambiguous or broad terms that need clarification.
2. Clarify Key Terms:
- Break the topic into key terms and briefly define each one.
- Suggest alternative phrasing for unclear terms or terms with multiple interpretations.
3. Generate Sub-Questions:
- Propose specific sub-questions related to the topic that can guide further analysis or research.
- Sub-questions should address different aspects, perspectives, or angles of the topic.
4. Explore Perspectives:
- Suggest distinct perspectives or frameworks to analyze the topic.
- These might include causes, effects, challenges, benefits, or broader contexts.
5. Refine the Topic Query:
- Rewrite the topic into a clear, focused, and actionable form based on your analysis.
- Ensure the topic is specific, concise, and relevant to your audience.
- Avoid wordy or complex phrases that may confuse readers."""
                },
                {
                    "role": "user",
                    "content": f"Topic Query: {essay_topic}"
                }
            ],
            temperature=0.1
        )
        logger.info("Successfully received OpenAI topic analysis response")
        return response
        
    except Exception as e:
        logger.info("Failed to get OpenAI completion for topic analysis", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process topic analysis with OpenAI: {str(e)}"
        )

async def _analyze_evidence(topic_analysis: Dict[str, Any], research_paper: str) -> Any:
    """Analyze and verify evidence based on the topic analysis."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "detailed_evidence_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "refined_topic": {
                            "description": "A refined version of the topic",
                            "type": "string"
                        },
                        "contextual_summary": {
                            "description": "A summary providing context for the topic",
                            "type": "string"
                        },
                        "evidence": {
                            "description": "A list of evidence objects with detailed and specific information",
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "exact_text": {
                                        "description": "The exact text from the source, including any citations or additional details",
                                        "type": "string"
                                    },
                                    "category": {
                                        "description": "The type of evidence (support, against, informative)",
                                        "type": "string",
                                        "enum": ["support", "against", "informative"]
                                    },
                                    "reasoning": {
                                        "description": "Details about the reasoning behind the evidence",
                                        "type": "object",
                                        "properties": {
                                            "specific_relevance": {
                                                "description": "How this evidence specifically relates to the refined topic",
                                                "type": "string"
                                            },
                                            "application": {
                                                "description": "Specific situations or sections where this evidence can be applied",
                                                "type": "string"
                                            },
                                            "insights": {
                                                "description": "Detailed insights derived from the evidence",
                                                "type": "string"
                                            }
                                        },
                                        "required": ["specific_relevance", "application", "insights"],
                                    },
                                    "strength_of_evidence": {
                                        "description": "Assessment of the evidence's strength and justification",
                                        "type": "object",
                                        "properties": {
                                            "score": {
                                                "description": "The strength score of the evidence (e.g., high, moderate, low)",
                                                "type": "string"
                                            },
                                            "justification": {
                                                "description": "Detailed justification for the given score",
                                                "type": "string"
                                            }
                                        },
                                        "required": ["score", "justification"],
                                    }
                                },
                                "required": ["exact_text", "category", "reasoning", "strength_of_evidence"],
                            }
                        }
                    },
                }
            }
        },
            messages=[
                {
                    "role": "developer",
                    "content": """You are an advanced AI designed to verify, refine, and improve evidence extracted from a document. Your goal is to ensure the evidence is detailed, specific, and directly relevant to the topic while strictly adhering to the text of the document. Follow these steps:

1. Input Structure
You will receive:
- Evidence Data:
  refined_topic: The focused essay topic or question.
  evidence: A list of evidence entries, each containing:
    exact_text: Extracted evidence from the source.
    category: The evidence type (e.g., Support, Against, or Informative).
    reasoning: Detailed reasoning for why the evidence was included.
    strength_of_evidence: Score (Strong, Moderate, or Weak) and its justification.
- Document: The source text from which the evidence was extracted.

2. Your Task
Step 1: Verify Evidence
- Contextual Review:
  * Locate each piece of evidence in the document.
  * Analyze the paragraph, section, or entire document to determine if the evidence reflects the main argument or significant themes.
- Exact Text Verification:
  * Ensure the exact_text is copied verbatim from the source, including citations, qualifiers, or any nuanced wording.

Step 2: Refine Evidence
- Relevance: Keep only evidence that is directly relevant to the refined topic or sub-questions.
- Detail: Focus on specific and precise portions of text that provide substantial insights or arguments.
- Category Update:
  * Reclassify the category as Support, Against, or Informative.
  * Remove unhelpful labels like "Overview of Benefits."

Step 3: Simplify and Strengthen Reasoning
- Rewrite reasoning to be:
  * Specific: Explain the exact relevance of the evidence to the refined topic.
  * Concise: Use simple, clear language.
- Ensure the reasoning addresses:
  * Why the evidence is important.
  * How it contributes to answering the refined topic or supporting the essay.

Step 4: Strength Evaluation
- Reassess the strength of evidence based on:
  * Specificity: How detailed and precise the evidence is.
  * Relevance: Whether it aligns closely with the refined topic.
  * Credibility: The quality and trustworthiness of the source."""
                },
                {
                    "role": "user",
                    "content": f"Input Topic Query Analysis: {json.dumps(topic_analysis)}. Document: {research_paper}"
                }
            ],
            temperature=0.7,
            max_tokens=4000
        )
        logger.info("Successfully received OpenAI evidence analysis response")
        return response
        
    except Exception as e:
        logger.info("Failed to get OpenAI completion for evidence analysis", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process evidence analysis with OpenAI: {str(e)}"
        )

async def _verify_evidence(evidence_analysis: Dict[str, Any], research_paper: str) -> Any:
    """Verify and improve evidence using GPT-4."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "developer", 
                    "content": '''
You are an advanced AI tasked with verifying and improving evidence extracted from a document. Your goal is to ensure the accuracy, contextual relevance, and clarity of the evidence while simplifying reasoning for ease of understanding. Follow these steps:

Input Structure
You will receive the following:

Evidence Data:
refined_topic: The refined topic or question.
evidence: A list of evidence entries, each containing:
exact_text: The text extracted as evidence.
category: The type of evidence (Support, Against, or Informative).
reasoning: Detailed reasoning provided for the evidence.
strength_of_evidence: A score (Strong, Moderate, or Weak) and its justification.
Document: The full text of the document from which the evidence was extracted.

Your Task
1. Verify Evidence
Contextual Validation:
Locate each piece of evidence within the document.
Review the entire paragraph and broader section containing the evidence to verify its relevance.
Ensure the evidence aligns with the main themes or significant points of the document, not minor side notes or footnotes.

Exact Text Confirmation:
Ensure the exact_text matches the text from the source document. Correct any discrepancies.

2. Assess and Improve Reasoning
Reassess the provided reasoning to confirm its validity within the broader context of the document.
Adjust reasoning if it fails to capture the document's intent, is overly complex, or lacks sufficient depth.
Ensure the reasoning explains:
Why the evidence is relevant to the refined topic.
Why the evidence was chosen over other text.

3. Simplify Reasoning for Clarity
Rewrite the reasoning in clear, simple language to make it easier to understand. Use very simple language and wording that is easy to understand for most readers.
Focus on key points, avoiding jargon or overly technical terms.
Ensure the simplified reasoning communicates:
The importance of the evidence.
Why it is relevant to the refined topic or argument.

4. Output Results
Provide the verified and improved evidence in a structured JSON format, with a field for simplified reasoning.
'''
                },
                {
                    "role": "user", 
                    "content": f"Evidence Data: {json.dumps(evidence_analysis)}. Document: {research_paper}"
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "verified_evidence_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "refined_topic": {
                                "description": "The refined version of the topic being discussed.",
                                "type": "string"
                            },
                            "verified_evidence": {
                                "description": "A list of verified evidence with exact text, category, simplified reasoning, and strength of evidence.",
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "exact_text": {
                                            "description": "The exact text from the source document, verified for accuracy.",
                                            "type": "string"
                                        },
                                        "category": {
                                            "description": "The type of evidence, indicating whether it supports, opposes, or provides context to the topic.",
                                            "type": "string",
                                            "enum": ["Support", "Against", "Informative"]
                                        },
                                        "simplified_reasoning": {
                                            "description": "A simplified explanation of why the evidence is relevant and important to the topic.",
                                            "type": "string"
                                        },
                                        "strength_of_evidence": {
                                            "description": "An assessment of the strength and relevance of the evidence.",
                                            "type": "object",
                                            "properties": {
                                                "score": {
                                                    "description": "The strength score of the evidence, such as 'Strong', 'Moderate', or 'Weak'.",
                                                    "type": "string"
                                                },
                                                "justification": {
                                                    "description": "A detailed explanation justifying the assigned strength score.",
                                                    "type": "string"
                                                }
                                            },
                                            "required": ["score", "justification"]
                                        }
                                    },
                                    "required": ["exact_text", "category", "simplified_reasoning", "strength_of_evidence"]
                                }
                            }
                        },
                        "required": ["refined_topic", "verified_evidence"]
                    }
                }
            }
        )
        logger.info("Successfully received OpenAI evidence verification response")
        return response
        
    except Exception as e:
        logger.info("Failed to get OpenAI completion for evidence verification", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process evidence verification with OpenAI: {str(e)}"
        )

async def _parse_and_validate_response(
    response: Any, 
    file_name: str, 
    essay_topic: str, 
    extraction_response_model: Any
) -> Dict[str, Any]:
    """Parse and validate the OpenAI response."""
    try:
        content = json.loads(response.choices[0].message.content)
        logger.info("Successfully parsed OpenAI response JSON")
        
        validated_content = extraction_response_model(**content)
        logger.info("Successfully validated response against schema")

        return {
            "timestamp": datetime.now().isoformat(),
            "file_name": file_name,
            "essay_topic": essay_topic,
            "analysis": validated_content.model_dump(),
            "metadata": {
                "model": response.model,
                "tokens_used": response.usage.total_tokens,
                "processing_stats": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            }
        }
        
    except json.JSONDecodeError as e:
        logger.info(f"Failed to parse JSON response: {e}", exc_info=True)
        logger.info(f"Raw response content: {response.choices[0].message.content}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse model response as JSON: {str(e)}"
        )
    except ValidationError as e:
        logger.info(f"Response validation failed: {e}", exc_info=True)
        logger.info(f"Content being validated: {content}")
        raise HTTPException(
            status_code=500,
            detail=f"Model response did not match expected schema: {str(e)}"
        )
    except Exception as e:
        logger.info("Unexpected error processing response", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error processing response: {str(e)}"
        )

async def _save_evidence_to_s3(result: Dict[str, Any], document_name: str, file_name: str) -> None:
    """Save the extracted evidence to S3."""
    try:
        evidence_key = f"documents/{document_name}/extracted_evidence/{file_name.replace('.pdf', '_evidence.json')}"
        logger.info(f"Saving extracted evidence to S3: {evidence_key}")
        
        s3_client.put_object(
            Bucket=os.getenv('AWS_BUCKET_NAME'),
            Key=evidence_key,
            Body=json.dumps(result, indent=2),
            ContentType='application/json'
        )
        logger.info(f"Successfully saved evidence to S3: {evidence_key}")
    except Exception as e:
        logger.info(f"Failed to save evidence to S3: {str(e)}", exc_info=True)
        # Don't fail the request if S3 storage fails, just log the error
        pass
