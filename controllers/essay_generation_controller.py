"""
Essay Generation Controller

This module handles the generation of argumentative essays using GPT-4.
It provides endpoints for generating well-structured essays based on provided context and topics.

Key Features:
- Essay structure generation with introduction, body paragraphs, and conclusion
- Evidence-based argumentation using provided context
- Detailed planning and organization
- Word count control

Environment Variables Required:
- OPENAI_API_KEY: OpenAI API key
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError
from typing import List, Optional
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI router
router = APIRouter(tags=["Essay Generation"])

# Define Pydantic models
class Paragraph(BaseModel):
    paragraph_number: int
    content: str
    purpose: str

class Introduction(BaseModel):
    content: str
    purpose: str

class Conclusion(BaseModel):
    content: str
    purpose: str

class EssayStructure(BaseModel):
    introduction: Introduction
    body_paragraphs: List[Paragraph]
    conclusion: Conclusion

class DetailedEssayResponse(BaseModel):
    essay_planning: str
    essay_structure: EssayStructure

class EssayGenerationRequest(BaseModel):
    context: str
    topic: str
    word_count: int

@router.get("/test")
async def test_endpoint():
    return {"message": "Essay generation router is working"}

@router.post("/generate", response_model=DetailedEssayResponse)
async def generate_essay(request: EssayGenerationRequest) -> DetailedEssayResponse:
    """
    Generate a detailed argumentative essay based on provided context and topic.
    """
    try:
        print("\n=== Essay Generation Request ===")
        print(f"Topic: {request.topic}")
        print(f"Word Count: {request.word_count}")
        print("\nContext:")
        print("----------------------------------------")
        print(request.context)
        print("----------------------------------------\n")
        
        print(f"Received request with context length: {len(request.context)}, topic: {request.topic}, word_count: {request.word_count}")
        
        # Ensure OpenAI API key is set
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")

        client = OpenAI(api_key=openai_api_key)

        # Make the API call
        response = client.chat.completions.create(
            model="gpt-4o",  # Note: Using gpt-4 as gpt-4o-2024-08-06 isn't available yet
            messages=[
                {
                    "role": "developer", 
                    "content": '''

You are an expert at analyzing topics and evidence to create detailed, compelling argumentative essay plans. Your task is to think critically about the given topic and evidence, brainstorming the most effective essay structure, arguments, and use of evidence, while considering a specified word count. Ensure that the essay plan is persuasive, balanced, and well-structured.

Input Format
Context:

Evidence: A list of key pieces of information, facts, or findings relevant to the topic.
Reasoning: A detailed explanation of how the evidence supports or informs the topic. Include any limitations or alternative interpretations of the evidence.
Topic: The refined essay topic or question being addressed.
Word Count: The total number of words allowed for the essay. This will guide paragraph planning and detail depth.

Instructions
Analyze the Evidence:

Carefully evaluate each piece of evidence.
Identify multiple ways the evidence can support the argument (e.g., as a primary argument, a counterargument, or background context).
Explore alternative interpretations or limitations of the evidence to create a balanced discussion.
Plan Paragraph Structure and Content:

Consider the word count to estimate the number of paragraphs needed.
Assign approximate word counts to the introduction, body paragraphs, and conclusion.
Decide what each paragraph will focus on, ensuring each has a distinct purpose and logical flow.
Develop a Persuasive Argument:

Identify a clear thesis statement that aligns with the evidence and reasoning.
Use the evidence to construct logical, emotionally engaging, and authoritative arguments.
Integrate counterarguments seamlessly, addressing them in ways that strengthen the overall essay.
Outline the Structure:

Divide the essay into sections: introduction, body paragraphs, and conclusion.
For each paragraph, describe:
Its purpose (e.g., introducing the argument, presenting evidence, rebutting a counterargument).
The main idea, evidence, and reasoning to be used.
Ensure the outline reflects a persuasive, balanced essay that flows logically from start to finish.
Output Format
Brainstorming:

Main Ideas: List several potential arguments or themes for the essay.
Connections: Describe how the evidence can be tied to these main ideas.
Counterarguments: Identify potential opposing views and propose rebuttals to strengthen the argument.
Essay Length Plan:

Total Word Count: Specify the word count provided.
Estimated Paragraphs: Determine the number of paragraphs based on the word count and complexity of the topic.
Word Count Distribution:
Introduction: Approximate word count for the introduction.
Body Paragraphs: Word count per paragraph, with reasoning for distribution.
Conclusion: Approximate word count for the conclusion.
Proposed Essay Outline:

Introduction:
Purpose: Ideas for the hook, context, and thesis statement.
Key Focus: What the introduction will achieve within the word limit.
Body Paragraphs:
Paragraph 1: Main idea, evidence, reasoning, and purpose.
Paragraph 2: Main idea, evidence, reasoning, and purpose.
Paragraph 3 (if applicable): Main idea, evidence, reasoning, and purpose.
Conclusion:
Purpose: How the conclusion will tie together the essay’s arguments.
Key Focus: What message the conclusion will leave with the reader.
Guidelines for Originality
Avoid generic or overly simplified responses. Focus on tailoring the essay plan to the nuances of the evidence and topic.
Incorporate creative approaches where appropriate, such as unique hooks, engaging examples, or surprising connections.


'''
                },
                {
                    "role": "user", 
                    "content": f"Evidence with Reasoning:\n{request.context}\n\nTopic: {request.topic}\n\nWord Count: {request.word_count}"
                }
            ],
            temperature=0.5,
            max_tokens=request.word_count * 3
        )

        print("Received response from GPT-4")
        content = response.choices[0].message.content
        print("Response content:", content)

        # Return the content directly
        return DetailedEssayResponse(
            essay_planning=content,
            essay_structure=EssayStructure(
                introduction=Introduction(content="", purpose=""),
                body_paragraphs=[],
                conclusion=Conclusion(content="", purpose="")
            )
        )

    except Exception as e:
        print(f"Error in essay generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate essay: {str(e)}")
