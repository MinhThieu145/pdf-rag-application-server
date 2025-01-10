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
    
    Args:
        request (EssayGenerationRequest): Contains context, topic, and word count for the essay
        
    Returns:
        DetailedEssayResponse: Contains essay planning and detailed essay structure
        
    Raises:
        HTTPException: If there's an error in essay generation or API communication
    """
    try:
        # Ensure OpenAI API key is set
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")

        client = OpenAI(api_key=openai_api_key)

        # Define the prompt
        prompt = f"""
You are an advanced AI tasked with writing an argumentative essay on a given topic. Your goal is to produce a well-written, engaging essay that demonstrates college-level argumentation while using language that feels natural for a high school student. 

First, review the following information:

Context to use as evidence (do not use any information outside of this):
<context>
{request.context}
</context>

Topic for your essay:
<topic>
{request.topic}
</topic>

Required word count:
<word_count>
{request.word_count}
</word_count>

Before writing the essay, wrap your brainstorming process inside <essay_planning> tags. In your essay planning:
1. Analyze the topic and context
2. Develop a clear, strong thesis statement
3. Outline the main arguments and supporting points
4. Identify key evidence from the context that supports each main point
5. Plan the structure of your essay, including how to transition between ideas
6. Consider potential counterarguments and how to address them
7. Brainstorm ways to make the essay relatable to high school students
8. List any relevant examples or anecdotes that could engage the target audience

After brainstorming, outline the structure of your essay with detailed explanations. Wrap this outline inside <essay_structure> tags. In your essay structure:
1. **Introduction**
   - **Content:** [Your introduction text]
   - **Purpose:** [Explanation of what the introduction does]
2. **Body Paragraphs**
   - **Paragraph 1**
     - **Content:** [Text of paragraph 1]
     - **Purpose:** [Explanation of paragraph 1's role]
   - **Paragraph 2**
     - **Content:** [Text of paragraph 2]
     - **Purpose:** [Explanation of paragraph 2's role]
   - **Paragraph 3**
     - **Content:** [Text of paragraph 3]
     - **Purpose:** [Explanation of paragraph 3's role]
   <!-- Add more paragraphs as needed -->
3. **Conclusion**
   - **Content:** [Your conclusion text]
   - **Purpose:** [Explanation of what the conclusion does]

Ensure that each section is clearly labeled and that the purpose of each part of the essay is explicitly stated.
"""

        # Make the API call
        response = client.chat.completions.create(
            model="gpt-4",  # Using standard GPT-4 model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=request.word_count * 3  # Adjust as needed
        )

        # Extract and parse the response
        content = response.choices[0].message.content

        try:
            # Extract essay planning
            try:
                essay_planning = content.split("<essay_planning>")[1].split("</essay_planning>")[0].strip()
            except IndexError:
                print("Failed to extract essay planning. Content:", content)
                raise HTTPException(status_code=500, detail="Failed to extract essay planning")

            # Extract essay structure
            try:
                structure_text = content.split("<essay_structure>")[1].split("</essay_structure>")[0].strip()
            except IndexError:
                print("Failed to extract essay structure. Content:", content)
                raise HTTPException(status_code=500, detail="Failed to extract essay structure")

            # Parse introduction
            try:
                intro_text = structure_text.split("**Introduction**")[1].split("**Body Paragraphs**")[0].strip()
                intro_content = intro_text.split("**Content:**")[1].split("**Purpose:**")[0].strip()
                intro_purpose = intro_text.split("**Purpose:**")[1].strip()
                introduction = Introduction(
                    content=intro_content,
                    purpose=intro_purpose
                )
            except IndexError:
                print("Failed to parse introduction. Structure text:", structure_text)
                raise HTTPException(status_code=500, detail="Failed to parse introduction")

            # Extract body paragraphs
            try:
                body_parts = structure_text.split("**Body Paragraphs**")[1].split("**Conclusion**")[0].split("**Paragraph")
                body_paragraphs = []
                for i, part in enumerate(body_parts[1:], 1):  # Skip the first empty split
                    try:
                        content = part.split("**Content:**")[1].split("**Purpose:**")[0].strip()
                        purpose = part.split("**Purpose:**")[1].split("**Paragraph" if i < len(body_parts)-1 else "**Conclusion")[0].strip()
                        body_paragraphs.append(Paragraph(
                            paragraph_number=i,
                            content=content,
                            purpose=purpose
                        ))
                    except IndexError:
                        print(f"Failed to parse body paragraph {i}. Part:", part)
                        continue  # Skip malformed paragraphs
            except IndexError:
                print("Failed to parse body paragraphs. Structure text:", structure_text)
                raise HTTPException(status_code=500, detail="Failed to parse body paragraphs")

            # Parse conclusion
            try:
                conclusion_text = structure_text.split("**Conclusion**")[1].strip()
                conclusion_content = conclusion_text.split("**Content:**")[1].split("**Purpose:**")[0].strip()
                conclusion_purpose = conclusion_text.split("**Purpose:**")[1].strip()
                conclusion = Conclusion(
                    content=conclusion_content,
                    purpose=conclusion_purpose
                )
            except IndexError:
                print("Failed to parse conclusion. Structure text:", structure_text)
                raise HTTPException(status_code=500, detail="Failed to parse conclusion")

            # Create the final response
            essay_structure = EssayStructure(
                introduction=introduction,
                body_paragraphs=body_paragraphs,
                conclusion=conclusion
            )

            return DetailedEssayResponse(
                essay_planning=essay_planning,
                essay_structure=essay_structure
            )

        except HTTPException:
            raise
        except Exception as e:
            print("Unexpected error:", str(e))
            raise HTTPException(status_code=500, detail=str(e))

    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
