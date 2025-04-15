import os
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Define the structure of a Business Requirements Document using Pydantic
class BusinessRequirement(BaseModel):
    id: str = Field(description="Unique identifier for the requirement")
    description: str = Field(description="Detailed description of the requirement")
    priority: str = Field(description="Priority level (High/Medium/Low)")
    stakeholders: List[str] = Field(description="List of stakeholders related to this requirement")
    acceptance_criteria: List[str] = Field(description="List of conditions to satisfy this requirement")
    
class BRDSection(BaseModel):
    title: str = Field(description="Section title")
    content: str = Field(description="Section content")
    
class BusinessRequirementsDocument(BaseModel):
    project_name: str = Field(description="Name of the project")
    project_overview: str = Field(description="Brief overview of the project")
    business_objectives: List[str] = Field(description="List of business objectives")
    scope: str = Field(description="Project scope definition")
    stakeholders: List[str] = Field(description="List of all stakeholders")
    constraints: List[str] = Field(description="List of project constraints")
    assumptions: List[str] = Field(description="List of project assumptions")
    requirements: List[BusinessRequirement] = Field(description="List of business requirements")
    additional_sections: Optional[List[BRDSection]] = Field(description="Additional document sections")

class BRDGenerator:
    def __init__(self, model_id="meta-llama/Llama-3.3-70B-Instruct"):
        """Initialize the BRD Generator with Llama3 model"""
        # Set up device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Llama3 model and tokenizer
        print(f"Loading Llama3 model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        # Create text generation pipeline
        self.text_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=2048,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True
        )
        
        # Create LangChain HuggingFacePipeline instance
        self.llm = HuggingFacePipeline(pipeline=self.text_pipeline)
        
        # Set up the output parser
        self.parser = PydanticOutputParser(pydantic_object=BusinessRequirementsDocument)
        
        # Set up the BRD generation prompt template
        self.brd_template = """
        You are an expert business analyst tasked with creating a comprehensive Business Requirements Document.
        
        Based on the following project information, create a detailed BRD:
        
        Project Name: {project_name}
        Project Description: {project_description}
        Industry: {industry}
        Target Users: {target_users}
        Project Goals: {project_goals}
        Timeline: {timeline}
        Budget Constraints: {budget}
        
        {format_instructions}
        
        Generate a comprehensive Business Requirements Document in the specified format.
        """
        
        self.prompt = PromptTemplate(
            template=self.brd_template,
            input_variables=[
                "project_name", 
                "project_description", 
                "industry", 
                "target_users", 
                "project_goals", 
                "timeline", 
                "budget"
            ],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        # Create the LLMChain
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def generate_brd(self, project_info):
        """
        Generate a Business Requirements Document based on project information
        
        Args:
            project_info (dict): Dictionary containing project information
            
        Returns:
            BusinessRequirementsDocument: The generated BRD
        """
        try:
            # Generate raw BRD content
            raw_brd = self.chain.run(**project_info)
            
            # Parse the output into our structured format
            brd = self.parser.parse(raw_brd)
            
            return brd
        except Exception as e:
            print(f"Error generating BRD: {str(e)}")
            raise
    
    def generate_brd_as_dict(self, project_info):
        """Generate BRD and return as a dictionary for API responses"""
        brd = self.generate_brd(project_info)
        return brd.dict()
    
    def generate_brd_as_markdown(self, project_info):
        """Generate BRD and convert to markdown format"""
        brd = self.generate_brd(project_info)
        
        # Convert to markdown
        markdown = f"# Business Requirements Document: {brd.project_name}\n\n"
        
        markdown += f"## Project Overview\n{brd.project_overview}\n\n"
        
        markdown += "## Business Objectives\n"
        for i, objective in enumerate(brd.business_objectives, 1):
            markdown += f"{i}. {objective}\n"
        markdown += "\n"
        
        markdown += f"## Scope\n{brd.scope}\n\n"
        
        markdown += "## Stakeholders\n"
        for stakeholder in brd.stakeholders:
            markdown += f"- {stakeholder}\n"
        markdown += "\n"
        
        markdown += "## Constraints\n"
        for constraint in brd.constraints:
            markdown += f"- {constraint}\n"
        markdown += "\n"
        
        markdown += "## Assumptions\n"
        for assumption in brd.assumptions:
            markdown += f"- {assumption}\n"
        markdown += "\n"
        
        markdown += "## Business Requirements\n"
        for req in brd.requirements:
            markdown += f"### {req.id}: {req.description}\n"
            markdown += f"**Priority:** {req.priority}\n\n"
            markdown += "**Stakeholders:**\n"
            for s in req.stakeholders:
                markdown += f"- {s}\n"
            markdown += "\n**Acceptance Criteria:**\n"
            for ac in req.acceptance_criteria:
                markdown += f"- {ac}\n"
            markdown += "\n"
        
        if brd.additional_sections:
            for section in brd.additional_sections:
                markdown += f"## {section.title}\n{section.content}\n\n"
        
        return markdown

# Example usage
if __name__ == "__main__":
    # Initialize the BRD generator
    brd_generator = BRDGenerator()
    
    # Example project information
    project_info = {
        "project_name": "E-Commerce Platform Upgrade",
        "project_description": "Upgrade the existing e-commerce platform to improve user experience, add new payment methods, and optimize for mobile devices.",
        "industry": "Retail",
        "target_users": "Online shoppers aged 18-65",
        "project_goals": "Increase conversion rate by 15%, reduce cart abandonment by 20%, improve mobile user experience",
        "timeline": "6 months",
        "budget": "$250,000"
    }
    
    # Generate BRD in markdown format
    brd_markdown = brd_generator.generate_brd_as_markdown(project_info)
    
    # Save the generated BRD to a file
    with open("generated_brd.md", "w") as f:
        f.write(brd_markdown)
    
    print("BRD generated and saved to 'generated_brd.md'")