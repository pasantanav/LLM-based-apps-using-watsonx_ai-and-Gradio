from ibm_watson_machine_learning.foundation_models import Model
import gradio as gr

model_id = "meta-llama/llama-2-70b-chat"

my_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
}

gen_params = {
    "max_new_tokens": 512,
    "temperature": 0.7
}
project_id = "skills-network"
space_id = None
verify = False

# Initialize the model
model = Model(model_id, my_credentials, gen_params, project_id, space_id, verify)

# Function to generate a customized cover letter
def generate_cover_letter(company_name, position_name, job_description, resume_content):
    # Craft the prompt for the model to generate a cover letter
    prompt = f"Generate a customized cover letter using the company name: {company_name}, the position applied for: {position_name}, and the job description: {job_description}. Ensure the cover letter highlights my qualifications and experience as detailed in the resume content: {resume_content}. Adapt the content carefully to avoid including experiences not present in my resume but mentioned in the job description. The goal is to emphasize the alignment between my existing skills and the requirements of the role."
    generated_response = model.generate(prompt, gen_params)
    # Extract the generated text
    cover_letter = generated_response["results"][0]["generated_text"]
    return cover_letter

# Create Gradio interface for the cover letter generation application
cover_letter_app = gr.Interface(
    fn=generate_cover_letter,
    allow_flagging="never",
    inputs=[
        gr.Textbox(label="Company Name", placeholder="Enter the name of the company..."),
        gr.Textbox(label="Position Name", placeholder="Enter the name of the position..."),
        # gr.Textbox(label="Job Description Information", placeholder="Paste the job description here...", lines=10),
        gr.Textbox(label="Job Skills Keywords", placeholder="Paste the job required skills keywords here...", lines=2),
        gr.Textbox(label="Resume Content", placeholder="Paste your resume content here...", lines=10),
    ],
    outputs=gr.Textbox(label="Customized Cover Letter"),
    title="Customized Cover Letter Generator",
    description="Generate a customized cover letter by entering the company name, position name, job description and your resume."
)

cover_letter_app.launch()