import google.generativeai as genai
import os
from dotenv import load_dotenv
import pprint

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel('models/gemini-2.5-flash-image')

print("Generating image...")
try:
    response = model.generate_content("Draw a cute cat")
    
    print("\n--- Response Feedback ---")
    print(response.prompt_feedback)
    
    print("\n--- Candidates ---")
    for i, candidate in enumerate(response.candidates):
        print(f"Candidate {i}:")
        print(f"  Finish Reason: {candidate.finish_reason}")
        print(f"  Safety Ratings: {candidate.safety_ratings}")
        print(f"  Content Parts: {len(candidate.content.parts)}")
        for part in candidate.content.parts:
            print(f"    Part text: {part.text}")
            print(f"    Part inline_data mime_type: {part.inline_data.mime_type}")
            print(f"    Part inline_data data length: {len(part.inline_data.data)}")
            
except Exception as e:
    import traceback
    traceback.print_exc()
