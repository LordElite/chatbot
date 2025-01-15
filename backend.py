from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_model import RAGModel

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the RAGModel
rag_model = RAGModel()


@app.route("/chat", methods=[ "POST"])
def chat():
    try:
        # Get the data from the POST request
        data = request.json
        print(f"Received data: {data}")  # Debugging

        # Extract the question
        question = data.get("question")
        if not question:
            return jsonify({"error": "Please provide a valid question!"}), 400
        
        # moderation
        if  rag_model.moderate_response(question):
            response = "warning: So sorry!, but your statement has been flagged as inappropriate. Please rephrase your input and try again"
        else:
            # Process the question using RAGModel
            response = rag_model.get_response(question)
            if str("I don't know") in  str(response):
                response = rag_model.get_response(rag_model.get_new_context(question))
                
            
            print(f"Generated response: {response}, {type(response)}")  # Debugging
                
            # Return the response
            return jsonify({"answer": response})

        
    except Exception as e:
        print(f"Error occurred: {e}")  # Debugging
        return jsonify({"error": "An error occurred while processing your request."}), 500



if __name__ == "__main__":
    app.run(debug=True, port=3000)
