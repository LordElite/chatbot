// DOM Elements
const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");

// Function to append messages to the chat
function appendMessage(message, sender) {
  const messageDiv = document.createElement("div");
  messageDiv.classList.add("message", sender);
  messageDiv.textContent = message;
  chatBox.appendChild(messageDiv);
  chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to the bottom
}

// Function to handle user input
async function handleUserInput() {
  const userMessage = userInput.value.trim();
  if (!userMessage) return; // Don't send empty messages

  // Append user message to chat
  appendMessage(userMessage, "user");
  userInput.value = ""; // Clear input box

  // Call chatbot backend 
  try {
    const response = await fetch("http://localhost:3000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question: userMessage }),
    });

    const data = await response.json();
    appendMessage(data.answer, "bot"); // Append bot response to chat
  } catch (error) {
    appendMessage("Sorry, something went wrong. Please try again.", "bot");
    console.error("Error:", error);
  }
}

// Add event listener to the Send button
sendButton.addEventListener("click", handleUserInput);

// Allow "Enter" key to send messages
userInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    handleUserInput();
  }
});
