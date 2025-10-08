document.addEventListener("DOMContentLoaded", () => {
    const textInput = document.getElementById("text-input");
    const analyzeTextBtn = document.getElementById("analyze-text-btn");
    const fileInput = document.getElementById("file-input");
    const analyzeFileBtn = document.getElementById("analyze-file-btn");
    const resultContainer = document.getElementById("result-container");
    const resultText = document.getElementById("result-text");

    // Function to display the result
    const displayResult = (sentiment) => {
        resultText.textContent = sentiment;
        resultText.className = ""; // Clear previous classes
        resultText.classList.add(sentiment.toLowerCase());
        resultContainer.style.display = "block";
    };

    // Function to handle API request for text
    const analyzeText = async (text) => {
        if (!text.trim()) {
            alert("Please enter some text to analyze.");
            return;
        }

        try {
            const response = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text }),
            });

            if (!response.ok) {
                throw new Error("An error occurred while fetching the sentiment.");
            }

            const data = await response.json();
            displayResult(data.sentiment);
        } catch (error) {
            console.error("Error:", error);
            alert(error.message);
        }
    };

    // Function to handle API request for file
    const analyzeFile = async (file) => {
        if (!file) {
            alert("Please select a file to analyze.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("http://127.0.0.1:5000/predict_file", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || "An error occurred while analyzing the file.");
            }

            const data = await response.json();
            displayResult(data.sentiment);
        } catch (error) {
            console.error("Error:", error);
            alert(error.message);
        }
    };

    // Event listeners
    analyzeTextBtn.addEventListener("click", () => {
        analyzeText(textInput.value);
    });

    analyzeFileBtn.addEventListener("click", () => {
        analyzeFile(fileInput.files[0]);
    });
});