const modelSelect = document.getElementById('model-select');
const inputSection = document.querySelector('.input-section');
const predictionDisplay = document.getElementById('prediction-display');
const visualizationDisplay = document.getElementById('visualization-display');

// Model-specific input handlers
const inputHandlers = {
    mnist: createMNISTInput,
    text: createTextInput,
    audio: createAudioInput,
    image: createImageInput
};

// Model-specific visualization handlers
const visualizationHandlers = {
    mnist: showMNISTVisualizations,
    text: showTextVisualizations,
    audio: showAudioVisualizations,
    image: showImageVisualizations
};

modelSelect.addEventListener('change', (e) => {
    const modelType = e.target.value;
    if (inputHandlers[modelType]) {
        inputHandlers[modelType]();
    }
});

function predict(modelName, input) {
    fetch(`/predict/${modelName}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) throw new Error(data.error);
        
        const handler = visualizationHandlers[modelName];
        if (handler) {
            handler(data.predictions, data.visualizations);
        }
    })
    .catch(error => console.error('Prediction error:', error));
}